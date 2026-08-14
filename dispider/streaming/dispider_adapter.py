"""Adapter from released Dispider checkpoints to the streaming scheduler.

The exact no-cache path remains available through ``forward_inference``.  The
optional cached path runs Perception once per clip and persists only the
Decision transformer's stable prefix.  Pending clip memory is evaluated on a
cloned KV branch and discarded; cross-clip summaries are always uncached.
"""

from __future__ import annotations

import math
import os
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Protocol, Sequence, Tuple

from .decision_cache import DecisionCache
from .kv_backend import (
    DecisionKVBackend,
    PerceptionDecisionBlock,
)
from .session import StreamingSession
from .types import SampledWindow, WindowRecord


_CACHE_INTERFACE = (
    "build a cache for the stable [ANS, Q, summaries..., ANS] prefix",
    "fork that cache before appending pending memory, Q, and TODO",
    "return both the updated past_key_values and the TODO hidden state",
    "score the same sequence without cache as an equality oracle",
)


@dataclass(frozen=True)
class DecisionKVCacheCapability:
    """What a checkpoint backend can safely cache during decision scoring."""

    supported: bool
    stable_prefix_safe: bool
    pending_suffix_committable: bool
    reason: str
    required_interface: Tuple[str, ...] = _CACHE_INTERFACE


RELEASED_CACHE_CAPABILITY = DecisionKVCacheCapability(
    supported=True,
    stable_prefix_safe=True,
    pending_suffix_committable=False,
    reason=(
        "stable Decision prefixes use tuple KV tensors; every pending suffix "
        "runs on an isolated clone and cross-clip summaries remain uncached"
    ),
)


class DecisionKVCacheUnavailable(RuntimeError):
    """Raised when KV caching is requested from an unsupported backend."""


@dataclass(frozen=True)
class DecisionOutput:
    """One score returned by the Decision model."""

    score: float
    observed_trigger_positions: Optional[Tuple[int, ...]] = None
    kv_cache_used: bool = False
    cached_score: Optional[float] = None
    oracle_score: Optional[float] = None
    used_oracle_for_decision: bool = False


@dataclass(frozen=True)
class DispiderClipState:
    """One preprocessed 16-frame clip retained by the adapter."""

    position: int
    start_s: float
    end_s: float
    timestamp_s: float
    payload: Any = field(repr=False, compare=False)


@dataclass(frozen=True)
class DispiderPerception:
    """Perception value passed from the scheduler to decision and reaction."""

    clip: DispiderClipState
    _owner: object = field(repr=False, compare=False)


@dataclass(frozen=True)
class DispiderDecisionTrace:
    """Auditable stable-prefix and pending-suffix state for one clip."""

    clip_position: int
    timestamp_s: float
    score: float
    triggered: bool
    stable_prefix_trigger_positions_before: Tuple[int, ...]
    pending_suffix_clip_positions: Tuple[int, ...]
    stable_prefix_trigger_positions_after: Tuple[int, ...]
    kv_cache_used: bool
    cached_score: Optional[float] = None
    oracle_score: Optional[float] = None
    used_oracle_for_decision: bool = False


class PerceptionDecisionReactionBackend(Protocol):
    """Paper-level model operations used by the streaming adapter."""

    cache_capability: DecisionKVCacheCapability

    def perceive_clip(self, window: SampledWindow) -> Any: ...

    def decide(
        self,
        clips: Sequence[DispiderClipState],
        *,
        question: str,
        committed_trigger_positions: Tuple[int, ...],
        use_cache: bool,
        verify_with_oracle: bool,
    ) -> DecisionOutput: ...

    def react(
        self,
        clips: Sequence[DispiderClipState],
        *,
        question: str,
        committed_trigger_positions: Tuple[int, ...],
    ) -> str: ...


class DispiderStreamingAdapter:
    """Shared perception, decision, and reaction adapter for one video stream.

    Pass the same instance as all three components of ``StreamingSession``, or
    use :meth:`new_session`.  Trigger positions are one-based clip positions,
    matching the released model's ``ans_position`` convention.
    """

    def __init__(
        self,
        runtime: PerceptionDecisionReactionBackend,
        question: str,
        *,
        threshold: float = 0.0,
        enable_decision_kv_cache: bool = False,
        verify_cache_with_oracle: bool = False,
    ) -> None:
        if not isinstance(question, str) or not question.strip():
            raise ValueError("question must be a non-empty string")
        if not math.isfinite(threshold):
            raise ValueError("threshold must be finite")
        native_threshold = getattr(runtime, "native_decision_threshold", None)
        if native_threshold is not None and not math.isclose(
            threshold, float(native_threshold), rel_tol=0.0, abs_tol=0.0
        ):
            raise ValueError(
                "threshold must match the released checkpoint threshold "
                f"({native_threshold})"
            )

        capability = runtime.cache_capability
        if enable_decision_kv_cache and not capability.supported:
            requirements = "; ".join(capability.required_interface)
            raise DecisionKVCacheUnavailable(
                f"decision KV cache is unavailable: {capability.reason}. "
                f"A cache backend must {requirements}."
            )
        if verify_cache_with_oracle and not enable_decision_kv_cache:
            raise ValueError(
                "cache oracle verification requires enable_decision_kv_cache"
            )

        self.runtime = runtime
        self.question = question.strip()
        self.threshold = float(threshold)
        self.enable_decision_kv_cache = enable_decision_kv_cache
        self.verify_cache_with_oracle = verify_cache_with_oracle

        self._lock = threading.RLock()
        self._owner = object()
        self._clips = []
        self._decisions = []
        self._committed_trigger_positions: Tuple[int, ...] = ()
        self._last_decision = None

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        question: str,
        *,
        threshold: float = 0.0,
        enable_decision_kv_cache: bool = False,
        verify_cache_with_oracle: bool = False,
        **load_kwargs: Any,
    ) -> "DispiderStreamingAdapter":
        """Load a local/Hugging Face checkpoint with the public loader.

        Offline inference remains available through :class:`VideoStream`.
        """

        runtime_type = ReleasedPerceptionDecisionReactionBackend
        runtime = runtime_type.from_pretrained(model_path, **load_kwargs)
        return cls(
            runtime,
            question,
            threshold=threshold,
            enable_decision_kv_cache=enable_decision_kv_cache,
            verify_cache_with_oracle=verify_cache_with_oracle,
        )

    @property
    def cache_capability(self) -> DecisionKVCacheCapability:
        return self.runtime.cache_capability

    @property
    def clips(self) -> Tuple[DispiderClipState, ...]:
        with self._lock:
            return tuple(self._clips)

    @property
    def decisions(self) -> Tuple[DispiderDecisionTrace, ...]:
        with self._lock:
            return tuple(self._decisions)

    @property
    def committed_trigger_positions(self) -> Tuple[int, ...]:
        with self._lock:
            return self._committed_trigger_positions

    @property
    def pending_clip_positions(self) -> Tuple[int, ...]:
        with self._lock:
            start = (
                self._committed_trigger_positions[-1] + 1
                if self._committed_trigger_positions
                else 1
            )
            return tuple(range(start, len(self._clips) + 1))

    def new_session(self, **session_kwargs: Any) -> StreamingSession:
        """Create a scheduler using this instance for all model components."""

        return StreamingSession(self, self, self, **session_kwargs)

    def perceive(self, window: SampledWindow) -> DispiderPerception:
        """Preprocess one clip without committing it to decision history."""

        with self._lock:
            if self._has_unanswered_trigger():
                raise RuntimeError(
                    "the triggered clip must be answered before perception"
                )
            payload = self.runtime.perceive_clip(window)
            clip = DispiderClipState(
                position=len(self._clips) + 1,
                start_s=window.start_s,
                end_s=window.end_s,
                timestamp_s=window.timestamp_s,
                payload=payload,
            )
            return DispiderPerception(clip=clip, _owner=self._owner)

    def should_respond(
        self,
        perception: DispiderPerception,
        *,
        window: SampledWindow,
        history: Sequence[WindowRecord],
    ) -> bool:
        """Score and atomically commit one newly perceived clip."""

        del history
        with self._lock:
            self._validate_perception(perception, window)
            if self._has_unanswered_trigger():
                raise RuntimeError("answer the triggered clip before scoring")

            candidate_clips = tuple(self._clips) + (perception.clip,)
            result = self.runtime.decide(
                candidate_clips,
                question=self.question,
                committed_trigger_positions=self._committed_trigger_positions,
                use_cache=self.enable_decision_kv_cache,
                verify_with_oracle=self.verify_cache_with_oracle,
            )
            score = float(result.score)
            if not math.isfinite(score):
                raise ValueError("checkpoint decision score must be finite")
            if result.kv_cache_used and not self.enable_decision_kv_cache:
                raise RuntimeError(
                    "backend reported KV-cache use while caching is disabled"
                )

            triggered = score > self.threshold
            position = perception.clip.position
            self._validate_observed_positions(
                result.observed_trigger_positions, position, triggered
            )

            stable_before = self._committed_trigger_positions
            pending_start = stable_before[-1] + 1 if stable_before else 1
            pending = tuple(range(pending_start, position + 1))
            if triggered:
                stable_after = stable_before + (position,)
            else:
                stable_after = stable_before
            trace = DispiderDecisionTrace(
                clip_position=position,
                timestamp_s=perception.clip.timestamp_s,
                score=score,
                triggered=triggered,
                stable_prefix_trigger_positions_before=stable_before,
                pending_suffix_clip_positions=pending,
                stable_prefix_trigger_positions_after=stable_after,
                kv_cache_used=result.kv_cache_used,
                cached_score=result.cached_score,
                oracle_score=result.oracle_score,
                used_oracle_for_decision=result.used_oracle_for_decision,
            )

            self._clips.append(perception.clip)
            self._decisions.append(trace)
            self._committed_trigger_positions = stable_after
            self._last_decision = [perception, trace, False]
            return triggered

    def respond(
        self,
        perception: DispiderPerception,
        *,
        window: SampledWindow,
        history: Sequence[WindowRecord],
    ) -> str:
        """Generate for the pending suffix that caused the latest trigger."""

        del history
        with self._lock:
            self._validate_perception_owner(perception)
            if self._last_decision is None:
                raise RuntimeError("respond called before a decision")
            latest_perception, trace, already_answered = self._last_decision
            if perception is not latest_perception:
                raise RuntimeError("respond must use the latest perception")
            if perception.clip.timestamp_s != window.timestamp_s:
                raise ValueError("window does not match the latest perception")
            if not trace.triggered:
                raise RuntimeError("respond called for a silent decision")
            if already_answered:
                raise RuntimeError("the latest trigger was already answered")

            answer = self.runtime.react(
                tuple(self._clips),
                question=self.question,
                committed_trigger_positions=(
                    trace.stable_prefix_trigger_positions_before
                ),
            )
            if not isinstance(answer, str):
                raise TypeError("Reaction must return a string")
            self._last_decision[2] = True
            return answer

    def reset(self) -> None:
        """Clear per-video state while retaining the loaded checkpoint."""

        with self._lock:
            reset = getattr(self.runtime, "reset", None)
            if callable(reset):
                reset()
            self._clips.clear()
            self._decisions.clear()
            self._committed_trigger_positions = ()
            self._last_decision = None

    def _validate_perception(
        self, perception: DispiderPerception, window: SampledWindow
    ) -> None:
        self._validate_perception_owner(perception)
        expected_position = len(self._clips) + 1
        if perception.clip.position != expected_position:
            raise RuntimeError("stale or already committed perception")
        if (
            perception.clip.start_s != window.start_s
            or perception.clip.end_s != window.end_s
            or perception.clip.timestamp_s != window.timestamp_s
        ):
            raise ValueError("window does not match perception")

    def _validate_perception_owner(
        self,
        perception: DispiderPerception,
    ) -> None:
        if not isinstance(perception, DispiderPerception):
            raise TypeError("perception was not produced by this adapter")
        if perception._owner is not self._owner:
            raise ValueError("perception belongs to a different adapter")

    def _validate_observed_positions(
        self,
        observed: Optional[Tuple[int, ...]],
        current_position: int,
        triggered: bool,
    ) -> None:
        if observed is None:
            return
        normalized = tuple(int(value) for value in observed)
        if normalized != tuple(sorted(set(normalized))):
            raise RuntimeError("unordered checkpoint trigger positions")
        outside_range = normalized and (
            normalized[0] < 1 or normalized[-1] > current_position
        )
        if outside_range:
            raise RuntimeError("checkpoint returned an invalid position")
        observed_before = tuple(
            value for value in normalized if value < current_position
        )
        if observed_before != self._committed_trigger_positions:
            raise RuntimeError(
                "checkpoint decision history diverged from adapter state"
            )
        if (current_position in normalized) != triggered:
            raise RuntimeError(
                "checkpoint trigger positions disagree with its latest score"
            )

    def _has_unanswered_trigger(self) -> bool:
        if self._last_decision is None:
            return False
        _, trace, answered = self._last_decision
        return trace.triggered and not answered


@dataclass(frozen=True)
class _ReleasedClipPayload:
    pixel_values: Any
    time_ids: Any


class ReleasedPerceptionDecisionReactionBackend:
    """Released Perception/Decision and Reaction model integration."""

    cache_capability = RELEASED_CACHE_CAPABILITY
    native_decision_threshold = 0.0

    def __init__(
        self,
        reaction_model: Any,
        tokenizer: Any,
        image_processor: Any,
        time_tokenizer: Any,
        *,
        generation_kwargs: Optional[Dict[str, Any]] = None,
        cache_oracle_atol: float = 0.125,
        cache_fallback_margin: float = 0.125,
    ) -> None:
        perception_decision = reaction_model.get_perception_decision()
        perception_decision_model = perception_decision.decision
        if perception_decision_model is None or not callable(
            getattr(perception_decision_model, "forward_inference", None)
        ):
            raise TypeError("Perception/Decision needs forward_inference")
        if not callable(getattr(reaction_model, "generate", None)):
            raise TypeError("released Reaction model requires generate")

        self.reaction_model = reaction_model
        self.perception_decision_model = perception_decision_model
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.time_tokenizer = time_tokenizer
        self.perception_decision = perception_decision
        self.generation_kwargs = dict(generation_kwargs or {})
        if not math.isfinite(cache_oracle_atol) or cache_oracle_atol < 0:
            raise ValueError("cache oracle tolerance must be nonnegative")
        if not math.isfinite(cache_fallback_margin) or cache_fallback_margin < 0:
            raise ValueError("cache fallback margin must be nonnegative")
        self.cache_oracle_atol = float(cache_oracle_atol)
        self.cache_fallback_margin = float(cache_fallback_margin)
        self._question_ids: Dict[str, Any] = {}
        self._prompt_ids: Dict[str, Any] = {}
        self._decision_cache = None
        self._decision_kv_backend = None
        self._decision_cache_question = None
        self._decision_cache_verify = None
        self._decision_cache_clip_count = 0
        self._decision_cache_trigger_positions: Tuple[int, ...] = ()

        if getattr(self.time_tokenizer, "pad_token", None) is None:
            self.time_tokenizer.pad_token = "<pad>"
        eval_reaction = getattr(self.reaction_model, "eval", None)
        if callable(eval_reaction):
            eval_reaction()
        eval_decision = getattr(self.perception_decision_model, "eval", None)
        if callable(eval_decision):
            eval_decision()

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        *,
        device: str = "cuda",
        device_map: Any = "auto",
        generation_kwargs: Optional[Dict[str, Any]] = None,
        cache_oracle_atol: float = 0.125,
        cache_fallback_margin: float = 0.125,
        loader: Any = None,
        **loader_kwargs: Any,
    ) -> "ReleasedPerceptionDecisionReactionBackend":
        """Load through ``dispider.model.builder.load_pretrained_model``."""

        expanded_path = os.path.expanduser(model_path)
        if loader is None:
            from dispider.mm_utils import get_model_name_from_path
            from dispider.model.builder import load_pretrained_model

            loader = load_pretrained_model
            model_name = get_model_name_from_path(expanded_path)
        else:
            model_name = os.path.basename(expanded_path.rstrip(os.sep))
        tokenizer, model, processors, _ = loader(
            expanded_path,
            None,
            model_name,
            device=device,
            device_map=device_map,
            **loader_kwargs,
        )
        try:
            image_processor, time_tokenizer = processors
        except (TypeError, ValueError) as error:
            raise TypeError(
                "public loader must return (image_processor, time_tokenizer)"
            ) from error
        return cls(
            model,
            tokenizer,
            image_processor,
            time_tokenizer,
            generation_kwargs=generation_kwargs,
            cache_oracle_atol=cache_oracle_atol,
            cache_fallback_margin=cache_fallback_margin,
        )

    def perceive_clip(self, window: SampledWindow) -> _ReleasedClipPayload:
        """Preprocess exactly one checkpoint-native 16-frame clip."""

        if len(window.frames) != 16:
            raise ValueError("released Dispider checkpoints require 16 frames")
        frames = self._to_square_pil_frames(window.frames)
        preprocess = self.image_processor.preprocess
        processed = preprocess(frames, return_tensors="pt")
        try:
            pixel_values = processed["pixel_values"]
        except (KeyError, TypeError):
            pixel_values = processed.pixel_values
        if getattr(pixel_values, "ndim", None) != 4:
            raise ValueError("image processor must return [frames, C, H, W]")
        output_shape = (1, len(frames), *pixel_values.shape[1:])
        pixel_values = pixel_values.reshape(output_shape)

        end_s = window.end_s if window.complete else window.timestamps_s[-1]
        sentence = "This contains a clip sampled in %d to %d seconds" % (
            int(round(window.start_s)),
            int(round(end_s)),
        )
        from dispider.constants import DEFAULT_IMAGE_TOKEN
        from dispider.mm_utils import tokenizer_image_token

        time_ids = tokenizer_image_token(
            sentence + DEFAULT_IMAGE_TOKEN,
            self.time_tokenizer,
            return_tensors="pt",
        )
        return _ReleasedClipPayload(pixel_values, time_ids)

    def decide(
        self,
        clips: Sequence[DispiderClipState],
        *,
        question: str,
        committed_trigger_positions: Tuple[int, ...],
        use_cache: bool,
        verify_with_oracle: bool,
    ) -> DecisionOutput:
        """Score a new clip with the cached or exact released Decision path."""

        if use_cache:
            return self._decide_with_kv_cache(
                clips,
                question=question,
                committed_trigger_positions=committed_trigger_positions,
                verify_with_oracle=verify_with_oracle,
            )
        if verify_with_oracle:
            raise ValueError("cache oracle requires use_cache=True")
        if self._decision_cache is not None:
            raise RuntimeError("Decision cache policy changed mid-stream")
        torch = self._torch()
        images, seqs, compress_mask, qs, qs_mask = self._model_inputs(
            clips, question, self._decision_device()
        )
        ans_token, todo_token = self._delimiter_tokens(self._decision_device())
        run_decision = self.perception_decision_model.forward_inference
        with torch.inference_mode():
            positions, scores = run_decision(
                input_ids=seqs,
                attention_mask=compress_mask,
                qs_ids=qs,
                qs_mask=qs_mask,
                images=images,
                ans_token=ans_token,
                todo_token=todo_token,
            )
        normalized_scores = tuple(self._to_float(value) for value in scores)
        if len(normalized_scores) != len(clips):
            raise RuntimeError(
                "forward_inference returned one score per clip incorrectly"
            )
        normalized_positions = tuple(int(value) for value in positions)
        previous = list(normalized_positions)
        if previous and previous[-1] == len(clips):
            previous.pop()
        previous = tuple(previous)
        if previous != committed_trigger_positions:
            raise RuntimeError(
                "released checkpoint recomputation changed earlier decisions"
            )
        return DecisionOutput(
            score=normalized_scores[-1],
            observed_trigger_positions=normalized_positions,
            kv_cache_used=False,
        )

    def _decide_with_kv_cache(
        self,
        clips: Sequence[DispiderClipState],
        *,
        question: str,
        committed_trigger_positions: Tuple[int, ...],
        verify_with_oracle: bool,
    ) -> DecisionOutput:
        if len(clips) != self._decision_cache_clip_count + 1:
            raise RuntimeError("cached clips must arrive one at a time")
        cached_positions = self._decision_cache_trigger_positions
        if committed_trigger_positions != cached_positions:
            raise RuntimeError("cached Decision trigger history diverged")
        if self._decision_cache_question not in (None, question):
            raise RuntimeError("Decision question changed mid-stream")

        block, answer, question_block, todo = self._encode_cached_clip(
            clips[-1], question
        )
        if self._decision_cache is None:
            backend = DecisionKVBackend(
                self.perception_decision_model,
                select_layer=getattr(
                    self.perception_decision,
                    "select_layer",
                    100,
                ),
            )
            state = DecisionCache(
                backend,
                ans=answer,
                question=question_block,
                todo=todo,
                threshold=self.native_decision_threshold,
                verify_with_oracle=verify_with_oracle,
                oracle_atol=self.cache_oracle_atol,
                oracle_fallback_margin=self.cache_fallback_margin,
            )
            self._decision_kv_backend = backend
            self._decision_cache = state
            self._decision_cache_question = question
            self._decision_cache_verify = verify_with_oracle
        elif verify_with_oracle != self._decision_cache_verify:
            raise RuntimeError("cannot change cache oracle policy mid-stream")

        result = self._decision_cache.observe(
            block,
            summarize=self._decision_kv_backend.summarize,
        )
        position = len(clips)
        observed = committed_trigger_positions
        if result.triggered:
            observed += (position,)
        self._decision_cache_clip_count = position
        self._decision_cache_trigger_positions = observed
        return DecisionOutput(
            score=result.score,
            observed_trigger_positions=observed,
            kv_cache_used=True,
            cached_score=result.cached_score,
            oracle_score=result.oracle_score,
            used_oracle_for_decision=result.used_oracle_for_decision,
        )

    def _encode_cached_clip(self, clip: DispiderClipState, question: str):
        """Run one Perception step and return typed Decision blocks."""

        torch = self._torch()
        device = self._decision_device()
        images, seqs, compress_mask, qs, qs_mask = self._model_inputs(
            (clip,),
            question,
            device,
            first_position=clip.position,
        )
        ans_token, todo_token = self._delimiter_tokens(device)
        model = self.perception_decision_model
        prepare = model.prepare_inference_inputs
        with torch.inference_mode():
            prepared = prepare(
                input_ids=seqs,
                position_ids=None,
                attention_mask=compress_mask,
                question_ids=qs,
                question_mask=qs_mask,
                past_key_values=None,
                images=images,
                answer_token=ans_token,
                todo_token=todo_token,
            )
        if not isinstance(prepared, tuple) or len(prepared) != 11:
            raise RuntimeError("unexpected Perception preparation output")
        (
            _,
            position_ids,
            attention_mask,
            _,
            inputs_embeds,
            _,
            question_embeds,
            prepared_question_mask,
            indicators,
            answer_embeds,
            todo_embeds,
        ) = prepared
        decoder = self.perception_decision_model.get_model()
        select_layer = getattr(
            self.perception_decision,
            "select_layer",
            100,
        )
        with torch.inference_mode():
            outputs = decoder(
                input_ids=None,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=None,
                indicators=indicators,
                use_cache=False,
                output_attentions=False,
                output_hidden_states=False,
                return_dict=True,
                select_layer=select_layer,
            )
        hidden = getattr(outputs, "last_hidden_state", None)
        if hidden is None:
            hidden = outputs[0]
        clip_memory = hidden[indicators == 100].detach()
        time_state = hidden[indicators == 200].detach()
        if clip_memory.shape[0] < 1 or time_state.shape[0] < 1:
            raise RuntimeError("Perception omitted memory or time state")

        question_mask = prepared_question_mask[0].bool()
        return (
            PerceptionDecisionBlock.perception(clip_memory, time_state),
            PerceptionDecisionBlock.answer(answer_embeds[0].detach()),
            PerceptionDecisionBlock.question(
                question_embeds[0][question_mask].detach()
            ),
            PerceptionDecisionBlock.todo(todo_embeds[0].detach()),
        )

    def react(
        self,
        clips: Sequence[DispiderClipState],
        *,
        question: str,
        committed_trigger_positions: Tuple[int, ...],
    ) -> str:
        """Generate using stable triggers before the current pending suffix."""

        torch = self._torch()
        device = self._outer_device()
        images, seqs, compress_mask, qs, qs_mask = self._model_inputs(
            clips, question, device
        )
        input_ids = self._prompt_token_ids(question).unsqueeze(0).to(device)
        ans_token, todo_token = self._delimiter_tokens(device)

        kwargs = {
            "do_sample": False,
            "max_new_tokens": 1024,
            "pad_token_id": self.tokenizer.eos_token_id,
            "stopping_criteria": self._stopping_criteria(device),
            "use_cache": True,
        }
        kwargs.update(self.generation_kwargs)
        with torch.inference_mode():
            output_ids = self.reaction_model.generate(
                input_ids,
                images=images,
                images_large=images[:, :1].contiguous(),
                seqs=seqs,
                compress_mask=compress_mask,
                qs=qs,
                qs_mask=qs_mask,
                ans_token=ans_token,
                todo_token=todo_token,
                q_id=None,
                insert_position=0,
                ans_position=list(committed_trigger_positions),
                **kwargs,
            )
        sequences = getattr(output_ids, "sequences", output_ids)
        batch_decode = self.tokenizer.batch_decode
        answer = batch_decode(sequences, skip_special_tokens=True)[0]
        return answer.strip()

    def reset(self) -> None:
        """Drop all per-stream Decision state while retaining model weights."""

        self._decision_cache = None
        self._decision_kv_backend = None
        self._decision_cache_question = None
        self._decision_cache_verify = None
        self._decision_cache_clip_count = 0
        self._decision_cache_trigger_positions = ()

    def _model_inputs(
        self,
        clips,
        question,
        device,
        *,
        first_position=1,
    ):
        torch = self._torch()
        if not clips:
            raise ValueError("at least one clip is required")
        payloads = []
        for expected_position, clip in enumerate(
            clips,
            start=first_position,
        ):
            if clip.position != expected_position:
                raise ValueError("clip positions must be contiguous")
            if not isinstance(clip.payload, _ReleasedClipPayload):
                raise TypeError("clip payload belongs to another backend")
            payloads.append(clip.payload)

        pixel_values = [payload.pixel_values for payload in payloads]
        images = torch.cat(pixel_values, dim=0)
        images = images.to(
            device=device,
            dtype=self._decision_dtype(),
            non_blocking=True,
        )
        seqs = torch.nn.utils.rnn.pad_sequence(
            [payload.time_ids for payload in payloads],
            batch_first=True,
            padding_value=self.time_tokenizer.pad_token_id,
        ).to(device=device, non_blocking=True)
        compress_mask = seqs.ne(self.time_tokenizer.pad_token_id)
        question_ids = self._question_token_ids(question)
        qs = torch.nn.utils.rnn.pad_sequence(
            [question_ids],
            batch_first=True,
            padding_value=self.time_tokenizer.pad_token_id,
        ).to(device=device, non_blocking=True)
        qs_mask = qs.ne(self.time_tokenizer.pad_token_id)
        return images, seqs, compress_mask, qs, qs_mask

    def _question_token_ids(self, question: str):
        if question not in self._question_ids:
            from dispider.constants import DEFAULT_TODO_TOKEN
            from dispider.mm_utils import tokenizer_image_token

            self._question_ids[question] = tokenizer_image_token(
                question + DEFAULT_TODO_TOKEN,
                self.time_tokenizer,
                return_tensors="pt",
            )
        return self._question_ids[question]

    def _prompt_token_ids(self, question: str):
        if question not in self._prompt_ids:
            from dispider.constants import (
                DEFAULT_IMAGE_TOKEN,
                DEFAULT_IM_END_TOKEN,
                DEFAULT_IM_START_TOKEN,
                IMAGE_TOKEN_INDEX,
            )
            from dispider.conversation import conv_templates
            from dispider.mm_utils import tokenizer_image_token

            if getattr(
                self.reaction_model.config,
                "mm_use_im_start_end",
                False,
            ):
                image = DEFAULT_IM_START_TOKEN
                image += DEFAULT_IMAGE_TOKEN
                image += DEFAULT_IM_END_TOKEN
            else:
                image = DEFAULT_IMAGE_TOKEN
            conversation = conv_templates["qwen"].copy()
            user_message = image + "\n" + question
            conversation.append_message(conversation.roles[0], user_message)
            conversation.append_message(conversation.roles[1], None)
            self._prompt_ids[question] = tokenizer_image_token(
                conversation.get_prompt(),
                self.tokenizer,
                IMAGE_TOKEN_INDEX,
                return_tensors="pt",
            )
        return self._prompt_ids[question]

    def _delimiter_tokens(self, device):
        from dispider.constants import DEFAULT_ANS_TOKEN, DEFAULT_TODO_TOKEN

        ans_tokens = self.time_tokenizer(
            DEFAULT_ANS_TOKEN, return_tensors="pt"
        ).input_ids
        ans = ans_tokens.to(device=device, non_blocking=True)
        todo = self.time_tokenizer(
            DEFAULT_TODO_TOKEN, return_tensors="pt"
        ).input_ids.to(device=device, non_blocking=True)
        return ans, todo

    def _stopping_criteria(self, device):
        torch = self._torch()
        from transformers import StoppingCriteria, StoppingCriteriaList

        stop_ids = self.tokenizer("<|im_end|>").input_ids
        stop = torch.tensor(stop_ids, device=device)

        class TokenSequenceStop(StoppingCriteria):
            def __call__(self, input_ids, scores, **kwargs):
                del scores, kwargs
                if input_ids.shape[1] < stop.shape[0]:
                    return False
                stop_length = stop.shape[0]
                suffix = input_ids[0, -stop_length:]
                return torch.equal(suffix, stop)

        return StoppingCriteriaList([TokenSequenceStop()])

    def _decision_device(self):
        return getattr(
            self.perception_decision,
            "device",
            getattr(self.reaction_model, "device", "cuda"),
        )

    def _outer_device(self):
        return getattr(self.reaction_model, "device", self._decision_device())

    def _decision_dtype(self):
        torch = self._torch()
        return getattr(
            self.perception_decision,
            "dtype",
            torch.float16,
        )

    @staticmethod
    def _to_float(value):
        item = getattr(value, "item", None)
        return float(item() if callable(item) else value)

    @staticmethod
    def _torch():
        import torch

        return torch

    @classmethod
    def _to_square_pil_frames(cls, frames):
        import numpy as np
        from PIL import Image

        arrays = [np.asarray(frame) for frame in frames]
        shapes = {array.shape for array in arrays}
        if len(shapes) != 1:
            raise ValueError("all frames in a clip must have the same shape")
        array = np.stack(arrays, axis=0)
        if array.ndim != 4 or array.shape[-1] not in (3, 4):
            raise ValueError("frames must have shape [H, W, 3 or 4]")
        height, width = array.shape[1:3]
        if height != width:
            torch = cls._torch()
            tensor = torch.from_numpy(array).permute(0, 3, 1, 2).float()
            side = min(height, width)
            tensor = torch.nn.functional.interpolate(tensor, size=(side, side))
            array = tensor.permute(0, 2, 3, 1).to(torch.uint8).numpy()
        elif array.dtype != np.uint8:
            array = array.astype(np.uint8)
        return tuple(Image.fromarray(frame) for frame in array)


__all__ = [
    "DecisionOutput",
    "DecisionKVCacheCapability",
    "DecisionKVCacheUnavailable",
    "DispiderClipState",
    "DispiderDecisionTrace",
    "DispiderPerception",
    "DispiderStreamingAdapter",
    "PerceptionDecisionReactionBackend",
    "RELEASED_CACHE_CAPABILITY",
    "ReleasedPerceptionDecisionReactionBackend",
]
