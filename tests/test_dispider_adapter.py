from __future__ import annotations

from types import SimpleNamespace
from typing import Sequence, Tuple

import pytest
import torch

import dispider.streaming.dispider_adapter as adapter_module
from dispider.streaming.dispider_adapter import (
    DecisionKVCacheCapability,
    DecisionKVCacheUnavailable,
    DecisionOutput,
    DispiderClipState,
    DispiderStreamingAdapter,
    RELEASED_CACHE_CAPABILITY,
    ReleasedPerceptionDecisionReactionBackend,
)
from dispider.streaming.types import SampledWindow


class FakeBackend:
    cache_capability = DecisionKVCacheCapability(
        supported=False,
        stable_prefix_safe=True,
        pending_suffix_committable=False,
        reason="test backend has no KV implementation",
    )

    def __init__(self, scores: Sequence[float]) -> None:
        self.scores = list(scores)
        self.score_calls = []
        self.generate_calls = []
        self.reset_calls = 0

    def perceive_clip(self, window: SampledWindow):
        return (window.frames[-1], window.timestamp_s)

    def decide(
        self,
        clips: Sequence[DispiderClipState],
        *,
        question: str,
        committed_trigger_positions: Tuple[int, ...],
        use_cache: bool,
        verify_with_oracle: bool,
    ) -> DecisionOutput:
        self.score_calls.append(
            (
                tuple(clip.position for clip in clips),
                question,
                committed_trigger_positions,
                use_cache,
                verify_with_oracle,
            )
        )
        score = self.scores[len(clips) - 1]
        positions = committed_trigger_positions
        if score > 0:
            positions += (len(clips),)
        return DecisionOutput(score, positions, kv_cache_used=False)

    def react(
        self,
        clips: Sequence[DispiderClipState],
        *,
        question: str,
        committed_trigger_positions: Tuple[int, ...],
    ) -> str:
        self.generate_calls.append(
            (
                tuple(clip.position for clip in clips),
                question,
                committed_trigger_positions,
            )
        )
        return "answer-%d" % len(clips)

    def reset(self) -> None:
        self.reset_calls += 1


def window(position: int) -> SampledWindow:
    start = float((position - 1) * 16)
    timestamps = tuple(start + value for value in range(16))
    return SampledWindow(
        start_s=start,
        end_s=start + 16.0,
        frames=tuple("%d:%d" % (position, value) for value in range(16)),
        timestamps_s=timestamps,
        complete=True,
    )


def decide(adapter, clip_window):
    perception = adapter.perceive(clip_window)
    triggered = adapter.should_respond(
        perception,
        window=clip_window,
        history=(),
    )
    return perception, triggered


def test_adapter_tracks_stable_prefix_and_disposable_pending_suffix() -> None:
    backend = FakeBackend([-1.0, 0.5, 0.75])
    adapter = DispiderStreamingAdapter(backend, "What happened?")

    _, first_triggered = decide(adapter, window(1))
    second, second_triggered = decide(adapter, window(2))
    second_answer = adapter.respond(second, window=window(2), history=())
    third, third_triggered = decide(adapter, window(3))
    third_answer = adapter.respond(third, window=window(3), history=())

    assert not first_triggered
    assert second_triggered and third_triggered
    assert (second_answer, third_answer) == ("answer-2", "answer-3")
    assert adapter.committed_trigger_positions == (2, 3)
    assert adapter.pending_clip_positions == ()

    first, second_trace, third_trace = adapter.decisions
    assert first.pending_suffix_clip_positions == (1,)
    assert second_trace.stable_prefix_trigger_positions_before == ()
    assert second_trace.pending_suffix_clip_positions == (1, 2)
    assert second_trace.stable_prefix_trigger_positions_after == (2,)
    assert third_trace.stable_prefix_trigger_positions_before == (2,)
    assert third_trace.pending_suffix_clip_positions == (3,)
    assert third_trace.stable_prefix_trigger_positions_after == (2, 3)
    assert not any(trace.kv_cache_used for trace in adapter.decisions)

    assert backend.generate_calls == [
        ((1, 2), "What happened?", ()),
        ((1, 2, 3), "What happened?", (2,)),
    ]


def test_shared_adapter_produces_timestamped_scheduler_events() -> None:
    backend = FakeBackend([-1.0, 1.0])
    adapter = DispiderStreamingAdapter(backend, "Question")
    session = adapter.new_session(window_seconds=16.0, frames_per_window=16)

    timestamps = tuple(float(value) for value in range(32))
    events = session.push_frames(timestamps, timestamps)
    events.extend(session.flush())

    assert [event.as_dict() for event in events] == [
        {"timestamp_s": 31.0, "answer": "answer-2"}
    ]
    assert [trace.score for trace in adapter.decisions] == [-1.0, 1.0]


def test_released_backend_refuses_unimplemented_decision_kv_cache() -> None:
    backend = FakeBackend([-1.0])

    with pytest.raises(DecisionKVCacheUnavailable) as error:
        DispiderStreamingAdapter(
            backend,
            "Question",
            enable_decision_kv_cache=True,
        )

    message = str(error.value)
    assert "past_key_values" in message
    assert "stable" in message
    assert backend.cache_capability.stable_prefix_safe
    assert not backend.cache_capability.pending_suffix_committable


def test_supported_backend_receives_cache_and_oracle_flags() -> None:
    class CachedBackend(FakeBackend):
        cache_capability = RELEASED_CACHE_CAPABILITY

        def decide(self, clips, **kwargs):
            output = super().decide(clips, **kwargs)
            return DecisionOutput(
                output.score,
                output.observed_trigger_positions,
                kv_cache_used=True,
            )

    backend = CachedBackend([1.0])
    adapter = DispiderStreamingAdapter(
        backend,
        "Question",
        enable_decision_kv_cache=True,
        verify_cache_with_oracle=True,
    )

    perception, triggered = decide(adapter, window(1))
    adapter.respond(perception, window=window(1), history=())

    assert triggered
    assert backend.score_calls[0][-2:] == (True, True)
    assert adapter.decisions[0].kv_cache_used


def test_failed_score_does_not_commit_clip_state() -> None:
    class FailingBackend(FakeBackend):
        def decide(self, *args, **kwargs):
            raise RuntimeError("model failed")

    adapter = DispiderStreamingAdapter(FailingBackend([1.0]), "Question")
    clip_window = window(1)
    perception = adapter.perceive(clip_window)

    with pytest.raises(RuntimeError, match="model failed"):
        adapter.should_respond(perception, window=clip_window, history=())

    assert adapter.clips == ()
    assert adapter.decisions == ()


def test_stale_and_cross_adapter_perceptions_are_rejected() -> None:
    first = DispiderStreamingAdapter(FakeBackend([-1.0]), "Question")
    second = DispiderStreamingAdapter(FakeBackend([-1.0]), "Question")
    clip_window = window(1)
    perception = first.perceive(clip_window)
    first.should_respond(perception, window=clip_window, history=())

    with pytest.raises(RuntimeError, match="stale"):
        first.should_respond(perception, window=clip_window, history=())
    with pytest.raises(ValueError, match="different adapter"):
        second.should_respond(perception, window=clip_window, history=())


def test_trigger_must_be_answered_before_next_clip() -> None:
    adapter = DispiderStreamingAdapter(FakeBackend([1.0, -1.0]), "Question")
    decide(adapter, window(1))

    with pytest.raises(RuntimeError, match="must be answered"):
        adapter.perceive(window(2))


def test_reset_clears_video_state_and_retains_loaded_backend() -> None:
    backend = FakeBackend([-1.0])
    adapter = DispiderStreamingAdapter(backend, "Question")
    decide(adapter, window(1))

    adapter.reset()

    assert adapter.clips == ()
    assert adapter.decisions == ()
    assert adapter.committed_trigger_positions == ()
    assert backend.reset_calls == 1
    assert adapter.perceive(window(1)).clip.position == 1


def test_backend_position_drift_is_rejected_transactionally() -> None:
    class DriftingBackend(FakeBackend):
        def decide(self, clips, **kwargs):
            return DecisionOutput(-1.0, (1,))

    adapter = DispiderStreamingAdapter(DriftingBackend([-1.0]), "Question")
    clip_window = window(1)
    perception = adapter.perceive(clip_window)

    with pytest.raises(RuntimeError, match="disagree"):
        adapter.should_respond(perception, window=clip_window, history=())

    assert adapter.clips == ()
    assert adapter.decisions == ()


class FakeTokenizer:
    bos_token_id = 1
    eos_token_id = 9
    pad_token_id = 0
    pad_token = "<pad>"

    def __call__(self, text, return_tensors=None):
        token_ids = [self.bos_token_id, len(text) % 7 + 2]
        if text == "<|im_end|>":
            token_ids = [7, 8]
        if return_tensors == "pt":
            token_ids = torch.tensor([token_ids])
        return SimpleNamespace(input_ids=token_ids)

    def batch_decode(self, sequences, skip_special_tokens):
        assert skip_special_tokens
        assert sequences.shape[0] == 1
        return ["  generated answer  "]


class FakePerceptionDecisionModel:
    def __init__(self):
        self.positions = []
        self.scores = []
        self.calls = []
        self.eval_calls = 0

    def eval(self):
        self.eval_calls += 1

    def forward_inference(self, **kwargs):
        self.calls.append(kwargs)
        return self.positions, self.scores


class FakeReactionModel:
    def __init__(self, perception_decision_model):
        self.config = SimpleNamespace(mm_use_im_start_end=False)
        self.device = torch.device("cpu")
        self.generated = []
        self.eval_calls = 0
        self.container = SimpleNamespace(
            decision=perception_decision_model,
            device=torch.device("cpu"),
            dtype=torch.float32,
            select_layer=77,
        )

    def get_perception_decision(self):
        return self.container

    def eval(self):
        self.eval_calls += 1

    def generate(self, input_ids, **kwargs):
        self.generated.append((input_ids, kwargs))
        return torch.tensor([[4, 5]])


def released_runtime():
    decision_model = FakePerceptionDecisionModel()
    reaction_model = FakeReactionModel(decision_model)
    tokenizer = FakeTokenizer()
    runtime = ReleasedPerceptionDecisionReactionBackend(
        reaction_model,
        tokenizer,
        image_processor=object(),
        time_tokenizer=FakeTokenizer(),
    )
    return runtime, reaction_model, decision_model


def released_clip(position):
    payload = adapter_module._ReleasedClipPayload(
        pixel_values=torch.zeros(1, 16, 3, 2, 2),
        time_ids=torch.tensor([1, -200, 2]),
    )
    return DispiderClipState(
        position=position,
        start_s=float((position - 1) * 16),
        end_s=float(position * 16),
        timestamp_s=float(position * 16 - 1),
        payload=payload,
    )


def test_released_decision_checks_recomputed_history() -> None:
    runtime, _, decision_model = released_runtime()
    clips = (released_clip(1), released_clip(2))
    decision_model.positions = [2]
    decision_model.scores = [-0.5, 0.25]

    result = runtime.decide(
        clips,
        question="Question",
        committed_trigger_positions=(),
        use_cache=False,
        verify_with_oracle=False,
    )
    runtime.decide(
        clips,
        question="Question",
        committed_trigger_positions=(),
        use_cache=False,
        verify_with_oracle=False,
    )

    assert result == DecisionOutput(0.25, (2,), kv_cache_used=False)
    assert len(decision_model.calls) == 2
    call = decision_model.calls[0]
    assert call["images"].shape == (2, 16, 3, 2, 2)
    assert "select_layer" not in call
    assert "use_cache" not in call

    decision_model.positions = [1, 2]
    with pytest.raises(RuntimeError, match="changed earlier decisions"):
        runtime.decide(
            clips,
            question="Question",
            committed_trigger_positions=(),
            use_cache=False,
            verify_with_oracle=False,
        )


def test_released_cached_decision_tracks_state_and_resets(monkeypatch) -> None:
    runtime, _, _ = released_runtime()

    class StubBackend:
        def __init__(self, model, *, select_layer):
            self.model = model
            self.select_layer = select_layer

        def summarize(self, pending):
            return ("summary", pending)

    class StubCache:
        def __init__(self, backend, **kwargs):
            self.backend = backend
            self.kwargs = kwargs
            self.calls = []

        def observe(self, block, *, summarize):
            self.calls.append((block, summarize))
            triggered = len(self.calls) == 1
            score = 0.5 if triggered else -0.5
            return SimpleNamespace(
                score=score,
                triggered=triggered,
                cached_score=score,
                oracle_score=score,
                used_oracle_for_decision=False,
            )

    monkeypatch.setattr(adapter_module, "DecisionKVBackend", StubBackend)
    monkeypatch.setattr(adapter_module, "DecisionCache", StubCache)
    runtime._encode_cached_clip = lambda clip, question: (
        (clip.position, question),
        "answer",
        "question",
        "todo",
    )

    first = runtime.decide(
        (released_clip(1),),
        question="Question",
        committed_trigger_positions=(),
        use_cache=True,
        verify_with_oracle=True,
    )
    second = runtime.decide(
        (released_clip(1), released_clip(2)),
        question="Question",
        committed_trigger_positions=(1,),
        use_cache=True,
        verify_with_oracle=True,
    )

    assert first.score == 0.5
    assert first.observed_trigger_positions == (1,)
    assert first.kv_cache_used
    assert second.score == -0.5
    assert second.observed_trigger_positions == (1,)
    assert second.kv_cache_used
    assert runtime._decision_cache.kwargs["verify_with_oracle"]
    assert runtime._decision_kv_backend.select_layer == 77
    runtime.reset()
    assert runtime._decision_cache is None
    assert runtime._decision_cache_clip_count == 0
    assert runtime._decision_cache_trigger_positions == ()


def test_cached_clip_encoder_extracts_perception_and_delimiters() -> None:
    class PreparedDecisionModel(FakePerceptionDecisionModel):
        def __init__(self):
            super().__init__()
            self.decoder_kwargs = None

        def prepare_inference_inputs(self, **kwargs):
            del kwargs
            input_rows = [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]]
            inputs = torch.tensor([input_rows])
            indicators = torch.tensor([[100, 100, 200, 0]])
            question = torch.tensor([[[5.0, 5.0], [6.0, 6.0]]])
            question_mask = torch.tensor([[True, False]])
            answer = torch.tensor([[[7.0, 7.0]]])
            todo = torch.tensor([[[8.0, 8.0]]])
            return (
                None,
                None,
                torch.ones(1, 4, dtype=torch.bool),
                None,
                inputs,
                None,
                question,
                question_mask,
                indicators,
                answer,
                todo,
            )

        def get_model(self):
            owner = self

            class Decoder:
                def __call__(self, **kwargs):
                    owner.decoder_kwargs = kwargs
                    hidden = kwargs["inputs_embeds"] + 10
                    return SimpleNamespace(last_hidden_state=hidden)

            return Decoder()

    decision_model = PreparedDecisionModel()
    runtime = ReleasedPerceptionDecisionReactionBackend(
        FakeReactionModel(decision_model),
        FakeTokenizer(),
        image_processor=object(),
        time_tokenizer=FakeTokenizer(),
    )
    runtime._model_inputs = lambda clips, question, device, **kwargs: (
        torch.zeros(1, 16, 3, 2, 2),
        torch.tensor([[1, 2]]),
        torch.ones(1, 2, dtype=torch.bool),
        torch.tensor([[3, 4]]),
        torch.ones(1, 2, dtype=torch.bool),
    )
    runtime._delimiter_tokens = lambda device: (
        torch.tensor([[5]]),
        torch.tensor([[6]]),
    )

    perception, answer, question, todo = runtime._encode_cached_clip(
        released_clip(2), "Question"
    )

    assert perception.clip_memory.tolist() == [[11.0, 11.0], [12.0, 12.0]]
    assert perception.time_state.tolist() == [[13.0, 13.0]]
    assert answer.embeddings.tolist() == [[7.0, 7.0]]
    assert question.embeddings.tolist() == [[5.0, 5.0]]
    assert todo.embeddings.tolist() == [[8.0, 8.0]]
    assert decision_model.decoder_kwargs["use_cache"] is False
    assert decision_model.decoder_kwargs["select_layer"] == 77


def test_reaction_uses_one_large_frame_and_public_stop_token() -> None:
    runtime, reaction_model, _ = released_runtime()
    clips = (released_clip(1), released_clip(2))

    answer = runtime.react(
        clips,
        question="Question",
        committed_trigger_positions=(1,),
    )

    assert answer == "generated answer"
    _, kwargs = reaction_model.generated[0]
    assert kwargs["images"].shape == (2, 16, 3, 2, 2)
    assert kwargs["images_large"].shape == (2, 1, 3, 2, 2)
    assert kwargs["ans_position"] == [1]
    assert kwargs["use_cache"] is True
    criteria = kwargs["stopping_criteria"]
    assert len(criteria) == 1
    assert criteria(torch.tensor([[3, 7, 8]]), None)
    assert not criteria(torch.tensor([[3, 7]]), None)


def test_from_pretrained_forwards_device_and_device_map(
    monkeypatch,
) -> None:
    calls = []
    _, reaction_model, _ = released_runtime()
    tokenizer = FakeTokenizer()
    processors = (object(), FakeTokenizer())

    def loader(path, base, name, **kwargs):
        calls.append((path, base, name, kwargs))
        return tokenizer, reaction_model, processors, 2048

    monkeypatch.setenv("HOME", "/tmp/dispider-home")
    runtime = ReleasedPerceptionDecisionReactionBackend.from_pretrained(
        "~/weights/Dispider",
        device="cpu",
        device_map={"": "cpu"},
        loader=loader,
        local_files_only=True,
    )

    assert runtime.reaction_model is reaction_model
    path, base, name, kwargs = calls[0]
    assert path == "/tmp/dispider-home/weights/Dispider"
    assert base is None
    assert name == "Dispider"
    assert kwargs == {
        "device": "cpu",
        "device_map": {"": "cpu"},
        "local_files_only": True,
    }
