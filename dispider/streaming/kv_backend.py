"""Safe KV caching for Dispider's Perception/Decision transformer.

Only the stable ``ANS, Q, (summary, ANS)*`` prefix is persistent.  Pending
clip memory is evaluated on an isolated tuple-cache branch and discarded.
Summaries are computed separately from ``all clip memory, all time state``;
that cross-clip aggregation is never appended to the decision cache.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn


class DecisionBlockRole(str, Enum):
    """Semantic role of one Perception/Decision input block."""

    ANSWER = "answer"
    QUESTION = "question"
    PERCEPTION = "perception"
    SUMMARY = "summary"
    TODO = "todo"


@dataclass(frozen=True)
class PerceptionDecisionBlock:
    """A typed block consumed by the Perception/Decision transformer.

    Perception blocks retain both model outputs from a clip.  ``clip_memory``
    is used by Decision scoring.  ``time_state`` is used only when a triggered
    range is summarized.  Other roles carry their transformer inputs in
    ``embeddings``.
    """

    role: DecisionBlockRole
    embeddings: Optional[Tensor] = None
    clip_memory: Optional[Tensor] = None
    time_state: Optional[Tensor] = None

    def __post_init__(self) -> None:
        role = DecisionBlockRole(self.role)
        object.__setattr__(self, "role", role)
        if role is DecisionBlockRole.PERCEPTION:
            if self.embeddings is not None:
                raise ValueError("Perception blocks cannot carry embeddings")
            _validate_states(self.clip_memory, "clip_memory")
            _validate_states(self.time_state, "time_state")
            if self.clip_memory.shape[-1] != self.time_state.shape[-1]:
                raise ValueError("clip_memory and time_state hidden sizes must match")
            return

        _validate_states(self.embeddings, f"{role.value} embeddings")
        if self.clip_memory is not None or self.time_state is not None:
            raise ValueError(f"{role.value} blocks cannot carry Perception state")

    @classmethod
    def answer(cls, embeddings: Tensor) -> "PerceptionDecisionBlock":
        return cls(DecisionBlockRole.ANSWER, embeddings=embeddings)

    @classmethod
    def question(cls, embeddings: Tensor) -> "PerceptionDecisionBlock":
        return cls(DecisionBlockRole.QUESTION, embeddings=embeddings)

    @classmethod
    def perception(
        cls, clip_memory: Tensor, time_state: Tensor
    ) -> "PerceptionDecisionBlock":
        return cls(
            DecisionBlockRole.PERCEPTION,
            clip_memory=clip_memory,
            time_state=time_state,
        )

    @classmethod
    def summary(cls, embeddings: Tensor) -> "PerceptionDecisionBlock":
        return cls(DecisionBlockRole.SUMMARY, embeddings=embeddings)

    @classmethod
    def todo(cls, embeddings: Tensor) -> "PerceptionDecisionBlock":
        return cls(DecisionBlockRole.TODO, embeddings=embeddings)

    def decision_embeddings(self) -> Tensor:
        """Return inputs used in a Decision prefix or speculative suffix."""

        if self.role is DecisionBlockRole.PERCEPTION:
            assert self.clip_memory is not None
            return self.clip_memory
        assert self.embeddings is not None
        return self.embeddings


LayerCache = Tuple[Tensor, ...]
TupleCache = Tuple[LayerCache, ...]


@dataclass(frozen=True)
class DecisionPrefixCache:
    """Immutable-by-contract cache for one stable Decision prefix."""

    past_key_values: TupleCache
    attention_mask: Tensor
    sequence_length: int


class DecisionKVBackend:
    """DecisionCache backend for the released custom Qwen2 decoder."""

    def __init__(
        self,
        perception_decision_model: nn.Module,
        *,
        decision_head: Optional[nn.Module] = None,
        select_layer: Optional[int] = 100,
    ) -> None:
        if not isinstance(perception_decision_model, nn.Module):
            raise TypeError("perception_decision_model must be a torch module")
        self.perception_decision_model = perception_decision_model
        self._decoder = self._resolve_decoder(perception_decision_model)
        self._decision_head = decision_head or self._resolve_decision_head(
            perception_decision_model
        )
        if not callable(self._decision_head):
            raise TypeError("Perception/Decision model needs a decision head")
        self._select_layer = select_layer
        self.perception_decision_model.eval()
        if decision_head is not None:
            decision_head.eval()

    def build_prefix_cache(
        self, prefix: Tuple[PerceptionDecisionBlock, ...]
    ) -> DecisionPrefixCache:
        """Build a fresh persistent cache for a stable prefix."""

        self._validate_stable_prefix(prefix)
        embeddings = self._flatten_decision(prefix)
        outputs = self._run_decoder(
            embeddings,
            past_key_values=None,
            prefix_mask=None,
            use_cache=True,
        )
        cache = self._output_tuple_cache(outputs)
        expected_length = embeddings.shape[0]
        self._validate_cache_length(cache, expected_length)
        mask = torch.ones(
            (1, expected_length),
            dtype=torch.bool,
            device=embeddings.device,
        )
        return DecisionPrefixCache(cache, mask, expected_length)

    def fork_prefix_cache(self, cache: DecisionPrefixCache) -> DecisionPrefixCache:
        """Clone all KV tensors for a speculative branch."""

        self._validate_prefix_cache(cache)
        return DecisionPrefixCache(
            _clone_tuple_cache(cache.past_key_values),
            cache.attention_mask.clone(),
            cache.sequence_length,
        )

    def score_cached(
        self,
        cache: DecisionPrefixCache,
        suffix: Tuple[PerceptionDecisionBlock, ...],
    ) -> float:
        """Score ``pending clip memory, Q, TODO`` without committing it."""

        self._validate_pending_suffix(suffix)
        self._validate_prefix_cache(cache)
        embeddings = self._flatten_decision(suffix)

        outputs = self._run_decoder(
            embeddings,
            past_key_values=cache.past_key_values,
            prefix_mask=cache.attention_mask,
            use_cache=True,
        )
        updated = self._output_tuple_cache(outputs)
        expected_length = cache.sequence_length + embeddings.shape[0]
        self._validate_cache_length(updated, expected_length)
        return self._decision_score(self._output_hidden_states(outputs))

    def score_uncached(self, sequence: Tuple[PerceptionDecisionBlock, ...]) -> float:
        """Run the complete Decision sequence as the equality oracle."""

        self._validate_complete_sequence(sequence)
        embeddings = self._flatten_decision(sequence)
        outputs = self._run_decoder(
            embeddings,
            past_key_values=None,
            prefix_mask=None,
            use_cache=False,
        )
        return self._decision_score(self._output_hidden_states(outputs))

    def summarize(
        self, pending: Tuple[PerceptionDecisionBlock, ...]
    ) -> PerceptionDecisionBlock:
        """Summarize a triggered range from ``all M, all T``.

        This call deliberately disables caching.  Adding a new clip inserts
        memory before earlier time tokens, so reusing an aggregation cache
        would assign incorrect RoPE positions.
        """

        if not pending or any(
            block.role is not DecisionBlockRole.PERCEPTION for block in pending
        ):
            raise ValueError("summaries require one or more Perception blocks")
        self._validate_compatible_blocks(pending)
        memories = [block.clip_memory for block in pending]
        time_states = [block.time_state for block in pending]
        if any(state is None for state in memories + time_states):
            raise RuntimeError("invalid Perception block state")
        memory = torch.cat(memories, dim=0)
        time_state = torch.cat(time_states, dim=0)
        aggregation = torch.cat((memory, time_state), dim=0)
        outputs = self._run_decoder(
            aggregation,
            past_key_values=None,
            prefix_mask=None,
            use_cache=False,
        )
        hidden = self._output_hidden_states(outputs)
        summary = hidden[0, -time_state.shape[0] :].detach()
        return PerceptionDecisionBlock.summary(summary)

    def _run_decoder(
        self,
        embeddings: Tensor,
        *,
        past_key_values: Optional[TupleCache],
        prefix_mask: Optional[Tensor],
        use_cache: bool,
    ) -> Any:
        if embeddings.ndim != 2 or embeddings.shape[0] < 1:
            raise ValueError("decoder embeddings must have shape [tokens, hidden]")
        prefix_length = (
            _tuple_cache_length(past_key_values) if past_key_values is not None else 0
        )
        if prefix_mask is None:
            if prefix_length:
                raise ValueError("cached decoding requires a prefix mask")
            prefix_mask = torch.empty(
                (1, 0), dtype=torch.bool, device=embeddings.device
            )
        self._validate_mask(prefix_mask, prefix_length, embeddings.device)

        suffix_mask = torch.ones(
            (1, embeddings.shape[0]),
            dtype=prefix_mask.dtype,
            device=embeddings.device,
        )
        attention_mask = torch.cat((prefix_mask, suffix_mask), dim=1)
        position_ids = torch.arange(
            prefix_length,
            prefix_length + embeddings.shape[0],
            dtype=torch.long,
            device=embeddings.device,
        )
        indicators = torch.zeros_like(suffix_mask, dtype=torch.long)
        indicators[:, -1] = 1
        kwargs = {
            "input_ids": None,
            "inputs_embeds": embeddings.unsqueeze(0),
            "attention_mask": attention_mask,
            "position_ids": position_ids.unsqueeze(0),
            "cache_position": position_ids,
            "past_key_values": past_key_values,
            "indicators": indicators,
            "use_cache": use_cache,
            "output_attentions": False,
            "output_hidden_states": False,
            "return_dict": True,
        }
        if self._select_layer is not None:
            kwargs["select_layer"] = self._select_layer
        with torch.inference_mode():
            return self._decoder(**kwargs)

    def _decision_score(self, hidden_states: Tensor) -> float:
        if hidden_states.ndim != 3 or hidden_states.shape[0] != 1:
            raise RuntimeError("decoder returned invalid hidden states")
        todo_hidden = hidden_states[:, -1, :]
        with torch.inference_mode():
            score = self._decision_head(todo_hidden)
        if not isinstance(score, Tensor) or score.numel() != 1:
            raise RuntimeError("decision head must return one score")
        value = float(score.detach().reshape(()).item())
        if not torch.isfinite(score.detach()).all().item():
            raise ValueError("decision head returned a non-finite score")
        return value

    @staticmethod
    def _resolve_decoder(model: nn.Module) -> nn.Module:
        decoder = getattr(model, "model", None)
        return decoder if isinstance(decoder, nn.Module) else model

    @staticmethod
    def _resolve_decision_head(model: nn.Module) -> nn.Module:
        head = getattr(model, "decision_head", None)
        if isinstance(head, nn.Module):
            return head
        raise TypeError("Perception/Decision model needs a decision head")

    @staticmethod
    def _output_hidden_states(outputs: Any) -> Tensor:
        hidden = getattr(outputs, "last_hidden_state", None)
        if hidden is None and isinstance(outputs, (tuple, list)) and outputs:
            hidden = outputs[0]
        if not isinstance(hidden, Tensor):
            raise RuntimeError("decoder did not return last_hidden_state")
        return hidden

    @staticmethod
    def _output_tuple_cache(outputs: Any) -> TupleCache:
        cache = getattr(outputs, "past_key_values", None)
        if cache is None and isinstance(outputs, (tuple, list)):
            if len(outputs) > 1:
                cache = outputs[1]
        to_tuple = getattr(cache, "to_legacy_cache", None)
        if callable(to_tuple):
            cache = to_tuple()
        if not isinstance(cache, tuple):
            raise RuntimeError("decoder did not return a tuple KV cache")
        _validate_tuple_cache(cache)
        return cache

    @classmethod
    def _flatten_decision(cls, blocks: Sequence[PerceptionDecisionBlock]) -> Tensor:
        cls._validate_compatible_blocks(blocks)
        return torch.cat([block.decision_embeddings() for block in blocks], dim=0)

    @staticmethod
    def _validate_compatible_blocks(
        blocks: Sequence[PerceptionDecisionBlock],
    ) -> None:
        if not blocks:
            raise ValueError("at least one Decision block is required")
        tensors = []
        for block in blocks:
            if not isinstance(block, PerceptionDecisionBlock):
                raise TypeError("all inputs must be PerceptionDecisionBlock")
            if block.role is DecisionBlockRole.PERCEPTION:
                assert block.clip_memory is not None
                assert block.time_state is not None
                tensors.extend((block.clip_memory, block.time_state))
            else:
                assert block.embeddings is not None
                tensors.append(block.embeddings)
        reference = tensors[0]
        for tensor in tensors[1:]:
            if tensor.shape[-1] != reference.shape[-1]:
                raise ValueError("Decision blocks have different hidden sizes")
            if tensor.device != reference.device:
                raise ValueError("Decision blocks are on different devices")
            if tensor.dtype != reference.dtype:
                raise ValueError("Decision blocks have different dtypes")

    @classmethod
    def _validate_stable_prefix(
        cls, prefix: Tuple[PerceptionDecisionBlock, ...]
    ) -> None:
        roles = tuple(block.role for block in prefix)
        if len(roles) < 2 or roles[:2] != (
            DecisionBlockRole.ANSWER,
            DecisionBlockRole.QUESTION,
        ):
            raise ValueError("stable prefix must start with ANS, Q")
        tail = roles[2:]
        expected = (
            DecisionBlockRole.SUMMARY,
            DecisionBlockRole.ANSWER,
        )
        if len(tail) % 2 or any(
            tail[index : index + 2] != expected for index in range(0, len(tail), 2)
        ):
            raise ValueError("stable prefix must contain (summary, ANS) pairs")
        cls._validate_compatible_blocks(prefix)

    @classmethod
    def _validate_pending_suffix(
        cls, suffix: Tuple[PerceptionDecisionBlock, ...]
    ) -> None:
        roles = tuple(block.role for block in suffix)
        if len(roles) < 3:
            raise ValueError("Decision suffix needs pending memory, Q, TODO")
        if roles[-2:] != (
            DecisionBlockRole.QUESTION,
            DecisionBlockRole.TODO,
        ) or any(role is not DecisionBlockRole.PERCEPTION for role in roles[:-2]):
            raise ValueError("Decision suffix must be perception+, Q, TODO")
        cls._validate_compatible_blocks(suffix)

    @classmethod
    def _validate_complete_sequence(
        cls, sequence: Tuple[PerceptionDecisionBlock, ...]
    ) -> None:
        roles = tuple(block.role for block in sequence)
        try:
            pending_start = roles.index(DecisionBlockRole.PERCEPTION)
        except ValueError as error:
            raise ValueError("Decision sequence has no pending memory") from error
        cls._validate_stable_prefix(sequence[:pending_start])
        cls._validate_pending_suffix(sequence[pending_start:])
        cls._validate_compatible_blocks(sequence)

    @staticmethod
    def _validate_mask(mask: Tensor, length: int, device: torch.device) -> None:
        if mask.ndim != 2 or tuple(mask.shape) != (1, length):
            raise ValueError("prefix attention mask has the wrong shape")
        if mask.device != device:
            raise ValueError("prefix attention mask is on the wrong device")
        if length and not torch.all(mask != 0).item():
            raise ValueError("stable prefix attention mask cannot contain padding")

    @staticmethod
    def _validate_prefix_cache(cache: DecisionPrefixCache) -> None:
        if not isinstance(cache, DecisionPrefixCache):
            raise TypeError("cache must be a DecisionPrefixCache")
        if cache.sequence_length < 1:
            raise ValueError("prefix cache must not be empty")
        _validate_tuple_cache(cache.past_key_values)
        if _tuple_cache_length(cache.past_key_values) != cache.sequence_length:
            raise ValueError("prefix cache length metadata is inconsistent")
        first_key = cache.past_key_values[0][0]
        DecisionKVBackend._validate_mask(
            cache.attention_mask, cache.sequence_length, first_key.device
        )

    @staticmethod
    def _validate_cache_length(cache: TupleCache, expected: int) -> None:
        actual = _tuple_cache_length(cache)
        if actual != expected:
            raise RuntimeError(
                f"decoder returned KV length {actual}; expected {expected}"
            )


def _validate_states(states: Optional[Tensor], name: str) -> None:
    if not isinstance(states, Tensor):
        raise TypeError(f"{name} must be a torch tensor")
    if states.ndim != 2 or states.shape[0] < 1 or states.shape[1] < 1:
        raise ValueError(f"{name} must have shape [tokens, hidden]")
    if not states.is_floating_point():
        raise TypeError(f"{name} must use a floating-point dtype")


def _validate_tuple_cache(cache: TupleCache) -> None:
    if not cache:
        raise ValueError("tuple KV cache must contain at least one layer")
    expected_length = None
    for layer in cache:
        if not isinstance(layer, tuple) or len(layer) < 2:
            raise TypeError("each cache layer must be a tensor tuple")
        for state in layer:
            if not isinstance(state, Tensor) or state.ndim < 3:
                raise TypeError("cache states must be tensors")
            length = state.shape[-2]
            if expected_length is None:
                expected_length = length
            elif length != expected_length:
                raise ValueError("cache layers have different lengths")


def _tuple_cache_length(cache: TupleCache) -> int:
    _validate_tuple_cache(cache)
    return int(cache[0][0].shape[-2])


def _clone_tuple_cache(cache: TupleCache) -> TupleCache:
    _validate_tuple_cache(cache)
    return tuple(tuple(state.clone() for state in layer) for layer in cache)
