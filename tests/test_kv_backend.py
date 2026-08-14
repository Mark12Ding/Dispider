from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import pytest
import torch
from torch import Tensor, nn

from dispider.streaming.decision_cache import (
    CacheDivergenceError,
    DecisionCache,
)
from dispider.streaming.kv_backend import (
    DecisionKVBackend,
    DecisionPrefixCache,
    PerceptionDecisionBlock,
)


TupleCache = Tuple[Tuple[Tensor, Tensor], ...]


@dataclass
class DecoderCall:
    inputs: Tensor
    attention_mask: Tensor
    position_ids: Tensor
    cache_position: Tensor
    indicators: Tensor
    use_cache: bool
    past_length: int


class TinyDecoder(nn.Module):
    """A deterministic causal model with a Qwen-compatible cache surface."""

    def __init__(self, *, cached_bias: float = 0.0) -> None:
        super().__init__()
        self.cached_bias = cached_bias
        self.calls = []

    def forward(
        self,
        input_ids=None,
        *,
        inputs_embeds: Tensor,
        attention_mask: Tensor,
        position_ids: Tensor,
        cache_position: Tensor,
        past_key_values: Optional[TupleCache],
        indicators: Tensor,
        use_cache: bool,
        output_attentions: bool,
        output_hidden_states: bool,
        return_dict: bool,
        select_layer: int,
    ):
        del input_ids, output_attentions, output_hidden_states, select_layer
        assert return_dict
        if past_key_values is None:
            prefix = inputs_embeds[:, :0]
        else:
            prefix = past_key_values[0][0][:, 0]
        past_length = prefix.shape[1]
        expected_positions = torch.arange(
            past_length,
            past_length + inputs_embeds.shape[1],
            device=inputs_embeds.device,
        )
        assert torch.equal(position_ids[0], expected_positions)
        assert torch.equal(cache_position, expected_positions)
        assert attention_mask.shape == (
            inputs_embeds.shape[0],
            past_length + inputs_embeds.shape[1],
        )
        assert torch.all(attention_mask)
        assert indicators.shape == inputs_embeds.shape[:2]

        self.calls.append(
            DecoderCall(
                inputs_embeds.detach().clone(),
                attention_mask.detach().clone(),
                position_ids.detach().clone(),
                cache_position.detach().clone(),
                indicators.detach().clone(),
                use_cache,
                past_length,
            )
        )
        full_inputs = torch.cat((prefix, inputs_embeds), dim=1)
        full_hidden = torch.cumsum(full_inputs, dim=1)
        hidden = full_hidden[:, past_length:]
        if past_key_values is not None:
            hidden = hidden + self.cached_bias

        cache = None
        if use_cache:
            key = full_inputs.unsqueeze(1).clone()
            value = (full_inputs * 2).unsqueeze(1).clone()
            cache = ((key, value),)
        return TinyOutput(hidden, cache)


class TinyOutput:
    def __init__(self, hidden: Tensor, cache: Optional[TupleCache]) -> None:
        self.last_hidden_state = hidden
        self.past_key_values = cache


class TinyPerceptionDecisionModel(nn.Module):
    def __init__(self, *, cached_bias: float = 0.0) -> None:
        super().__init__()
        self.model = TinyDecoder(cached_bias=cached_bias)
        self.decision_head = nn.Linear(2, 1, bias=False)
        with torch.no_grad():
            self.decision_head.weight.copy_(torch.tensor([[1.0, -0.5]]))


def states(*rows) -> Tensor:
    return torch.tensor(rows, dtype=torch.float32)


def blocks():
    answer = PerceptionDecisionBlock.answer(states((0.5, 0.0)))
    question = PerceptionDecisionBlock.question(states((0.0, 0.5), (0.25, 0.0)))
    todo = PerceptionDecisionBlock.todo(states((0.0, 0.0)))
    first = PerceptionDecisionBlock.perception(
        states((0.1, 0.2), (0.3, 0.1)), states((0.2, 0.2))
    )
    second = PerceptionDecisionBlock.perception(states((1.5, 0.0)), states((0.4, 0.1)))
    return answer, question, todo, first, second


def test_cached_score_matches_oracle_and_preserves_prefix_cache() -> None:
    model = TinyPerceptionDecisionModel()
    backend = DecisionKVBackend(model)
    answer, question, todo, first, _ = blocks()
    prefix = (answer, question)
    suffix = (first, question, todo)
    cache = backend.build_prefix_cache(prefix)
    original_key = cache.past_key_values[0][0].clone()
    original_value = cache.past_key_values[0][1].clone()

    branch = backend.fork_prefix_cache(cache)
    cached_score = backend.score_cached(branch, suffix)
    oracle_score = backend.score_uncached(prefix + suffix)

    assert cached_score == pytest.approx(oracle_score, abs=1e-7)
    assert torch.equal(cache.past_key_values[0][0], original_key)
    assert torch.equal(cache.past_key_values[0][1], original_value)
    assert cache.sequence_length == 3
    assert model.model.calls[0].use_cache
    assert model.model.calls[1].use_cache
    assert model.model.calls[1].past_length == 3
    assert model.model.calls[1].attention_mask.shape == (1, 8)
    assert model.model.calls[1].position_ids.tolist() == [list(range(3, 8))]
    assert not model.model.calls[2].use_cache
    assert model.model.calls[2].past_length == 0


def test_fork_is_deep_and_rejects_inconsistent_cache_metadata() -> None:
    model = TinyPerceptionDecisionModel()
    backend = DecisionKVBackend(model)
    answer, question, _, _, _ = blocks()
    cache = backend.build_prefix_cache((answer, question))

    fork = backend.fork_prefix_cache(cache)
    fork.past_key_values[0][0].add_(10)
    fork.attention_mask.zero_()

    assert not torch.equal(fork.past_key_values[0][0], cache.past_key_values[0][0])
    assert torch.all(cache.attention_mask)
    invalid = DecisionPrefixCache(
        cache.past_key_values, cache.attention_mask, cache.sequence_length + 1
    )
    with pytest.raises(ValueError, match="metadata"):
        backend.fork_prefix_cache(invalid)


def test_trigger_summarizes_all_memory_then_all_time_and_rebuilds() -> None:
    model = TinyPerceptionDecisionModel()
    backend = DecisionKVBackend(model)
    answer, question, todo, first, second = blocks()
    state = DecisionCache(
        backend,
        ans=answer,
        question=question,
        todo=todo,
        threshold=1.5,
        verify_with_oracle=True,
        oracle_atol=1e-7,
    )

    deferred = state.observe(first, summarize=backend.summarize)
    triggered = state.observe(second, summarize=backend.summarize)

    assert not deferred.triggered
    assert triggered.triggered
    assert state.pending == ()
    assert state.committed_summary_count == 1
    assert state.cache_revision == 1
    assert len(state.stable_prefix) == 4
    summary = state.stable_prefix[2]
    assert summary.embeddings is not None
    assert summary.embeddings.shape == (2, 2)

    expected_aggregation = torch.cat(
        (
            first.clip_memory,
            second.clip_memory,
            first.time_state,
            second.time_state,
        ),
        dim=0,
    )
    aggregation_calls = [
        call
        for call in model.model.calls
        if torch.equal(call.inputs[0], expected_aggregation)
    ]
    assert len(aggregation_calls) == 1
    assert not aggregation_calls[0].use_cache
    assert aggregation_calls[0].past_length == 0

    prefix_builds = [
        call for call in model.model.calls if call.use_cache and call.past_length == 0
    ]
    assert [call.inputs.shape[1] for call in prefix_builds] == [3, 6]


def test_reconstructing_state_resets_and_rebuilds_initial_prefix() -> None:
    model = TinyPerceptionDecisionModel()
    backend = DecisionKVBackend(model)
    answer, question, todo, _, second = blocks()
    state = DecisionCache(
        backend,
        ans=answer,
        question=question,
        todo=todo,
        threshold=0.0,
        verify_with_oracle=True,
    )
    assert state.observe(second, summarize=backend.summarize).triggered

    state = DecisionCache(
        backend,
        ans=answer,
        question=question,
        todo=todo,
        threshold=0.0,
        verify_with_oracle=True,
    )

    assert state.stable_prefix == (answer, question)
    assert state.pending == ()
    assert state.committed_summary_count == 0
    assert state.cache_revision == 0
    prefix_builds = [
        call for call in model.model.calls if call.use_cache and call.past_length == 0
    ]
    assert [call.inputs.shape[1] for call in prefix_builds] == [3, 5, 3]


@pytest.mark.parametrize(
    ("cached_bias", "oracle_atol", "raises"),
    [(1e-6, 1e-5, False), (1e-2, 1e-5, True)],
)
def test_oracle_tolerance_is_fail_closed(
    cached_bias: float, oracle_atol: float, raises: bool
) -> None:
    model = TinyPerceptionDecisionModel(cached_bias=cached_bias)
    backend = DecisionKVBackend(model)
    answer, question, todo, first, _ = blocks()
    state = DecisionCache(
        backend,
        ans=answer,
        question=question,
        todo=todo,
        threshold=100.0,
        verify_with_oracle=True,
        oracle_atol=oracle_atol,
    )

    if raises:
        with pytest.raises(CacheDivergenceError):
            state.observe(first, summarize=backend.summarize)
        assert state.pending == ()
        assert state.cache_revision == 0
    else:
        result = state.observe(first, summarize=backend.summarize)
        assert not result.triggered
        assert state.pending == (first,)


def test_invalid_cross_clip_sequences_fail_before_model_execution() -> None:
    model = TinyPerceptionDecisionModel()
    backend = DecisionKVBackend(model)
    answer, question, todo, first, _ = blocks()

    with pytest.raises(ValueError, match="stable prefix"):
        backend.build_prefix_cache((answer, question, first))
    with pytest.raises(ValueError, match="suffix"):
        backend.score_uncached((answer, question, first, todo))
    with pytest.raises(ValueError, match="Perception blocks"):
        backend.summarize((first, question))
    assert model.model.calls == []
