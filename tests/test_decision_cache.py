from __future__ import annotations

from copy import deepcopy
from typing import Dict, List, Tuple

import pytest

from dispider.streaming.decision_cache import (
    CacheDivergenceError,
    DecisionCache,
)


Token = str
Cache = Dict[str, List[Token]]


class MutableBackend:
    def __init__(self) -> None:
        self.built_caches: List[Cache] = []
        self.fork_count = 0
        self.uncached_count = 0

    def build_prefix_cache(self, prefix: Tuple[Token, ...]) -> Cache:
        cache = {"tokens": list(prefix)}
        self.built_caches.append(cache)
        return cache

    def fork_prefix_cache(self, cache: Cache) -> Cache:
        self.fork_count += 1
        return deepcopy(cache)

    def score_cached(self, cache: Cache, suffix: Tuple[Token, ...]) -> float:
        cache["tokens"].extend(suffix)
        return self._score(tuple(cache["tokens"]))

    def score_uncached(self, sequence: Tuple[Token, ...]) -> float:
        self.uncached_count += 1
        return self._score(sequence)

    @staticmethod
    def _score(sequence: Tuple[Token, ...]) -> float:
        return 1.0 if "fire" in sequence or "fire2" in sequence else -1.0


def summarize(pending: Tuple[Token, ...]) -> Token:
    return "S(" + ",".join(pending) + ")"


def new_cache(
    backend: MutableBackend,
    *,
    cache_enabled: bool = True,
    verify_with_oracle: bool = False,
    oracle_fallback_margin: float = 0.0,
) -> DecisionCache[Token, Cache]:
    return DecisionCache(
        backend,
        ans="ANS",
        question="Q",
        todo="TODO",
        cache_enabled=cache_enabled,
        verify_with_oracle=verify_with_oracle,
        oracle_fallback_margin=oracle_fallback_margin,
    )


def test_silent_branch_is_discarded_without_mutating_prefix_cache() -> None:
    backend = MutableBackend()
    state = new_cache(backend)

    result = state.observe("m1", summarize=summarize)

    assert not result.triggered
    assert result.evaluated_pending_count == 1
    assert state.stable_prefix == ("ANS", "Q")
    assert state.pending == ("m1",)
    assert state.cache_revision == 0
    assert backend.built_caches == [{"tokens": ["ANS", "Q"]}]


def test_first_trigger_rebuilds_from_new_stable_prefix() -> None:
    backend = MutableBackend()
    state = new_cache(backend)
    state.observe("m1", summarize=summarize)

    result = state.observe("fire", summarize=summarize)

    assert result.triggered
    assert result.evaluated_pending_count == 2
    assert result.committed_summary_count == 1
    assert state.pending == ()
    assert state.stable_prefix == ("ANS", "Q", "S(m1,fire)", "ANS")
    assert state.cache_revision == 1
    assert backend.built_caches == [
        {"tokens": ["ANS", "Q"]},
        {"tokens": ["ANS", "Q", "S(m1,fire)", "ANS"]},
    ]


def test_second_trigger_keeps_summary_delimiters_and_rebuilds_again() -> None:
    backend = MutableBackend()
    state = new_cache(backend)
    state.observe("fire", summarize=summarize)
    silent = state.observe("m2", summarize=summarize)
    triggered = state.observe("fire2", summarize=summarize)

    assert not silent.triggered
    assert triggered.triggered
    assert state.stable_prefix == (
        "ANS",
        "Q",
        "S(fire)",
        "ANS",
        "S(m2,fire2)",
        "ANS",
    )
    assert state.pending == ()
    assert state.committed_summary_count == 2
    assert state.cache_revision == 2
    assert len(backend.built_caches) == 3


def test_cached_trace_matches_no_cache_oracle() -> None:
    cached_backend = MutableBackend()
    oracle_backend = MutableBackend()
    cached = new_cache(cached_backend, verify_with_oracle=True)
    oracle = new_cache(oracle_backend, cache_enabled=False)

    for memory in ("m1", "fire", "m2", "fire2"):
        cached_result = cached.observe(memory, summarize=summarize)
        oracle_result = oracle.observe(memory, summarize=summarize)
        assert cached_result.score == oracle_result.score
        assert cached_result.triggered == oracle_result.triggered
        assert cached.stable_prefix == oracle.stable_prefix
        assert cached.pending == oracle.pending

    assert cached_backend.uncached_count == 4
    assert oracle_backend.built_caches == []
    assert oracle_backend.fork_count == 0


def test_oracle_mismatch_fails_without_changing_state() -> None:
    class DivergentBackend(MutableBackend):
        def score_uncached(self, sequence: Tuple[Token, ...]) -> float:
            return 0.5

    backend = DivergentBackend()
    state = new_cache(backend, verify_with_oracle=True)

    with pytest.raises(CacheDivergenceError):
        state.observe("m1", summarize=summarize)

    assert state.stable_prefix == ("ANS", "Q")
    assert state.pending == ()
    assert state.cache_revision == 0


def test_near_threshold_uses_oracle_as_a_guard_band() -> None:
    class NearThresholdBackend(MutableBackend):
        def score_cached(self, cache, suffix):
            cache["tokens"].extend(suffix)
            return 0.02

        def score_uncached(self, sequence):
            self.uncached_count += 1
            return -0.01

    backend = NearThresholdBackend()
    state = new_cache(backend, oracle_fallback_margin=0.125)

    result = state.observe("m1", summarize=summarize)

    assert not result.triggered
    assert result.score == -0.01
    assert result.cached_score == 0.02
    assert result.oracle_score == -0.01
    assert result.used_oracle_for_decision
    assert backend.uncached_count == 1
    assert state.pending == ("m1",)


def test_failed_trigger_rebuild_is_transactional() -> None:
    class FailingRebuildBackend(MutableBackend):
        def build_prefix_cache(self, prefix: Tuple[Token, ...]) -> Cache:
            if len(prefix) > 2:
                raise RuntimeError("rebuild failed")
            return super().build_prefix_cache(prefix)

    backend = FailingRebuildBackend()
    state = new_cache(backend)

    with pytest.raises(RuntimeError, match="rebuild failed"):
        state.observe("fire", summarize=summarize)

    assert state.stable_prefix == ("ANS", "Q")
    assert state.pending == ()
    assert state.cache_revision == 0
