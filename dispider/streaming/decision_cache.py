"""Transactional KV-cache state for Dispider's streaming decision model.

The decision sequence has two parts.  The stable prefix is ``ANS, Q`` followed
by zero or more ``summary, ANS`` pairs.  The speculative suffix is all memory
since the last trigger followed by ``Q, TODO``.  Only the stable prefix may be
cached persistently.

The released model's ``forward_inference`` method cannot implement this
protocol directly: it does not expose the cache returned by the inner Qwen
model, and its decision calls do not forward ``use_cache``.  A model-specific
adapter should call the inner model with ``use_cache=True``, implement an
isolated cache fork for every speculative suffix, and provide the last TODO
hidden state to ``silent_head``.  Attention masks and positions must cover the
stable prefix plus the suffix.  DynamicCache objects are mutable, so passing
the persistent object directly to a speculative call is invalid.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Generic, Optional, Protocol, Tuple, TypeVar


BlockT = TypeVar("BlockT")
CacheT = TypeVar("CacheT")


class DecisionBackend(Protocol[BlockT, CacheT]):
    """Model-specific operations required by :class:`DecisionCache`.

    ``fork_prefix_cache`` must return an isolated cache.  ``score_cached`` may
    mutate that fork, because the state machine always discards it afterward.
    It must never mutate the cache passed to ``fork_prefix_cache``.
    """

    def build_prefix_cache(self, prefix: Tuple[BlockT, ...]) -> CacheT:
        """Build a fresh cache containing exactly ``prefix``."""

    def fork_prefix_cache(self, cache: CacheT) -> CacheT:
        """Return branch-local cache state isolated from ``cache``."""

    def score_cached(self, cache: CacheT, suffix: Tuple[BlockT, ...]) -> float:
        """Return the silent-head logit after appending the suffix."""

    def score_uncached(self, sequence: Tuple[BlockT, ...]) -> float:
        """Return the no-cache oracle logit for the complete ``sequence``."""


class CacheDivergenceError(RuntimeError):
    """Raised when cached and no-cache decision logits disagree."""


@dataclass(frozen=True)
class DecisionResult:
    """One streaming decision and the committed state after it."""

    score: float
    triggered: bool
    evaluated_pending_count: int
    committed_summary_count: int
    cache_revision: int
    oracle_score: Optional[float] = None
    cached_score: Optional[float] = None
    used_oracle_for_decision: bool = False


class DecisionCache(Generic[BlockT, CacheT]):
    """Maintain a stable decision prefix and disposable speculative branches.

    State changes are transactional.  A silent decision commits only the new
    pending block.  A trigger summarizes all pending blocks, appends the
    summary and ANS delimiter to the stable prefix, and rebuilds the persistent
    cache from that complete prefix.  Failed scoring, summarization, or cache
    construction leaves the state unchanged.
    """

    def __init__(
        self,
        backend: DecisionBackend[BlockT, CacheT],
        *,
        ans: BlockT,
        question: BlockT,
        todo: BlockT,
        threshold: float = 0.0,
        cache_enabled: bool = True,
        verify_with_oracle: bool = False,
        oracle_atol: float = 1e-5,
        oracle_fallback_margin: float = 0.0,
    ) -> None:
        if not math.isfinite(threshold):
            raise ValueError("threshold must be finite")
        if not math.isfinite(oracle_atol) or oracle_atol < 0:
            raise ValueError("oracle_atol must be finite and non-negative")
        if not math.isfinite(oracle_fallback_margin) or oracle_fallback_margin < 0:
            raise ValueError("oracle_fallback_margin must be finite and non-negative")
        if verify_with_oracle and not cache_enabled:
            raise ValueError("oracle verification requires cache_enabled=True")

        self._backend = backend
        self._ans = ans
        self._question = question
        self._todo = todo
        self._threshold = threshold
        self._cache_enabled = cache_enabled
        self._verify_with_oracle = verify_with_oracle
        self._oracle_atol = oracle_atol
        self._oracle_fallback_margin = oracle_fallback_margin

        self._stable_prefix: Tuple[BlockT, ...] = (ans, question)
        self._pending: Tuple[BlockT, ...] = ()
        self._committed_summary_count = 0
        self._cache_revision = 0
        self._prefix_cache: Optional[CacheT]
        if cache_enabled:
            build_cache = backend.build_prefix_cache
            self._prefix_cache = build_cache(self._stable_prefix)
        else:
            self._prefix_cache = None

    @property
    def stable_prefix(self) -> Tuple[BlockT, ...]:
        return self._stable_prefix

    @property
    def pending(self) -> Tuple[BlockT, ...]:
        return self._pending

    @property
    def committed_summary_count(self) -> int:
        return self._committed_summary_count

    @property
    def cache_revision(self) -> int:
        return self._cache_revision

    @property
    def cache_enabled(self) -> bool:
        return self._cache_enabled

    def observe(
        self,
        memory: BlockT,
        *,
        summarize: Callable[[Tuple[BlockT, ...]], BlockT],
    ) -> DecisionResult:
        """Evaluate one block and atomically commit the resulting state."""

        candidate_pending = self._pending + (memory,)
        suffix = candidate_pending + (self._question, self._todo)
        full_sequence = self._stable_prefix + suffix

        oracle_score: Optional[float] = None
        cached_score: Optional[float] = None
        used_oracle_for_decision = False
        if self._cache_enabled:
            if self._prefix_cache is None:
                raise RuntimeError("cache-enabled state has no prefix cache")
            branch_cache = self._backend.fork_prefix_cache(self._prefix_cache)
            cached_score = self._finite_score(
                self._backend.score_cached(branch_cache, suffix), "cached"
            )
            score = cached_score
            near_threshold = (
                abs(cached_score - self._threshold) <= self._oracle_fallback_margin
            )
            if self._verify_with_oracle or near_threshold:
                oracle_score = self._finite_score(
                    self._backend.score_uncached(full_sequence), "oracle"
                )
                if self._verify_with_oracle and not math.isclose(
                    score,
                    oracle_score,
                    rel_tol=0.0,
                    abs_tol=self._oracle_atol,
                ):
                    raise CacheDivergenceError(
                        "cached decision score "
                        f"{score} differs from no-cache oracle {oracle_score}"
                    )
                if near_threshold:
                    score = oracle_score
                    used_oracle_for_decision = True
        else:
            score = self._finite_score(
                self._backend.score_uncached(full_sequence), "oracle"
            )
            oracle_score = score

        triggered = score > self._threshold
        if not triggered:
            self._pending = candidate_pending
            return DecisionResult(
                score=score,
                triggered=False,
                evaluated_pending_count=len(candidate_pending),
                committed_summary_count=self._committed_summary_count,
                cache_revision=self._cache_revision,
                oracle_score=oracle_score,
                cached_score=cached_score,
                used_oracle_for_decision=used_oracle_for_decision,
            )

        summary = summarize(candidate_pending)
        next_prefix = self._stable_prefix + (summary, self._ans)
        next_cache: Optional[CacheT] = None
        if self._cache_enabled:
            next_cache = self._backend.build_prefix_cache(next_prefix)

        self._stable_prefix = next_prefix
        self._pending = ()
        self._committed_summary_count += 1
        self._cache_revision += 1
        self._prefix_cache = next_cache
        return DecisionResult(
            score=score,
            triggered=True,
            evaluated_pending_count=len(candidate_pending),
            committed_summary_count=self._committed_summary_count,
            cache_revision=self._cache_revision,
            oracle_score=oracle_score,
            cached_score=cached_score,
            used_oracle_for_decision=used_oracle_for_decision,
        )

    @staticmethod
    def _finite_score(value: float, source: str) -> float:
        score = float(value)
        if not math.isfinite(score):
            raise ValueError(f"{source} decision score must be finite")
        return score
