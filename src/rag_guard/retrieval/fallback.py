"""Fallback chain: escalate through retrieval strategies until quality recovers."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from rag_guard.detection.detector import FailureDetector
from rag_guard.logging import get_logger
from rag_guard.retrieval.base import Retriever
from rag_guard.retrieval.reranking import Reranker
from rag_guard.types import DetectionResult, RecoveryStrategy, RetrievalResult
from rag_guard.utils.async_utils import retry_async, with_timeout

_log = get_logger(__name__)


@dataclass(frozen=True)
class FallbackStage:
    """One fallback step: a strategy label, a retriever, and an optional reranker."""

    strategy: RecoveryStrategy
    retriever: Retriever
    reranker: Reranker | None = None


@dataclass
class FallbackResult:
    """Outcome of running a retrieval with fallbacks, including a full audit trail."""

    chunks: list[RetrievalResult]
    detection: DetectionResult
    strategy_used: RecoveryStrategy
    recovered: bool
    attempts: list[dict[str, Any]] = field(default_factory=list)


class FallbackChain:
    """Runs a primary retriever, then fallback stages in order while quality stays degraded."""

    def __init__(
        self,
        primary: Retriever,
        detector: FailureDetector,
        stages: list[FallbackStage] | None = None,
        *,
        top_k: int = 5,
        max_attempts: int = 3,
        timeout_ms: float = 5000.0,
        retries: int = 2,
        backoff_base_ms: float = 50.0,
        backoff_max_ms: float = 1000.0,
    ) -> None:
        self.primary = primary
        self.detector = detector
        self.stages = stages or []
        self.top_k = top_k
        self.max_attempts = max_attempts
        self.timeout_ms = timeout_ms
        self.retries = retries
        self.backoff_base_ms = backoff_base_ms
        self.backoff_max_ms = backoff_max_ms

    async def _fetch(self, retriever: Retriever, query: str) -> list[RetrievalResult]:
        async def call() -> list[RetrievalResult]:
            return await with_timeout(retriever.retrieve(query, self.top_k), self.timeout_ms)

        return await retry_async(
            call,
            attempts=self.retries,
            base_ms=self.backoff_base_ms,
            max_ms=self.backoff_max_ms,
        )

    async def run(self, query: str) -> FallbackResult:
        """Retrieve for ``query``, escalating through fallback stages while degraded."""
        attempts: list[dict[str, Any]] = []
        best_chunks: list[RetrievalResult] = []
        best_detection: DetectionResult | None = None
        best_strategy = RecoveryStrategy.NONE

        plan: list[tuple[RecoveryStrategy, Retriever, Reranker | None]] = [
            (RecoveryStrategy.NONE, self.primary, None)
        ]
        plan.extend((s.strategy, s.retriever, s.reranker) for s in self.stages)

        for index, (strategy, retriever, reranker) in enumerate(plan[: self.max_attempts + 1]):
            started = time.perf_counter()
            try:
                chunks = await self._fetch(retriever, query)
            except Exception as exc:  # noqa: BLE001 - a failing backend must not abort the chain
                attempts.append(
                    {
                        "step": index,
                        "strategy": strategy.value,
                        "retriever": retriever.name,
                        "error": str(exc),
                    }
                )
                _log.warning("fallback_stage_failed", strategy=strategy.value, error=str(exc))
                continue
            if reranker is not None:
                chunks = reranker.rerank(query, chunks, self.top_k)
            detection = self.detector.detect(query, chunks)
            attempts.append(
                {
                    "step": index,
                    "strategy": strategy.value,
                    "retriever": retriever.name,
                    "score": detection.score,
                    "severity": detection.severity.value,
                    "n_chunks": len(chunks),
                    "latency_ms": (time.perf_counter() - started) * 1000.0,
                }
            )
            if best_detection is None or detection.score > best_detection.score:
                best_chunks, best_detection, best_strategy = chunks, detection, strategy
            if not detection.degraded:
                break

        if best_detection is None:
            best_detection = self.detector.detect(query, [])
        return FallbackResult(
            chunks=best_chunks,
            detection=best_detection,
            strategy_used=best_strategy,
            recovered=not best_detection.degraded and best_strategy is not RecoveryStrategy.NONE,
            attempts=attempts,
        )
