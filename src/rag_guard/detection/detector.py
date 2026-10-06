"""Threshold and drift based retrieval failure detection."""

from __future__ import annotations

from collections import deque

from rag_guard.detection.scorer import RetrievalScorer
from rag_guard.types import DetectionResult, RecoveryStrategy, RetrievalResult, SeverityLevel


class FailureDetector:
    """Flags degraded retrieval via an absolute threshold and a rolling-baseline drift check."""

    def __init__(
        self,
        threshold: float = 0.6,
        critical_threshold: float = 0.3,
        drift_window: int = 10,
        drift_threshold: float = 0.15,
        scorer: RetrievalScorer | None = None,
    ) -> None:
        if not 0.0 <= critical_threshold <= threshold <= 1.0:
            raise ValueError("require 0 <= critical_threshold <= threshold <= 1")
        self.threshold = threshold
        self.critical_threshold = critical_threshold
        self.drift_threshold = drift_threshold
        self.scorer = scorer or RetrievalScorer()
        self._history: deque[float] = deque(maxlen=max(1, drift_window))

    def baseline(self) -> float | None:
        """Mean of recent healthy scores, or None before any history exists."""
        if not self._history:
            return None
        return sum(self._history) / len(self._history)

    def detect(self, query: str, chunks: list[RetrievalResult]) -> DetectionResult:
        """Assess retrieval quality and recommend a recovery strategy."""
        scored = self.scorer.score(query, chunks)
        score = scored.relevance
        baseline = self.baseline()
        drifted = baseline is not None and (baseline - score) > self.drift_threshold

        if not chunks or score < self.critical_threshold:
            severity = SeverityLevel.CRITICAL
        elif score < self.threshold or drifted:
            severity = SeverityLevel.DEGRADED
        else:
            severity = SeverityLevel.HEALTHY

        if severity is SeverityLevel.HEALTHY:
            strategy = RecoveryStrategy.NONE
            self._history.append(score)
        elif severity is SeverityLevel.CRITICAL:
            strategy = RecoveryStrategy.ALTERNATIVE_STRATEGY
        elif scored.signals.get("coverage", 0.0) < 0.5:
            strategy = RecoveryStrategy.BROADER_SEARCH
        else:
            strategy = RecoveryStrategy.RERANK

        return DetectionResult(
            degraded=severity is not SeverityLevel.HEALTHY,
            severity=severity,
            score=score,
            recommended_strategy=strategy,
            confidence=scored.confidence,
            details={"signals": scored.signals, "baseline": baseline, "drifted": drifted},
        )
