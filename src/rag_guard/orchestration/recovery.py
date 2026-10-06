"""Adaptive recovery: order fallback stages by observed success rate."""

from __future__ import annotations

from collections import defaultdict

from rag_guard.retrieval.fallback import FallbackStage
from rag_guard.types import RecoveryStrategy


class AdaptiveRecovery:
    """Tracks per-strategy recovery outcomes and reorders stages, best first.

    Ordering is stable: strategies with equal observed rates (including unseen ones)
    keep their configured order.
    """

    def __init__(self) -> None:
        self._wins: dict[RecoveryStrategy, int] = defaultdict(int)
        self._tries: dict[RecoveryStrategy, int] = defaultdict(int)

    def record(self, strategy: RecoveryStrategy, recovered: bool) -> None:
        """Record whether ``strategy`` recovered a degraded retrieval."""
        if strategy is RecoveryStrategy.NONE:
            return
        self._tries[strategy] += 1
        if recovered:
            self._wins[strategy] += 1

    def success_rate(self, strategy: RecoveryStrategy) -> float:
        """Observed recovery rate, 0.0 when never tried."""
        tries = self._tries[strategy]
        return self._wins[strategy] / tries if tries else 0.0

    def order(self, stages: list[FallbackStage]) -> list[FallbackStage]:
        """Return ``stages`` sorted by descending observed success rate (stable)."""
        return sorted(stages, key=lambda s: -self.success_rate(s.strategy))
