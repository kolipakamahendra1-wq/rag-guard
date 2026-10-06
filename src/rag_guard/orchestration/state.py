"""Append-only pipeline state log (deterministic, replayable audit trail)."""

from __future__ import annotations

from typing import Any

from rag_guard.types import PipelineState


class StateManager:
    """Records immutable ``PipelineState`` snapshots for one pipeline run."""

    def __init__(self, query: str, trace_id: str) -> None:
        self.query = query
        self.trace_id = trace_id
        self._states: list[PipelineState] = []

    def record(self, stage: str, **updates: Any) -> PipelineState:
        """Append a snapshot for ``stage``, carrying forward the previous state's fields."""
        base = self._states[-1].model_dump() if self._states else {"query": self.query}
        base.update(updates)
        base["state_id"] = f"{self.trace_id}-{len(self._states)}"
        base["stage"] = stage
        state = PipelineState.model_validate(base)
        self._states.append(state)
        return state

    @property
    def states(self) -> list[PipelineState]:
        """Copy of the recorded snapshots, oldest first."""
        return list(self._states)

    def audit_trail(self) -> list[dict[str, Any]]:
        """Compact, JSON-friendly trail of stage transitions."""
        return [
            {
                "state_id": s.state_id,
                "stage": s.stage,
                "fallback_attempts": s.fallback_attempts,
                "elapsed_ms": s.elapsed_ms,
                "accumulated_cost_usd": s.accumulated_cost_usd,
            }
            for s in self._states
        ]
