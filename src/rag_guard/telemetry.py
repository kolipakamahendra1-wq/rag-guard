"""Lightweight in-process telemetry: trace IDs, timed spans, counters."""

from __future__ import annotations

import time
import uuid
from collections import defaultdict
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field


def new_trace_id() -> str:
    """Return a new unique trace identifier."""
    return uuid.uuid4().hex


@dataclass
class Span:
    """A completed timed span."""

    name: str
    trace_id: str
    duration_ms: float


@dataclass
class Telemetry:
    """Collects spans and counters for a pipeline run."""

    trace_id: str = field(default_factory=new_trace_id)
    spans: list[Span] = field(default_factory=list)
    counters: dict[str, int] = field(default_factory=lambda: defaultdict(int))

    @contextmanager
    def span(self, name: str) -> Iterator[None]:
        """Time a block of code and record it as a span."""
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = (time.perf_counter() - start) * 1000.0
            self.spans.append(Span(name=name, trace_id=self.trace_id, duration_ms=elapsed))

    def incr(self, counter: str, amount: int = 1) -> None:
        """Increment a named counter."""
        self.counters[counter] += amount

    def total_ms(self) -> float:
        """Sum of all recorded span durations."""
        return sum(s.duration_ms for s in self.spans)
