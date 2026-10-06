"""Benchmark harness and comparison table rendering."""

from __future__ import annotations

from rag_guard.evaluation.dataset import QADataset
from rag_guard.evaluation.metrics import aggregate
from rag_guard.orchestration.pipeline import RAGPipeline
from rag_guard.types import BenchmarkResult, RAGResult


class BenchmarkRunner:
    """Runs a pipeline over every question in a dataset."""

    async def run(self, name: str, pipeline: RAGPipeline, dataset: QADataset) -> BenchmarkResult:
        """Execute sequentially (deterministic) and aggregate metrics."""
        results: list[RAGResult] = []
        for pair in dataset.pairs:
            results.append(await pipeline.run(pair.question))
        return BenchmarkResult(
            approach_name=name,
            metrics=aggregate(results, dataset.pairs),
            num_queries=len(results),
        )


def _pct_change(baseline: float, value: float) -> str:
    if baseline == 0.0:
        return "n/a"
    return f"{(value - baseline) / baseline * 100.0:+.1f}%"


def _recovery(rate: float, degraded: int) -> str:
    return f"{rate:.1%} ({degraded} degraded)" if degraded else "n/a (none degraded)"


def comparison_table(baseline: BenchmarkResult, candidate: BenchmarkResult) -> str:
    """Markdown table: baseline vs candidate vs relative change."""
    b, c = baseline.metrics, candidate.metrics
    rows = [
        ("Recall", b.recall, c.recall, "{:.1%}"),
        ("Answer accuracy", b.answer_accuracy, c.answer_accuracy, "{:.1%}"),
        ("Faithfulness", b.faithfulness, c.faithfulness, "{:.1%}"),
        ("Hallucination rate", b.hallucination_rate, c.hallucination_rate, "{:.1%}"),
        ("Latency p50 (ms)", b.latency_p50_ms, c.latency_p50_ms, "{:.2f}"),
        ("Latency p95 (ms)", b.latency_p95_ms, c.latency_p95_ms, "{:.2f}"),
        ("Cost / query (USD)", b.cost_per_query_usd, c.cost_per_query_usd, "{:.6f}"),
        ("Recovery success rate", b.recovery_success_rate, c.recovery_success_rate, "{:.1%}"),
    ]
    lines = [
        f"| Metric | {baseline.approach_name} | {candidate.approach_name} | Change |",
        "|---|---|---|---|",
    ]
    for label, bv, cv, fmt in rows:
        lines.append(f"| {label} | {fmt.format(bv)} | {fmt.format(cv)} | {_pct_change(bv, cv)} |")
    lines[-1] = (
        f"| Recovery success rate | {_recovery(b.recovery_success_rate, b.degraded_queries)} "
        f"| {_recovery(c.recovery_success_rate, c.degraded_queries)} | n/a |"
    )
    return "\n".join(lines)
