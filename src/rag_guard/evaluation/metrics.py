"""Metric computation over pipeline results."""

from __future__ import annotations

from rag_guard.types import EvaluationMetrics, QAPair, RAGResult
from rag_guard.utils.text_utils import percentile


def recall_at_k(result: RAGResult, pair: QAPair) -> float:
    """Fraction of expected chunk ids present in the retrieved chunks."""
    if not pair.expected_chunk_ids:
        return 1.0
    got = {c.chunk_id for c in result.retrieved_chunks}
    return len(got & set(pair.expected_chunk_ids)) / len(pair.expected_chunk_ids)


def answer_correct(result: RAGResult, pair: QAPair) -> bool:
    """True when the expected answer string appears in the generated answer."""
    return pair.expected_answer.lower() in result.answer.lower()


def aggregate(results: list[RAGResult], pairs: list[QAPair]) -> EvaluationMetrics:
    """Aggregate per-query results into ``EvaluationMetrics``."""
    if not results or len(results) != len(pairs):
        raise ValueError("results and pairs must be non-empty and equal length")
    n = len(results)
    recall = sum(recall_at_k(r, p) for r, p in zip(results, pairs, strict=True)) / n
    accuracy = sum(answer_correct(r, p) for r, p in zip(results, pairs, strict=True)) / n
    faith = sum((r.faithfulness or 0.0) for r in results) / n
    latencies = [r.total_latency_ms for r in results]
    degraded = [
        r
        for r in results
        if r.fallback_attempts > 0 or (r.detection_result and r.detection_result.degraded)
    ]
    recovered = [r for r in degraded if r.detection_result and not r.detection_result.degraded]
    return EvaluationMetrics(
        recall=recall,
        answer_accuracy=accuracy,
        faithfulness=faith,
        hallucination_rate=1.0 - faith,
        latency_p50_ms=percentile(latencies, 50),
        latency_p95_ms=percentile(latencies, 95),
        latency_p99_ms=percentile(latencies, 99),
        cost_per_query_usd=sum(r.total_cost_usd for r in results) / n,
        recovery_success_rate=(len(recovered) / len(degraded)) if degraded else 1.0,
        degraded_queries=len(degraded),
    )
