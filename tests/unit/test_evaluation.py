import pytest

from mocks.mock_llms import ExtractiveMockLLM
from rag_guard.config import Settings
from rag_guard.evaluation import (
    BenchmarkRunner,
    QADataset,
    aggregate,
    answer_correct,
    comparison_table,
    recall_at_k,
)
from rag_guard.orchestration.builder import build_pipeline
from rag_guard.retrieval import BM25Retriever
from rag_guard.types import QAPair, RAGResult, RetrievalResult


def _result(
    answer: str, chunk_ids: list[str] | None = None, faith: float | None = 1.0
) -> RAGResult:
    chunk_ids = chunk_ids or []
    return RAGResult(
        query="q",
        answer=answer,
        retrieved_chunks=[RetrievalResult(chunk_id=c, content="x", score=0.5) for c in chunk_ids],
        faithfulness=faith,
        total_latency_ms=10.0,
        total_cost_usd=0.002,
        success=True,
    )


def _pair(expected: str = "90 days", ids: list[str] | None = None) -> QAPair:
    return QAPair(
        query_id="1",
        question="q",
        expected_answer=expected,
        expected_chunk_ids=["a"] if ids is None else ids,
        topic="t",
    )


def test_recall_and_accuracy() -> None:
    assert recall_at_k(_result("", ["a", "b"]), _pair(ids=["a", "c"])) == 0.5
    assert recall_at_k(_result("", []), _pair(ids=[])) == 1.0
    assert answer_correct(_result("It is 90 DAYS."), _pair())
    assert not answer_correct(_result("unknown"), _pair())


def test_aggregate_computes_means_and_percentiles() -> None:
    results = [_result("90 days", ["a"], 1.0), _result("nope", ["z"], 0.0)]
    metrics = aggregate(results, [_pair(), _pair()])
    assert metrics.recall == 0.5
    assert metrics.answer_accuracy == 0.5
    assert metrics.faithfulness == 0.5
    assert metrics.hallucination_rate == 0.5
    assert metrics.cost_per_query_usd == pytest.approx(0.002)
    assert metrics.latency_p50_ms == 10.0
    assert metrics.degraded_queries == 0
    assert metrics.recovery_success_rate == 1.0


def test_aggregate_validates_input() -> None:
    with pytest.raises(ValueError):
        aggregate([], [])
    with pytest.raises(ValueError):
        aggregate([_result("a", ["a"])], [])


def test_synthetic_dataset_is_deterministic_and_unique() -> None:
    a, b = QADataset.synthetic(30), QADataset.synthetic(30)
    assert a == b
    assert len(a.corpus) == 30 and len(a.pairs) == 30
    assert len({p.question for p in a.pairs}) == 30
    assert len(QADataset.synthetic(30, n_queries=5).pairs) == 5
    with pytest.raises(ValueError):
        QADataset.synthetic(0)
    with pytest.raises(ValueError):
        QADataset.synthetic(101)


async def test_runner_and_table() -> None:
    ds = QADataset.synthetic(20)
    settings = Settings(_env_file=None)
    runner = BenchmarkRunner()
    pipeline = build_pipeline(
        ds.corpus, ExtractiveMockLLM(), settings, primary=BM25Retriever(ds.corpus)
    )
    base = await runner.run("Baseline", pipeline, ds)
    cand = await runner.run("Candidate", pipeline, ds)
    assert base.num_queries == 20
    assert base.metrics.recall == 1.0
    table = comparison_table(base, cand)
    assert table.splitlines()[0] == "| Metric | Baseline | Candidate | Change |"
    assert "n/a (none degraded)" in table
    assert "| Recall | 100.0% | 100.0% | +0.0% |" in table
