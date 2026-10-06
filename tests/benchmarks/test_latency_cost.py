import pytest

from mocks.mock_llms import ExtractiveMockLLM
from mocks.mock_retrievers import DriftedDenseRetriever
from rag_guard.config import Settings
from rag_guard.evaluation import BenchmarkRunner, QADataset
from rag_guard.orchestration.builder import build_pipeline

pytestmark = pytest.mark.benchmark


async def _run(guarded: bool, noise: float = 5.0):  # type: ignore[no-untyped-def]
    ds = QADataset.synthetic(40)
    pipeline = build_pipeline(
        ds.corpus,
        ExtractiveMockLLM(),
        Settings(_env_file=None),
        primary=DriftedDenseRetriever(ds.corpus, noise=noise),
        guarded=guarded,
    )
    return await BenchmarkRunner().run("x", pipeline, ds)


async def test_pipeline_overhead_stays_small() -> None:
    result = await _run(guarded=True)
    assert result.metrics.latency_p95_ms < 250.0


async def test_fallbacks_do_not_inflate_token_cost() -> None:
    baseline = await _run(guarded=False)
    guarded = await _run(guarded=True)
    assert guarded.metrics.cost_per_query_usd <= baseline.metrics.cost_per_query_usd * 1.10
