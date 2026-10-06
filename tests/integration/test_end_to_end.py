import httpx
import pytest

from mocks.mock_llms import ExtractiveMockLLM
from mocks.mock_retrievers import DriftedDenseRetriever, FailingRetriever
from rag_guard.config import Settings
from rag_guard.evaluation import BenchmarkRunner, QADataset
from rag_guard.llm import AnthropicProvider, OpenAIProvider
from rag_guard.orchestration.builder import build_pipeline
from rag_guard.retrieval import BM25Retriever
from tests.conftest import QUERY, mock_client

SETTINGS = Settings(_env_file=None)


async def _bench(noise: float, guarded: bool):  # type: ignore[no-untyped-def]
    ds = QADataset.synthetic(40)
    pipeline = build_pipeline(
        ds.corpus,
        ExtractiveMockLLM(),
        SETTINGS,
        primary=DriftedDenseRetriever(ds.corpus, noise=noise),
        guarded=guarded,
    )
    return await BenchmarkRunner().run("x", pipeline, ds)


async def test_guarded_beats_baseline_under_heavy_drift() -> None:
    baseline = await _bench(8.0, guarded=False)
    guarded = await _bench(8.0, guarded=True)
    assert baseline.metrics.recall < 0.7
    assert guarded.metrics.recall == 1.0
    assert guarded.metrics.hallucination_rate < baseline.metrics.hallucination_rate
    assert guarded.metrics.recovery_success_rate == 1.0
    assert baseline.metrics.recovery_success_rate == 0.0


async def test_guarded_matches_baseline_without_drift() -> None:
    baseline = await _bench(0.0, guarded=False)
    guarded = await _bench(0.0, guarded=True)
    assert baseline.metrics.recall == guarded.metrics.recall == 1.0
    assert guarded.metrics.degraded_queries == 0
    assert guarded.metrics.cost_per_query_usd == pytest.approx(baseline.metrics.cost_per_query_usd)


async def test_benchmark_is_reproducible() -> None:
    a = await _bench(5.0, guarded=True)
    b = await _bench(5.0, guarded=True)
    assert a.metrics.recall == b.metrics.recall
    assert a.metrics.faithfulness == b.metrics.faithfulness
    assert a.metrics.cost_per_query_usd == b.metrics.cost_per_query_usd


async def test_primary_outage_is_survived(corpus: dict[str, str]) -> None:
    pipeline = build_pipeline(corpus, ExtractiveMockLLM(), SETTINGS, primary=FailingRetriever())
    result = await pipeline.run(QUERY)
    assert result.success and "90 days" in result.answer
    assert result.fallback_attempts >= 1


@pytest.mark.parametrize("provider_name", ["anthropic", "openai"])
async def test_same_pipeline_across_providers(corpus: dict[str, str], provider_name: str) -> None:
    anthropic_body = {
        "content": [{"type": "text", "text": "Orion retention period is 90 days."}],
        "usage": {"input_tokens": 50, "output_tokens": 8},
        "stop_reason": "end_turn",
    }
    openai_body = {
        "choices": [
            {"message": {"content": "Orion retention period is 90 days."}, "finish_reason": "stop"}
        ],
        "usage": {"prompt_tokens": 50, "completion_tokens": 8},
    }

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, json=anthropic_body if provider_name == "anthropic" else openai_body
        )

    client = mock_client(handler)
    llm = (
        AnthropicProvider("m", "k", client=client)
        if provider_name == "anthropic"
        else OpenAIProvider("m", "k", client=client)
    )
    pipeline = build_pipeline(corpus, llm, SETTINGS, primary=BM25Retriever(corpus))
    result = await pipeline.run(QUERY)
    assert result.success
    assert result.faithfulness == 1.0
    assert (
        result.llm_response is not None
        and result.llm_response.tokens_used.provider == provider_name
    )
