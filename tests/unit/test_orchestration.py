import pytest

from mocks.mock_llms import ExtractiveMockLLM, FailingLLM
from mocks.mock_retrievers import FailingRetriever, StaticRetriever
from rag_guard.config import Settings
from rag_guard.orchestration import AdaptiveRecovery, RAGPipeline, StateManager
from rag_guard.orchestration.builder import build_pipeline
from rag_guard.retrieval import BM25Retriever, FallbackStage
from rag_guard.types import RecoveryStrategy, RetrievalResult
from tests.conftest import QUERY


def test_state_manager_carries_fields_forward(good_chunk: RetrievalResult) -> None:
    sm = StateManager(QUERY, "trace")
    sm.record("start")
    sm.record("retrieved", retrieved_chunks=[good_chunk], fallback_attempts=2)
    last = sm.record("generated", accumulated_cost_usd=0.5)
    assert last.fallback_attempts == 2
    assert last.retrieved_chunks[0].chunk_id == "doc-orion"
    assert [s.stage for s in sm.states] == ["start", "retrieved", "generated"]
    assert sm.audit_trail()[-1]["accumulated_cost_usd"] == 0.5
    assert len({s.state_id for s in sm.states}) == 3


def test_adaptive_recovery_orders_by_success(corpus: dict[str, str]) -> None:
    a = FallbackStage(RecoveryStrategy.RERANK, BM25Retriever(corpus))
    b = FallbackStage(RecoveryStrategy.BROADER_SEARCH, BM25Retriever(corpus))
    rec = AdaptiveRecovery()
    assert rec.order([a, b]) == [a, b]
    rec.record(RecoveryStrategy.RERANK, False)
    rec.record(RecoveryStrategy.BROADER_SEARCH, True)
    rec.record(RecoveryStrategy.NONE, True)
    assert rec.order([a, b]) == [b, a]
    assert rec.success_rate(RecoveryStrategy.BROADER_SEARCH) == 1.0
    assert rec.success_rate(RecoveryStrategy.NONE) == 0.0


async def test_pipeline_happy_path(
    corpus: dict[str, str], settings: Settings, llm: ExtractiveMockLLM
) -> None:
    pipeline = build_pipeline(corpus, llm, settings, primary=BM25Retriever(corpus))
    result = await pipeline.run(QUERY)
    assert result.success
    assert "90 days" in result.answer
    assert result.faithfulness == 1.0
    assert result.fallback_attempts == 0
    assert result.total_cost_usd > 0
    assert [s["stage"] for s in result.audit_trail] == ["start", "retrieved", "generated"]


async def test_pipeline_recovers_from_bad_primary(
    corpus: dict[str, str], settings: Settings, llm: ExtractiveMockLLM, noise_chunk: RetrievalResult
) -> None:
    pipeline = build_pipeline(corpus, llm, settings, primary=StaticRetriever([noise_chunk]))
    result = await pipeline.run(QUERY)
    assert result.success
    assert "90 days" in result.answer
    assert result.fallback_used is RecoveryStrategy.ALTERNATIVE_STRATEGY
    assert result.fallback_attempts == 1
    assert result.detection_result is not None and not result.detection_result.degraded


async def test_baseline_does_not_recover_and_confabulates(
    corpus: dict[str, str], settings: Settings, llm: ExtractiveMockLLM, noise_chunk: RetrievalResult
) -> None:
    pipeline = build_pipeline(
        corpus, llm, settings, primary=StaticRetriever([noise_chunk]), guarded=False
    )
    result = await pipeline.run(QUERY)
    assert "90 days" not in result.answer
    assert result.faithfulness == 0.0
    assert result.detection_result is not None and result.detection_result.degraded


async def test_pipeline_no_chunks_fails_gracefully(
    corpus: dict[str, str], settings: Settings
) -> None:
    llm = ExtractiveMockLLM()
    pipeline = build_pipeline(corpus, llm, settings, primary=FailingRetriever(), guarded=False)
    result = await pipeline.run(QUERY)
    assert not result.success
    assert result.error == "no chunks retrieved"
    assert llm.calls == 0


async def test_pipeline_llm_failure_is_captured(corpus: dict[str, str], settings: Settings) -> None:
    pipeline = build_pipeline(corpus, FailingLLM(), settings, primary=BM25Retriever(corpus))
    result = await pipeline.run(QUERY)
    assert not result.success
    assert "simulated provider outage" in (result.error or "")
    assert result.llm_response is None


async def test_pipeline_flags_budget_overrun(
    corpus: dict[str, str], settings: Settings, llm: ExtractiveMockLLM
) -> None:
    pipeline = build_pipeline(corpus, llm, settings, primary=BM25Retriever(corpus))
    pipeline.max_query_cost_usd = 1e-9
    result = await pipeline.run(QUERY)
    assert not result.success
    assert result.error == "query cost budget exceeded"


async def test_pipeline_adaptive_recovery_learns(
    corpus: dict[str, str], settings: Settings, llm: ExtractiveMockLLM, noise_chunk: RetrievalResult
) -> None:
    pipeline = build_pipeline(
        corpus, llm, settings, primary=StaticRetriever([noise_chunk]), adaptive=True
    )
    await pipeline.run(QUERY)
    assert pipeline.recovery is not None
    assert pipeline.recovery.success_rate(RecoveryStrategy.ALTERNATIVE_STRATEGY) == 1.0


def test_pipeline_is_constructible_directly(corpus: dict[str, str], llm: ExtractiveMockLLM) -> None:
    built = build_pipeline(corpus, llm)
    assert isinstance(built, RAGPipeline)
    with pytest.raises(ValueError):
        build_pipeline(
            corpus, llm, Settings(_env_file=None), primary=None
        ).chain.detector.__class__(threshold=0.1, critical_threshold=0.9)
