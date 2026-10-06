import pytest

from mocks.mock_retrievers import (
    DriftedDenseRetriever,
    FailingRetriever,
    FlakyRetriever,
    StaticRetriever,
)
from rag_guard.detection import FailureDetector
from rag_guard.retrieval import (
    BM25Retriever,
    DenseRetriever,
    FallbackChain,
    FallbackStage,
    HybridRetriever,
    KeywordRetriever,
    OverlapReranker,
)
from rag_guard.types import RecoveryStrategy, RetrievalResult, SeverityLevel
from tests.conftest import QUERY


async def test_bm25_ranks_relevant_first(corpus: dict[str, str]) -> None:
    results = await BM25Retriever(corpus).retrieve(QUERY, k=3)
    assert results[0].chunk_id == "doc-orion"
    assert all(0.0 <= r.score <= 1.0 for r in results)
    assert results[0].retrieval_method == "bm25"


async def test_bm25_no_match_returns_empty(corpus: dict[str, str]) -> None:
    assert await BM25Retriever(corpus).retrieve("zzz qqq", k=3) == []


async def test_dense_ranks_relevant_first(corpus: dict[str, str]) -> None:
    results = await DenseRetriever(corpus).retrieve(QUERY, k=2)
    assert results[0].chunk_id == "doc-orion"
    assert len(results) <= 2


async def test_keyword_retriever_scores_by_term_fraction(corpus: dict[str, str]) -> None:
    results = await KeywordRetriever(corpus).retrieve(QUERY, k=2)
    assert results[0].chunk_id == "doc-orion"
    assert results[0].score == 1.0
    assert await KeywordRetriever(corpus).retrieve("the of", k=2) == []


async def test_hybrid_fuses_and_normalises(corpus: dict[str, str]) -> None:
    hybrid = HybridRetriever([BM25Retriever(corpus), DenseRetriever(corpus)])
    results = await hybrid.retrieve(QUERY, k=3)
    assert results[0].chunk_id == "doc-orion"
    assert results[0].score == pytest.approx(1.0)
    assert all(0.0 <= r.score <= 1.0 for r in results)


def test_hybrid_requires_retrievers() -> None:
    with pytest.raises(ValueError):
        HybridRetriever([])


async def test_batch_retrieve(corpus: dict[str, str]) -> None:
    out = await BM25Retriever(corpus).batch_retrieve([QUERY, "vega rate limit"], k=1)
    assert [r[0].chunk_id for r in out] == ["doc-orion", "doc-vega"]


def test_reranker_promotes_term_overlap() -> None:
    off = RetrievalResult(chunk_id="off", content="unrelated text", score=0.9)
    on = RetrievalResult(chunk_id="on", content="retention period of orion", score=0.5)
    ranked = OverlapReranker().rerank(QUERY, [off, on], top_k=1)
    assert [c.chunk_id for c in ranked] == ["on"]


def test_reranker_validates_weight() -> None:
    with pytest.raises(ValueError):
        OverlapReranker(1.5)


async def test_drifted_retriever_zero_noise_matches_dense(corpus: dict[str, str]) -> None:
    clean = await DenseRetriever(corpus).retrieve(QUERY, k=3)
    drifted = await DriftedDenseRetriever(corpus, noise=0.0).retrieve(QUERY, k=3)
    assert [r.chunk_id for r in clean] == [r.chunk_id for r in drifted]
    with pytest.raises(ValueError):
        DriftedDenseRetriever(corpus, noise=-1.0)


async def test_drifted_retriever_is_deterministic(corpus: dict[str, str]) -> None:
    r = DriftedDenseRetriever(corpus, noise=5.0)
    assert await r.retrieve(QUERY, k=3) == await r.retrieve(QUERY, k=3)


def _chain(primary, stages, **kw):  # type: ignore[no-untyped-def]
    return FallbackChain(primary, FailureDetector(), stages, retries=1, **kw)


async def test_chain_healthy_primary_skips_fallbacks(
    corpus: dict[str, str], good_chunk: RetrievalResult
) -> None:
    fallback = StaticRetriever([good_chunk])
    chain = _chain(
        StaticRetriever([good_chunk]),
        [FallbackStage(RecoveryStrategy.ALTERNATIVE_STRATEGY, fallback)],
    )
    result = await chain.run(QUERY)
    assert not result.detection.degraded
    assert result.strategy_used is RecoveryStrategy.NONE
    assert not result.recovered
    assert fallback.calls == 0
    assert len(result.attempts) == 1


async def test_chain_recovers_with_fallback(
    corpus: dict[str, str], noise_chunk: RetrievalResult
) -> None:
    chain = _chain(
        StaticRetriever([noise_chunk]),
        [FallbackStage(RecoveryStrategy.ALTERNATIVE_STRATEGY, BM25Retriever(corpus))],
    )
    result = await chain.run(QUERY)
    assert result.recovered
    assert result.strategy_used is RecoveryStrategy.ALTERNATIVE_STRATEGY
    assert result.chunks[0].chunk_id == "doc-orion"
    assert [a["step"] for a in result.attempts] == [0, 1]


async def test_chain_survives_failing_primary(corpus: dict[str, str]) -> None:
    chain = _chain(
        FailingRetriever(),
        [FallbackStage(RecoveryStrategy.ALTERNATIVE_STRATEGY, BM25Retriever(corpus))],
    )
    result = await chain.run(QUERY)
    assert result.recovered
    assert "error" in result.attempts[0]


async def test_chain_all_backends_down_returns_critical_empty() -> None:
    chain = _chain(
        FailingRetriever(),
        [FallbackStage(RecoveryStrategy.ALTERNATIVE_STRATEGY, FailingRetriever())],
    )
    result = await chain.run(QUERY)
    assert result.chunks == []
    assert result.detection.severity is SeverityLevel.CRITICAL
    assert not result.recovered


async def test_chain_keeps_best_when_nothing_recovers(noise_chunk: RetrievalResult) -> None:
    weak = RetrievalResult(chunk_id="weak", content="retention notes", score=0.5)
    chain = _chain(
        StaticRetriever([noise_chunk]),
        [FallbackStage(RecoveryStrategy.BROADER_SEARCH, StaticRetriever([weak]))],
    )
    result = await chain.run(QUERY)
    assert result.detection.degraded
    assert result.chunks[0].chunk_id == "weak"


async def test_chain_retries_transient_failures(
    corpus: dict[str, str], good_chunk: RetrievalResult
) -> None:
    flaky = FlakyRetriever(StaticRetriever([good_chunk]), failures=1)
    chain = FallbackChain(
        flaky, FailureDetector(), [], retries=2, backoff_base_ms=1.0, backoff_max_ms=2.0
    )
    result = await chain.run(QUERY)
    assert not result.detection.degraded
    assert flaky.calls == 2


async def test_chain_applies_reranker_and_respects_max_attempts(
    corpus: dict[str, str], noise_chunk: RetrievalResult
) -> None:
    never = StaticRetriever([noise_chunk])
    chain = _chain(
        StaticRetriever([noise_chunk]),
        [
            FallbackStage(
                RecoveryStrategy.RERANK, StaticRetriever([noise_chunk]), OverlapReranker()
            ),
            FallbackStage(RecoveryStrategy.BROADER_SEARCH, never),
        ],
        max_attempts=1,
    )
    result = await chain.run(QUERY)
    assert len(result.attempts) == 2
    assert never.calls == 0
