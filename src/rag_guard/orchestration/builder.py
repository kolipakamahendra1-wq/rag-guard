"""Convenience constructors for baseline and guarded pipelines."""

from __future__ import annotations

from rag_guard.config import Settings
from rag_guard.detection.detector import FailureDetector
from rag_guard.llm.base import LLMProvider
from rag_guard.orchestration.pipeline import RAGPipeline
from rag_guard.orchestration.recovery import AdaptiveRecovery
from rag_guard.retrieval.base import Retriever
from rag_guard.retrieval.fallback import FallbackChain, FallbackStage
from rag_guard.retrieval.reranking import OverlapReranker
from rag_guard.retrieval.strategies import (
    BM25Retriever,
    DenseRetriever,
    HybridRetriever,
    KeywordRetriever,
)
from rag_guard.types import RecoveryStrategy


def build_pipeline(
    corpus: dict[str, str],
    llm: LLMProvider,
    settings: Settings | None = None,
    *,
    primary: Retriever | None = None,
    guarded: bool = True,
    adaptive: bool = False,
) -> RAGPipeline:
    """Build a pipeline over ``corpus``.

    ``guarded=False`` yields the baseline: same primary retriever and LLM, but no fallback stages.
    """
    cfg = settings or Settings()
    primary_retriever = primary or DenseRetriever(corpus)
    bm25 = BM25Retriever(corpus)
    stages: list[FallbackStage] = []
    if guarded and cfg.enable_fallback:
        stages = [
            FallbackStage(RecoveryStrategy.ALTERNATIVE_STRATEGY, bm25),
            FallbackStage(
                RecoveryStrategy.DIVERSE_RETRIEVAL,
                HybridRetriever([bm25, primary_retriever]),
                OverlapReranker(),
            ),
            FallbackStage(RecoveryStrategy.BROADER_SEARCH, KeywordRetriever(corpus)),
        ]
    detector = FailureDetector(
        threshold=cfg.degradation_threshold,
        drift_window=cfg.drift_detection_window,
        drift_threshold=cfg.drift_threshold,
    )
    chain = FallbackChain(
        primary_retriever,
        detector,
        stages,
        top_k=cfg.retrieval_top_k,
        max_attempts=cfg.max_fallback_attempts,
        timeout_ms=cfg.retrieval_timeout_ms,
        backoff_base_ms=cfg.backoff_base_ms,
        backoff_max_ms=cfg.backoff_max_ms,
    )
    return RAGPipeline(
        chain,
        llm,
        recovery=AdaptiveRecovery() if adaptive else None,
        max_tokens=cfg.max_tokens,
        temperature=cfg.temperature,
        check_faithfulness=cfg.enable_faithfulness_check,
        max_query_cost_usd=cfg.max_query_cost_usd,
    )
