"""End-to-end RAG pipeline: retrieve, detect, recover, generate, verify."""

from __future__ import annotations

import time

from rag_guard.detection.metrics import faithfulness_score
from rag_guard.exceptions import RAGGuardException
from rag_guard.llm.base import LLMProvider
from rag_guard.logging import get_logger
from rag_guard.orchestration.recovery import AdaptiveRecovery
from rag_guard.orchestration.state import StateManager
from rag_guard.retrieval.fallback import FallbackChain
from rag_guard.telemetry import Telemetry
from rag_guard.types import RAGResult, RecoveryStrategy

_log = get_logger(__name__)


class RAGPipeline:
    """Runs a query through a fallback chain and an LLM, returning a fully audited result."""

    def __init__(
        self,
        chain: FallbackChain,
        llm: LLMProvider,
        *,
        recovery: AdaptiveRecovery | None = None,
        max_tokens: int = 512,
        temperature: float = 0.0,
        check_faithfulness: bool = True,
        max_query_cost_usd: float | None = None,
    ) -> None:
        self.chain = chain
        self.llm = llm
        self.recovery = recovery
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.check_faithfulness = check_faithfulness
        self.max_query_cost_usd = max_query_cost_usd

    async def run(self, query: str) -> RAGResult:
        """Answer ``query``; never raises for provider/retrieval failures, returns ``success=False``."""
        telemetry = Telemetry()
        state = StateManager(query, telemetry.trace_id)
        started = time.perf_counter()
        state.record("start")

        def elapsed_ms() -> float:
            return (time.perf_counter() - started) * 1000.0

        if self.recovery is not None:
            self.chain.stages = self.recovery.order(self.chain.stages)

        with telemetry.span("retrieve"):
            retrieval = await self.chain.run(query)
        attempts = max(0, len(retrieval.attempts) - 1)
        if self.recovery is not None and attempts > 0:
            self.recovery.record(retrieval.strategy_used, retrieval.recovered)
        state.record(
            "retrieved",
            retrieved_chunks=retrieval.chunks,
            detection_result=retrieval.detection,
            fallback_attempts=attempts,
            elapsed_ms=elapsed_ms(),
        )

        base = {
            "query": query,
            "retrieved_chunks": retrieval.chunks,
            "detection_result": retrieval.detection,
            "fallback_used": retrieval.strategy_used,
            "fallback_attempts": attempts,
        }

        if not retrieval.chunks:
            state.record("failed", elapsed_ms=elapsed_ms())
            return RAGResult(
                **base,
                answer="I could not find relevant information.",
                total_latency_ms=elapsed_ms(),
                total_cost_usd=0.0,
                audit_trail=state.audit_trail(),
                success=False,
                error="no chunks retrieved",
            )

        context = "\n\n".join(c.content for c in retrieval.chunks)
        try:
            with telemetry.span("generate"):
                response = await self.llm.generate(
                    query,
                    context,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    trace_id=telemetry.trace_id,
                )
        except RAGGuardException as exc:
            _log.error("generation_failed", trace_id=telemetry.trace_id, error=str(exc))
            state.record("failed", elapsed_ms=elapsed_ms())
            return RAGResult(
                **base,
                answer="",
                total_latency_ms=elapsed_ms(),
                total_cost_usd=0.0,
                audit_trail=state.audit_trail(),
                success=False,
                error=str(exc),
            )

        cost = response.tokens_used.cost_usd
        faith = faithfulness_score(response.text, context) if self.check_faithfulness else None
        state.record("generated", accumulated_cost_usd=cost, elapsed_ms=elapsed_ms())
        over_budget = self.max_query_cost_usd is not None and cost > self.max_query_cost_usd
        return RAGResult(
            **base,
            answer=response.text,
            llm_response=response,
            faithfulness=faith,
            total_latency_ms=elapsed_ms(),
            total_cost_usd=cost,
            audit_trail=state.audit_trail(),
            success=not over_budget,
            error="query cost budget exceeded" if over_budget else None,
        )


__all__ = ["RAGPipeline", "RecoveryStrategy"]
