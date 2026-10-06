"""FastAPI application factory."""

from __future__ import annotations

from fastapi import FastAPI

from rag_guard.api.schemas import QueryRequest, QueryResponse
from rag_guard.orchestration.pipeline import RAGPipeline


def create_app(pipeline: RAGPipeline) -> FastAPI:
    """Create an app serving ``pipeline``; the pipeline is injected so tests can mock it."""
    app = FastAPI(title="RAG-Guard", version="0.1.0")

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/query", response_model=QueryResponse)
    async def query(body: QueryRequest) -> QueryResponse:
        result = await pipeline.run(body.query)
        return QueryResponse(
            answer=result.answer,
            success=result.success,
            error=result.error,
            degraded=bool(result.detection_result and result.detection_result.degraded),
            fallback_used=result.fallback_used,
            fallback_attempts=result.fallback_attempts,
            faithfulness=result.faithfulness,
            cost_usd=result.total_cost_usd,
            latency_ms=result.total_latency_ms,
            sources=[c.chunk_id for c in result.retrieved_chunks],
        )

    return app
