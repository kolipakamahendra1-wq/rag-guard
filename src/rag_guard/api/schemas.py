"""Request and response models for the HTTP API."""

from __future__ import annotations

from pydantic import BaseModel, Field

from rag_guard.types import RecoveryStrategy


class QueryRequest(BaseModel):
    """Body of ``POST /query``."""

    query: str = Field(..., min_length=1, max_length=2000)


class QueryResponse(BaseModel):
    """Answer plus the recovery metadata that produced it."""

    answer: str
    success: bool
    error: str | None = None
    degraded: bool
    fallback_used: RecoveryStrategy
    fallback_attempts: int
    faithfulness: float | None = None
    cost_usd: float
    latency_ms: float
    sources: list[str]
