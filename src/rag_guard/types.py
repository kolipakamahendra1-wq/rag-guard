"""Core data models for RAG-Guard (Pydantic v2)."""

from datetime import datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


class SeverityLevel(str, Enum):
    """Degradation severity classification."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    CRITICAL = "critical"


class RecoveryStrategy(str, Enum):
    """Fallback recovery strategy types."""

    NONE = "none"
    RERANK = "rerank"
    DIVERSE_RETRIEVAL = "diverse_retrieval"
    BROADER_SEARCH = "broader_search"
    ALTERNATIVE_STRATEGY = "alternative_strategy"
    LLM_REPHRASE = "llm_rephrase"


class RetrievalResult(BaseModel):
    """Single retrieved document chunk."""

    chunk_id: str = Field(..., description="Unique identifier for this chunk")
    content: str = Field(..., description="The actual text content of the chunk")
    score: float = Field(..., ge=0.0, le=1.0, description="Relevance score [0.0, 1.0]")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Custom metadata")
    source_doc_id: str | None = Field(None, description="Parent document ID")
    chunk_index: int | None = Field(None, description="Chunk position in source")
    retrieval_method: str = Field(default="unknown", description="How this was retrieved")


class RetrievalScore(BaseModel):
    """Multi-dimensional quality score for retrieval results."""

    relevance: float = Field(..., ge=0.0, le=1.0, description="BM25 + semantic relevance")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Model confidence")
    signals: dict[str, float] = Field(
        default_factory=dict, description="Breakdown: bm25, semantic, token_overlap, etc."
    )
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class DetectionResult(BaseModel):
    """Failure detection outcome."""

    degraded: bool = Field(..., description="Is retrieval quality degraded?")
    severity: SeverityLevel = Field(..., description="Severity level")
    score: float = Field(..., ge=0.0, le=1.0, description="Overall quality score")
    recommended_strategy: RecoveryStrategy = Field(..., description="Which fallback to try first")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Detection confidence")
    details: dict[str, Any] = Field(default_factory=dict, description="Debug info")


class UsageMetrics(BaseModel):
    """Token usage and cost tracking."""

    input_tokens: int = Field(..., ge=0, description="Tokens in prompt")
    output_tokens: int = Field(..., ge=0, description="Tokens in response")
    total_tokens: int = Field(..., ge=0, description="Sum of input + output")
    cost_usd: float = Field(..., ge=0.0, description="Estimated cost in USD")
    provider: str = Field(..., description="LLM provider name")


class LLMResponse(BaseModel):
    """Structured response from LLM provider."""

    text: str = Field(..., description="Generated text response")
    tokens_used: UsageMetrics = Field(..., description="Token usage statistics")
    latency_ms: float = Field(..., ge=0.0, description="Total generation time")
    ttf_token_ms: float | None = Field(None, description="Time to first token")
    trace_id: str = Field(..., description="Unique trace ID for debugging")
    tool_calls: list[dict[str, Any]] = Field(
        default_factory=list, description="Extracted tool calls"
    )
    stop_reason: str = Field(default="end_turn", description="Why generation stopped")
    model: str = Field(..., description="Model identifier")


class RAGResult(BaseModel):
    """Complete RAG pipeline result with audit trail."""

    query: str = Field(..., description="Original user query")
    answer: str = Field(..., description="Final answer from LLM")
    retrieved_chunks: list[RetrievalResult] = Field(
        ..., description="Top-K chunks used for generation"
    )
    detection_result: DetectionResult | None = Field(None, description="Failure detection outcome")
    fallback_used: RecoveryStrategy = Field(
        default=RecoveryStrategy.NONE, description="Which recovery strategy was used"
    )
    fallback_attempts: int = Field(default=0, description="Number of fallback attempts")
    llm_response: LLMResponse | None = Field(None, description="LLM generation details")
    faithfulness: float | None = Field(None, ge=0.0, le=1.0, description="Answer grounding score")
    total_latency_ms: float = Field(..., ge=0.0, description="End-to-end latency")
    total_cost_usd: float = Field(..., ge=0.0, description="Total token cost")
    audit_trail: list[dict[str, Any]] = Field(
        default_factory=list, description="Step-by-step execution log"
    )
    success: bool = Field(..., description="Did the pipeline complete successfully?")
    error: str | None = Field(None, description="Error message if failed")
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class EvaluationMetrics(BaseModel):
    """Aggregated evaluation metrics for benchmarking."""

    recall: float = Field(..., ge=0.0, le=1.0, description="% correct docs retrieved")
    answer_accuracy: float = Field(default=0.0, ge=0.0, le=1.0, description="% answers correct")
    degraded_queries: int = Field(default=0, ge=0, description="Queries flagged degraded")
    faithfulness: float = Field(..., ge=0.0, le=1.0, description="% claims grounded in context")
    hallucination_rate: float = Field(..., ge=0.0, le=1.0, description="False claim rate")
    latency_p50_ms: float = Field(..., ge=0.0, description="Median latency")
    latency_p95_ms: float = Field(..., ge=0.0, description="95th percentile latency")
    latency_p99_ms: float = Field(..., ge=0.0, description="99th percentile latency")
    cost_per_query_usd: float = Field(..., ge=0.0, description="Avg cost per query")
    recovery_success_rate: float = Field(
        ..., ge=0.0, le=1.0, description="% of degraded queries that recovered"
    )


class BenchmarkResult(BaseModel):
    """Single benchmark run result."""

    approach_name: str = Field(..., description="Name of RAG approach tested")
    metrics: EvaluationMetrics = Field(..., description="Computed metrics")
    num_queries: int = Field(..., ge=1, description="Number of queries evaluated")
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class QAPair(BaseModel):
    """Question-answer pair with expected retrievals."""

    query_id: str = Field(..., description="Unique query identifier")
    question: str = Field(..., description="The user query")
    expected_answer: str = Field(..., description="Ground truth answer")
    expected_chunk_ids: list[str] = Field(..., description="Which chunks should be retrieved")
    topic: str = Field(..., description="Query topic/domain")
    difficulty: Literal["easy", "medium", "hard"] = Field(
        default="medium", description="Query difficulty"
    )


class PipelineState(BaseModel):
    """Immutable snapshot of pipeline execution state."""

    state_id: str = Field(..., description="Unique state identifier")
    stage: str = Field(..., description="Current pipeline stage")
    query: str = Field(..., description="Current query")
    retrieved_chunks: list[RetrievalResult] = Field(default_factory=list)
    detection_result: DetectionResult | None = Field(None)
    fallback_attempts: int = Field(default=0)
    accumulated_cost_usd: float = Field(default=0.0)
    elapsed_ms: float = Field(default=0.0)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = Field(default_factory=dict)


class StreamingChunk(BaseModel):
    """Single token from streaming LLM response."""

    token: str = Field(..., description="The token text")
    index: int = Field(..., ge=0, description="Token position")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
