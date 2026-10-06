"""Configuration management for RAG-Guard (Pydantic Settings v2)."""

from pydantic import Field, ValidationInfo, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Global configuration for RAG-Guard, loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Application
    app_name: str = Field(default="rag-guard", description="Application name")
    app_version: str = Field(default="0.1.0", description="Application version")
    debug: bool = Field(default=False, description="Enable debug mode")
    environment: str = Field(
        default="development", description="Environment: development, staging, production"
    )

    # Server
    server_host: str = Field(default="0.0.0.0", description="Server host")
    server_port: int = Field(default=8000, ge=1, le=65535, description="Server port")

    # LLM Providers
    openai_api_key: str | None = Field(default=None, description="OpenAI API key")
    anthropic_api_key: str | None = Field(default=None, description="Anthropic API key")
    azure_api_key: str | None = Field(default=None, description="Azure API key")
    azure_endpoint: str | None = Field(default=None, description="Azure endpoint URL")
    gemini_api_key: str | None = Field(default=None, description="Google Gemini API key")
    local_llm_endpoint: str | None = Field(default=None, description="Local LLM endpoint")
    default_llm_provider: str = Field(
        default="anthropic", description="Default LLM provider to use"
    )
    default_model: str = Field(default="claude-sonnet-5-5", description="Default model ID")

    # Detection & Scoring
    degradation_threshold: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description="Quality score threshold below which retrieval is degraded",
    )
    drift_detection_window: int = Field(
        default=10, ge=1, description="Number of recent queries to track for drift"
    )
    drift_threshold: float = Field(
        default=0.15,
        ge=0.0,
        le=1.0,
        description="Relative score drop to trigger drift detection",
    )

    # Retrieval
    retrieval_top_k: int = Field(default=5, ge=1, description="Number of chunks to retrieve")
    retrieval_timeout_ms: float = Field(
        default=5000.0, ge=100.0, description="Retrieval operation timeout"
    )
    chunk_overlap: int = Field(default=0, ge=0, description="Token overlap between chunks")

    # Reranking
    enable_reranking: bool = Field(default=True, description="Enable post-retrieval reranking")
    rerank_top_k: int = Field(default=3, ge=1, description="Keep top-K after reranking")
    rerank_threshold: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Minimum score to keep after reranking"
    )

    # Fallback & Recovery
    enable_fallback: bool = Field(default=True, description="Enable fallback chains")
    max_fallback_attempts: int = Field(default=3, ge=1, description="Max fallback attempts")
    fallback_strategies: list[str] = Field(
        default=["rerank", "diverse_retrieval", "broader_search"],
        description="Ordered list of fallback strategies to try",
    )
    backoff_base_ms: float = Field(
        default=100.0, ge=10.0, description="Base exponential backoff in ms"
    )
    backoff_max_ms: float = Field(
        default=5000.0, ge=100.0, description="Max exponential backoff in ms"
    )

    # Generation
    llm_timeout_ms: float = Field(default=30000.0, ge=1000.0, description="LLM call timeout")
    max_tokens: int = Field(default=1024, ge=1, description="Max tokens in LLM response")
    temperature: float = Field(
        default=0.3, ge=0.0, le=2.0, description="LLM temperature for determinism"
    )

    # Evaluation
    enable_faithfulness_check: bool = Field(
        default=True, description="Enable post-generation faithfulness verification"
    )
    faithfulness_model: str = Field(
        default="claude-haiku-4-5-20251001", description="Model for faithfulness scoring"
    )

    # Cost Tracking
    track_token_usage: bool = Field(default=True, description="Enable token usage tracking")
    max_query_cost_usd: float | None = Field(
        default=None, description="Max cost per query (None = unlimited)"
    )
    max_session_cost_usd: float | None = Field(
        default=None, description="Max cost per session (None = unlimited)"
    )

    # Observability
    enable_structured_logging: bool = Field(default=True, description="Enable structured logging")
    log_level: str = Field(default="INFO", description="Logging level")
    enable_tracing: bool = Field(default=True, description="Enable OpenTelemetry tracing")
    otel_exporter_endpoint: str | None = Field(
        default=None, description="OpenTelemetry exporter endpoint"
    )

    # Redis (for caching, state management)
    redis_url: str | None = Field(default=None, description="Redis connection URL")
    redis_cache_ttl_seconds: int = Field(
        default=3600, ge=1, description="Cache TTL for embeddings, etc."
    )

    # Testing
    use_mock_providers: bool = Field(default=False, description="Use mock providers in tests")
    random_seed: int | None = Field(default=42, description="Random seed for reproducibility")

    @field_validator("environment")
    @classmethod
    def validate_environment(cls, v: str) -> str:
        """Validate environment is one of allowed values."""
        allowed = {"development", "staging", "production"}
        if v not in allowed:
            raise ValueError(f"Environment must be one of {allowed}")
        return v

    @field_validator("default_llm_provider")
    @classmethod
    def validate_provider(cls, v: str) -> str:
        """Validate provider is supported."""
        allowed = {"openai", "anthropic", "azure", "gemini", "local"}
        if v not in allowed:
            raise ValueError(f"Provider must be one of {allowed}")
        return v

    @field_validator("backoff_max_ms")
    @classmethod
    def validate_backoff_max(cls, v: float, info: ValidationInfo) -> float:
        """Ensure backoff_max >= backoff_base."""
        if "backoff_base_ms" in info.data and v < info.data["backoff_base_ms"]:
            raise ValueError("backoff_max_ms must be >= backoff_base_ms")
        return v

    def get_provider_key(self, provider: str) -> str | None:
        """Get API key for a provider."""
        keys = {
            "openai": self.openai_api_key,
            "anthropic": self.anthropic_api_key,
            "azure": self.azure_api_key,
            "gemini": self.gemini_api_key,
        }
        return keys.get(provider)

    def is_production(self) -> bool:
        """Check if running in production."""
        return self.environment == "production"

    def is_development(self) -> bool:
        """Check if running in development."""
        return self.environment == "development"
