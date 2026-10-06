"""Custom exception hierarchy for RAG-Guard."""


class RAGGuardException(Exception):
    """Base exception for all RAG-Guard errors."""

    def __init__(self, message: str, code: str | None = None) -> None:
        self.message = message
        self.code = code or "UNKNOWN_ERROR"
        super().__init__(self.message)


class ConfigError(RAGGuardException):
    """Configuration validation or missing required settings."""

    def __init__(self, message: str) -> None:
        super().__init__(message, "CONFIG_ERROR")


class DetectionError(RAGGuardException):
    """Failure detection pipeline error."""

    def __init__(self, message: str) -> None:
        super().__init__(message, "DETECTION_ERROR")


class RetrievalError(RAGGuardException):
    """Retrieval operation failed or degraded."""

    def __init__(self, message: str, strategy: str | None = None) -> None:
        self.strategy = strategy
        super().__init__(message, "RETRIEVAL_ERROR")


class LLMProviderError(RAGGuardException):
    """LLM provider operation failed."""

    def __init__(self, message: str, provider: str | None = None) -> None:
        self.provider = provider
        super().__init__(message, "LLM_PROVIDER_ERROR")


class RecoveryError(RAGGuardException):
    """Fallback recovery strategy failed or exhausted."""

    def __init__(self, message: str, strategy_chain: list[str] | None = None) -> None:
        self.strategy_chain = strategy_chain or []
        super().__init__(message, "RECOVERY_ERROR")


class EvaluationError(RAGGuardException):
    """Metric evaluation or benchmarking error."""

    def __init__(self, message: str, metric_name: str | None = None) -> None:
        self.metric_name = metric_name
        super().__init__(message, "EVALUATION_ERROR")


class StreamingError(RAGGuardException):
    """Error in streaming token generation."""

    def __init__(self, message: str) -> None:
        super().__init__(message, "STREAMING_ERROR")


class OperationTimeoutError(RAGGuardException):
    """Operation exceeded configured timeout."""

    def __init__(self, message: str, timeout_ms: float | None = None) -> None:
        self.timeout_ms = timeout_ms
        super().__init__(message, "TIMEOUT_ERROR")


class RetryExhaustedError(RAGGuardException):
    """Exponential backoff retry strategy exhausted all attempts."""

    def __init__(self, message: str, attempts: int | None = None) -> None:
        self.attempts = attempts
        super().__init__(message, "RETRY_EXHAUSTED_ERROR")
