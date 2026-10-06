"""RAG-Guard: Production-grade hybrid RAG failure detection and recovery framework."""

__version__ = "0.1.0"
__author__ = "Mahendra Kolipaka"
__email__ = "kolipakamahendra1@gmail.com"

from rag_guard.config import Settings
from rag_guard.exceptions import DetectionError, RAGGuardException, RecoveryError
from rag_guard.types import (
    DetectionResult,
    LLMResponse,
    RAGResult,
    RetrievalResult,
    RetrievalScore,
)

__all__ = [
    "Settings",
    "RAGGuardException",
    "DetectionError",
    "RecoveryError",
    "RetrievalResult",
    "RetrievalScore",
    "DetectionResult",
    "LLMResponse",
    "RAGResult",
]
