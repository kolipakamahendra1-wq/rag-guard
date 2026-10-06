"""Retrieval failure detection."""

from rag_guard.detection.detector import FailureDetector
from rag_guard.detection.metrics import faithfulness_score, hallucination_rate
from rag_guard.detection.scorer import RetrievalScorer

__all__ = ["FailureDetector", "RetrievalScorer", "faithfulness_score", "hallucination_rate"]
