"""Post-retrieval reranking."""

from __future__ import annotations

from abc import ABC, abstractmethod

from rag_guard.types import RetrievalResult
from rag_guard.utils.text_utils import token_overlap, tokenize


class Reranker(ABC):
    """Reorders and filters retrieved chunks."""

    @abstractmethod
    def rerank(
        self, query: str, chunks: list[RetrievalResult], top_k: int
    ) -> list[RetrievalResult]:
        """Return the best ``top_k`` chunks, best first."""


class OverlapReranker(Reranker):
    """Blends the original retrieval score with query-term overlap."""

    def __init__(self, overlap_weight: float = 0.5) -> None:
        if not 0.0 <= overlap_weight <= 1.0:
            raise ValueError("overlap_weight must be in [0, 1]")
        self.overlap_weight = overlap_weight

    def rerank(
        self, query: str, chunks: list[RetrievalResult], top_k: int
    ) -> list[RetrievalResult]:
        q_tokens = tokenize(query)
        rescored = []
        for chunk in chunks:
            overlap = token_overlap(q_tokens, tokenize(chunk.content))
            blended = (1.0 - self.overlap_weight) * chunk.score + self.overlap_weight * overlap
            rescored.append(chunk.model_copy(update={"score": max(0.0, min(1.0, blended))}))
        rescored.sort(key=lambda c: (-c.score, c.chunk_id))
        return rescored[:top_k]
