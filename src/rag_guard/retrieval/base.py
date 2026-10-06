"""Retriever abstraction."""

from __future__ import annotations

from abc import ABC, abstractmethod

from rag_guard.types import RetrievalResult


class Retriever(ABC):
    """Async interface implemented by every retrieval backend."""

    name: str = "retriever"

    @abstractmethod
    async def retrieve(self, query: str, k: int = 5) -> list[RetrievalResult]:
        """Return up to ``k`` chunks ordered by descending score (scores in [0, 1])."""

    async def batch_retrieve(self, queries: list[str], k: int = 5) -> list[list[RetrievalResult]]:
        """Retrieve for several queries sequentially."""
        return [await self.retrieve(q, k) for q in queries]
