"""Deterministic retriever stand-ins."""

from __future__ import annotations

import hashlib
import math
import random

from rag_guard.exceptions import RetrievalError
from rag_guard.retrieval.base import Retriever
from rag_guard.types import RetrievalResult
from rag_guard.utils.text_utils import cosine, hashed_embedding


class StaticRetriever(Retriever):
    """Always returns the same chunks."""

    name = "static"

    def __init__(self, results: list[RetrievalResult]) -> None:
        self.results = results
        self.calls = 0

    async def retrieve(self, query: str, k: int = 5) -> list[RetrievalResult]:
        self.calls += 1
        return self.results[:k]


class FailingRetriever(Retriever):
    """Always raises ``RetrievalError``."""

    name = "failing"

    def __init__(self) -> None:
        self.calls = 0

    async def retrieve(self, query: str, k: int = 5) -> list[RetrievalResult]:
        self.calls += 1
        raise RetrievalError("simulated backend outage", self.name)


class DriftedDenseRetriever(Retriever):
    """Dense retriever whose query embeddings are perturbed by deterministic noise.

    Simulates embedding drift (model upgrade mismatch, stale index). ``noise`` is the noise norm
    relative to the unit-norm query vector; 0.0 behaves exactly like ``DenseRetriever``.
    """

    name = "dense-drifted"

    def __init__(self, corpus: dict[str, str], noise: float, dim: int = 256) -> None:
        if noise < 0.0:
            raise ValueError("noise must be >= 0")
        self.corpus = corpus
        self.noise = noise
        self.dim = dim
        self._vectors = {d: hashed_embedding(t, dim) for d, t in corpus.items()}

    def _noisy_query(self, query: str) -> list[float]:
        seed = int.from_bytes(hashlib.md5(query.encode("utf-8")).digest()[:8], "big")
        rng = random.Random(seed)
        base = hashed_embedding(query, self.dim)
        raw = [rng.gauss(0.0, 1.0) for _ in range(self.dim)]
        norm = math.sqrt(sum(v * v for v in raw)) or 1.0
        return [b + self.noise * r / norm for b, r in zip(base, raw, strict=True)]

    async def retrieve(self, query: str, k: int = 5) -> list[RetrievalResult]:
        q_vec = self._noisy_query(query)
        scored = [(d, cosine(q_vec, v)) for d, v in self._vectors.items()]
        scored = [(d, s) for d, s in scored if s > 0.0]
        scored.sort(key=lambda x: (-x[1], x[0]))
        return [
            RetrievalResult(
                chunk_id=d,
                content=self.corpus[d],
                score=max(0.0, min(1.0, s)),
                source_doc_id=d,
                retrieval_method=self.name,
            )
            for d, s in scored[:k]
        ]


class FlakyRetriever(Retriever):
    """Fails the first ``failures`` calls, then delegates to ``inner``."""

    name = "flaky"

    def __init__(self, inner: Retriever, failures: int = 1) -> None:
        self.inner = inner
        self.failures = failures
        self.calls = 0

    async def retrieve(self, query: str, k: int = 5) -> list[RetrievalResult]:
        self.calls += 1
        if self.calls <= self.failures:
            raise RetrievalError("transient failure", self.name)
        return await self.inner.retrieve(query, k)
