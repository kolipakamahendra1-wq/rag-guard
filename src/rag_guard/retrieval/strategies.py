"""In-memory retrieval strategies (sparse, dense, hybrid, keyword)."""

from __future__ import annotations

from rag_guard.retrieval.base import Retriever
from rag_guard.types import RetrievalResult
from rag_guard.utils.text_utils import BM25Index, cosine, hashed_embedding, tokenize

_BM25_SATURATION = 5.0
_RRF_K = 60.0


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


class BM25Retriever(Retriever):
    """Sparse lexical retrieval; raw BM25 is squashed into [0, 1] via s / (s + 5)."""

    name = "bm25"

    def __init__(self, corpus: dict[str, str]) -> None:
        self.corpus = corpus
        self._index = BM25Index(corpus)

    async def retrieve(self, query: str, k: int = 5) -> list[RetrievalResult]:
        scored = [(doc_id, self._index.score(query, doc_id)) for doc_id in self.corpus]
        scored = [(d, s) for d, s in scored if s > 0.0]
        scored.sort(key=lambda x: (-x[1], x[0]))
        return [
            RetrievalResult(
                chunk_id=d,
                content=self.corpus[d],
                score=_clamp01(s / (s + _BM25_SATURATION)),
                source_doc_id=d,
                retrieval_method=self.name,
            )
            for d, s in scored[:k]
        ]


class DenseRetriever(Retriever):
    """Dense retrieval over deterministic hashed embeddings (offline stand-in for a vector DB)."""

    name = "dense"

    def __init__(self, corpus: dict[str, str], dim: int = 256) -> None:
        self.corpus = corpus
        self.dim = dim
        self._vectors = {d: hashed_embedding(t, dim) for d, t in corpus.items()}

    async def retrieve(self, query: str, k: int = 5) -> list[RetrievalResult]:
        q_vec = hashed_embedding(query, self.dim)
        scored = [(d, cosine(q_vec, v)) for d, v in self._vectors.items()]
        scored = [(d, s) for d, s in scored if s > 0.0]
        scored.sort(key=lambda x: (-x[1], x[0]))
        return [
            RetrievalResult(
                chunk_id=d,
                content=self.corpus[d],
                score=_clamp01(s),
                source_doc_id=d,
                retrieval_method=self.name,
            )
            for d, s in scored[:k]
        ]


class KeywordRetriever(Retriever):
    """Broad recall fallback: ranks by fraction of query terms present, ignoring term weighting."""

    name = "keyword"

    def __init__(self, corpus: dict[str, str]) -> None:
        self.corpus = corpus
        self._tokens = {d: set(tokenize(t)) for d, t in corpus.items()}

    async def retrieve(self, query: str, k: int = 5) -> list[RetrievalResult]:
        q = set(tokenize(query))
        if not q:
            return []
        scored = [(d, len(q & toks) / len(q)) for d, toks in self._tokens.items()]
        scored = [(d, s) for d, s in scored if s > 0.0]
        scored.sort(key=lambda x: (-x[1], x[0]))
        return [
            RetrievalResult(
                chunk_id=d,
                content=self.corpus[d],
                score=_clamp01(s),
                source_doc_id=d,
                retrieval_method=self.name,
            )
            for d, s in scored[:k]
        ]


class HybridRetriever(Retriever):
    """Reciprocal Rank Fusion over several retrievers; scores normalised to [0, 1]."""

    name = "hybrid"

    def __init__(self, retrievers: list[Retriever]) -> None:
        if not retrievers:
            raise ValueError("HybridRetriever requires at least one retriever")
        self.retrievers = retrievers

    async def retrieve(self, query: str, k: int = 5) -> list[RetrievalResult]:
        fused: dict[str, float] = {}
        by_id: dict[str, RetrievalResult] = {}
        for retriever in self.retrievers:
            for rank, result in enumerate(await retriever.retrieve(query, k * 2)):
                fused[result.chunk_id] = fused.get(result.chunk_id, 0.0) + 1.0 / (_RRF_K + rank + 1)
                by_id.setdefault(result.chunk_id, result)
        max_possible = len(self.retrievers) / (_RRF_K + 1)
        ranked = sorted(fused.items(), key=lambda x: (-x[1], x[0]))[:k]
        return [
            by_id[d].model_copy(
                update={"score": _clamp01(s / max_possible), "retrieval_method": self.name}
            )
            for d, s in ranked
        ]
