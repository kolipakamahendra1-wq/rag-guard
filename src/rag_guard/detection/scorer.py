"""Multi-signal retrieval quality scoring."""

from __future__ import annotations

from rag_guard.types import RetrievalResult, RetrievalScore
from rag_guard.utils.text_utils import token_overlap, tokenize


class RetrievalScorer:
    """Scores retrieved chunks against a query using label-free signals."""

    def __init__(
        self,
        coverage_weight: float = 0.5,
        top_weight: float = 0.3,
        mean_weight: float = 0.2,
    ) -> None:
        total = coverage_weight + top_weight + mean_weight
        if total <= 0:
            raise ValueError("weights must sum to a positive number")
        self.coverage_weight = coverage_weight / total
        self.top_weight = top_weight / total
        self.mean_weight = mean_weight / total

    def score(self, query: str, chunks: list[RetrievalResult]) -> RetrievalScore:
        """Compute quality signals.

        Signals: ``coverage`` (query terms covered by the union of chunks),
        ``best_overlap`` (best single-chunk term overlap), ``mean_retrieval_score``.
        """
        if not chunks:
            return RetrievalScore(
                relevance=0.0,
                confidence=1.0,
                signals={"coverage": 0.0, "best_overlap": 0.0, "mean_retrieval_score": 0.0},
            )
        q_tokens = tokenize(query)
        union: list[str] = []
        best = 0.0
        for chunk in chunks:
            toks = tokenize(chunk.content)
            union.extend(toks)
            best = max(best, token_overlap(q_tokens, toks))
        coverage = token_overlap(q_tokens, union)
        mean_score = sum(c.score for c in chunks) / len(chunks)
        relevance = (
            self.coverage_weight * coverage + self.top_weight * best + self.mean_weight * mean_score
        )
        confidence = min(1.0, 0.4 + 0.15 * len(chunks))
        return RetrievalScore(
            relevance=max(0.0, min(1.0, relevance)),
            confidence=confidence,
            signals={
                "coverage": coverage,
                "best_overlap": best,
                "mean_retrieval_score": mean_score,
            },
        )
