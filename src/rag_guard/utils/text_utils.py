"""Deterministic, dependency-free text utilities."""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter

_TOKEN_RE = re.compile(r"[a-z0-9]+")
STOPWORDS: frozenset[str] = frozenset(
    [
        "a",
        "an",
        "the",
        "of",
        "to",
        "in",
        "on",
        "for",
        "and",
        "or",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "it",
        "its",
        "this",
        "that",
        "with",
        "as",
        "by",
        "at",
        "from",
        "what",
        "which",
        "who",
        "how",
        "why",
        "when",
        "where",
        "do",
        "does",
        "did",
        "can",
        "could",
        "should",
        "would",
        "will",
    ]
)


def tokenize(text: str, drop_stopwords: bool = True) -> list[str]:
    """Lowercase alphanumeric tokenization with optional stopword removal."""
    tokens = _TOKEN_RE.findall(text.lower())
    if drop_stopwords:
        return [t for t in tokens if t not in STOPWORDS]
    return tokens


def split_sentences(text: str) -> list[str]:
    """Split text into non-empty sentences."""
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p for p in parts if p]


def token_overlap(query_tokens: list[str], doc_tokens: list[str]) -> float:
    """Fraction of unique query tokens present in the document."""
    q = set(query_tokens)
    if not q:
        return 0.0
    return len(q & set(doc_tokens)) / len(q)


def hashed_embedding(text: str, dim: int = 256) -> list[float]:
    """Deterministic L2-normalised hashed bag-of-words embedding."""
    vec = [0.0] * dim
    for tok, count in Counter(tokenize(text)).items():
        digest = hashlib.md5(tok.encode("utf-8")).digest()
        idx = int.from_bytes(digest[:4], "big") % dim
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vec[idx] += sign * (1.0 + math.log(count))
    norm = math.sqrt(sum(v * v for v in vec))
    if norm == 0.0:
        return vec
    return [v / norm for v in vec]


def cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity of two equal-length vectors."""
    if len(a) != len(b):
        raise ValueError("vectors must have equal length")
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def percentile(values: list[float], pct: float) -> float:
    """Linear-interpolated percentile (pct in [0, 100])."""
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = (len(ordered) - 1) * pct / 100.0
    lo = math.floor(rank)
    hi = math.ceil(rank)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (rank - lo)


class BM25Index:
    """Minimal Okapi BM25 index over an in-memory corpus."""

    def __init__(self, corpus: dict[str, str], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.doc_tokens: dict[str, list[str]] = {d: tokenize(t) for d, t in corpus.items()}
        self.doc_freq: Counter[str] = Counter()
        for toks in self.doc_tokens.values():
            self.doc_freq.update(set(toks))
        n = len(self.doc_tokens)
        self.avg_len = (sum(len(t) for t in self.doc_tokens.values()) / n) if n else 0.0

    def score(self, query: str, doc_id: str) -> float:
        """Raw BM25 score of one document for a query."""
        toks = self.doc_tokens[doc_id]
        if not toks:
            return 0.0
        n = len(self.doc_tokens)
        tf = Counter(toks)
        total = 0.0
        for q in set(tokenize(query)):
            if q not in tf:
                continue
            df = self.doc_freq[q]
            idf = math.log(1.0 + (n - df + 0.5) / (df + 0.5))
            denom = tf[q] + self.k1 * (1.0 - self.b + self.b * len(toks) / (self.avg_len or 1.0))
            total += idf * tf[q] * (self.k1 + 1.0) / denom
        return total
