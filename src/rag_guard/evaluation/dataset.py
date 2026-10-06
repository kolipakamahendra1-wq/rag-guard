"""Deterministic synthetic QA dataset with a labelled corpus."""

from __future__ import annotations

from dataclasses import dataclass

from rag_guard.types import QAPair

_PREFIXES = ["Zor", "Kal", "Vex", "Tor", "Bra", "Nim", "Qua", "Syl", "Dra", "Mer"]
_SUFFIXES = ["vex", "lon", "dex", "mir", "tak", "rin", "phos", "nel", "gor", "zia"]
_ATTRIBUTES = [
    "retention period",
    "storage quota",
    "rate limit",
    "replication factor",
    "timeout budget",
    "cache capacity",
]
_TEAMS = ["platform", "edge", "data", "security", "billing"]


@dataclass(frozen=True)
class QADataset:
    """A corpus plus labelled questions over it."""

    corpus: dict[str, str]
    pairs: list[QAPair]

    @classmethod
    def synthetic(cls, n_entities: int = 60, n_queries: int | None = None) -> QADataset:
        """Build ``n_entities`` unique fact documents and one question per fact (up to ``n_queries``)."""
        if not 1 <= n_entities <= len(_PREFIXES) * len(_SUFFIXES):
            raise ValueError("n_entities out of range")
        corpus: dict[str, str] = {}
        pairs: list[QAPair] = []
        for i in range(n_entities):
            name = _PREFIXES[i % len(_PREFIXES)] + _SUFFIXES[(i // len(_PREFIXES)) % len(_SUFFIXES)]
            attr = _ATTRIBUTES[i % len(_ATTRIBUTES)]
            team = _TEAMS[i % len(_TEAMS)]
            value = f"{(i * 7) % 90 + 10} units"
            doc_id = f"doc-{i:03d}"
            corpus[doc_id] = (
                f"The {attr} of {name} is {value}. "
                f"The {name} service is maintained by the {team} team and released in {2015 + i % 10}."
            )
            pairs.append(
                QAPair(
                    query_id=f"q-{i:03d}",
                    question=f"What is the {attr} of {name}?",
                    expected_answer=value,
                    expected_chunk_ids=[doc_id],
                    topic=attr,
                    difficulty=("easy", "medium", "hard")[i % 3],
                )
            )
        if n_queries is not None:
            pairs = pairs[:n_queries]
        return cls(corpus=corpus, pairs=pairs)
