"""Deterministic answer-grounding metrics (no external calls)."""

from __future__ import annotations

from rag_guard.utils.text_utils import split_sentences, token_overlap, tokenize


def faithfulness_score(answer: str, context: str, support_threshold: float = 0.6) -> float:
    """Fraction of answer sentences whose content terms are supported by the context."""
    sentences = split_sentences(answer)
    if not sentences:
        return 0.0
    ctx_tokens = tokenize(context)
    supported = 0
    for sentence in sentences:
        toks = tokenize(sentence)
        if not toks:
            supported += 1
            continue
        if token_overlap(toks, ctx_tokens) >= support_threshold:
            supported += 1
    return supported / len(sentences)


def hallucination_rate(answer: str, context: str, support_threshold: float = 0.6) -> float:
    """Complement of faithfulness: share of unsupported answer sentences."""
    return 1.0 - faithfulness_score(answer, context, support_threshold)
