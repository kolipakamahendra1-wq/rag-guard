"""Deterministic LLM stand-ins.

``ExtractiveMockLLM`` simulates a well-behaved model: it answers from the best-matching context
sentence, but when the context does not cover the question it confabulates a fluent answer that is
not grounded in the context. This reproduces the failure mode RAG-Guard defends against without
any network access. Benchmark results using it measure the *pipeline*, not a real model.
"""

from __future__ import annotations

from typing import Any

from rag_guard.exceptions import LLMProviderError
from rag_guard.llm.base import LLMProvider
from rag_guard.utils.text_utils import split_sentences, token_overlap, tokenize

CONFABULATION = (
    "Reportedly the figure is approximately forty two according to broad industry consensus."
)


class ExtractiveMockLLM(LLMProvider):
    """Answers from context when it covers the question, otherwise confabulates."""

    name = "mock"

    def __init__(
        self, model: str = "mock-extractive", min_coverage: float = 0.5, **kwargs: Any
    ) -> None:
        kwargs.setdefault("input_cost_per_1k", 0.003)
        kwargs.setdefault("output_cost_per_1k", 0.015)
        super().__init__(model, **kwargs)
        self.min_coverage = min_coverage
        self.calls = 0

    async def _call(
        self, system_prompt: str, user_prompt: str, max_tokens: int, temperature: float
    ) -> tuple[str, int, int, str, list[dict[str, Any]]]:
        self.calls += 1
        context, _, question = user_prompt.partition("\n\nQuestion: ")
        context = context.removeprefix("Context:\n")
        q_tokens = tokenize(question)
        best_sentence, best_cov = "", 0.0
        for sentence in split_sentences(context):
            cov = token_overlap(q_tokens, tokenize(sentence))
            if cov > best_cov:
                best_sentence, best_cov = sentence, cov
        text = best_sentence if best_cov >= self.min_coverage else CONFABULATION
        return text, len((system_prompt + user_prompt).split()), len(text.split()), "end_turn", []


class FailingLLM(LLMProvider):
    """Always raises a provider error."""

    name = "failing"

    def __init__(self, model: str = "mock-failing", **kwargs: Any) -> None:
        super().__init__(model, **kwargs)

    async def _call(
        self, system_prompt: str, user_prompt: str, max_tokens: int, temperature: float
    ) -> tuple[str, int, int, str, list[dict[str, Any]]]:
        raise LLMProviderError("simulated provider outage", self.name)
