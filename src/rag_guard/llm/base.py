"""LLM provider abstraction shared by all backends."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import Any

import httpx

from rag_guard.exceptions import LLMProviderError
from rag_guard.types import LLMResponse, StreamingChunk, UsageMetrics

DEFAULT_SYSTEM_PROMPT = (
    "Answer the question using ONLY the provided context. "
    "If the context does not contain the answer, say you do not know."
)
RETRYABLE_STATUS = frozenset({408, 409, 425, 429, 500, 502, 503, 504})


class TransientLLMError(LLMProviderError):
    """Retryable provider failure (rate limit, 5xx, network)."""


def build_user_prompt(query: str, context: str) -> str:
    """Compose the grounded user prompt."""
    return f"Context:\n{context}\n\nQuestion: {query}"


class LLMProvider(ABC):
    """Base class for LLM backends. Subclasses implement ``_call`` only."""

    name: str = "llm"

    def __init__(
        self,
        model: str,
        *,
        client: httpx.AsyncClient | None = None,
        timeout_s: float = 30.0,
        input_cost_per_1k: float = 0.0,
        output_cost_per_1k: float = 0.0,
    ) -> None:
        self.model = model
        self._client = client
        self.timeout_s = timeout_s
        self.input_cost_per_1k = input_cost_per_1k
        self.output_cost_per_1k = output_cost_per_1k

    def compute_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Estimated USD cost for a call."""
        return (
            input_tokens / 1000.0 * self.input_cost_per_1k
            + output_tokens / 1000.0 * self.output_cost_per_1k
        )

    async def _post(
        self, url: str, headers: dict[str, str], payload: dict[str, Any]
    ) -> dict[str, Any]:
        """POST JSON and return the decoded body, mapping failures to typed errors."""
        try:
            if self._client is not None:
                response = await self._client.post(url, headers=headers, json=payload)
            else:
                async with httpx.AsyncClient(timeout=self.timeout_s) as client:
                    response = await client.post(url, headers=headers, json=payload)
        except httpx.HTTPError as exc:
            raise TransientLLMError(f"network error: {exc}", self.name) from exc
        if response.status_code in RETRYABLE_STATUS:
            raise TransientLLMError(f"HTTP {response.status_code}", self.name)
        if response.status_code >= 400:
            raise LLMProviderError(f"HTTP {response.status_code}: {response.text[:200]}", self.name)
        body = response.json()
        if not isinstance(body, dict):
            raise LLMProviderError("unexpected response shape", self.name)
        return body

    @abstractmethod
    async def _call(
        self, system_prompt: str, user_prompt: str, max_tokens: int, temperature: float
    ) -> tuple[str, int, int, str, list[dict[str, Any]]]:
        """Return (text, input_tokens, output_tokens, stop_reason, tool_calls)."""

    async def generate(
        self,
        query: str,
        context: str,
        *,
        system_prompt: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        trace_id: str = "",
    ) -> LLMResponse:
        """Generate a grounded answer for ``query`` given ``context``."""
        started = time.perf_counter()
        text, tokens_in, tokens_out, stop, tools = await self._call(
            system_prompt or DEFAULT_SYSTEM_PROMPT,
            build_user_prompt(query, context),
            max_tokens,
            temperature,
        )
        latency = (time.perf_counter() - started) * 1000.0
        return LLMResponse(
            text=text,
            tokens_used=UsageMetrics(
                input_tokens=tokens_in,
                output_tokens=tokens_out,
                total_tokens=tokens_in + tokens_out,
                cost_usd=self.compute_cost(tokens_in, tokens_out),
                provider=self.name,
            ),
            latency_ms=latency,
            trace_id=trace_id,
            tool_calls=tools,
            stop_reason=stop,
            model=self.model,
        )

    async def stream_generate(
        self, query: str, context: str, **kwargs: Any
    ) -> AsyncIterator[StreamingChunk]:
        """Yield the response as word-level chunks (response is fetched whole, then streamed)."""
        response = await self.generate(query, context, **kwargs)
        for index, token in enumerate(response.text.split(" ")):
            yield StreamingChunk(token=token, index=index)
