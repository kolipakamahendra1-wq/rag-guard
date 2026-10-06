"""Anthropic Messages API provider."""

from __future__ import annotations

from typing import Any

import httpx

from rag_guard.exceptions import LLMProviderError
from rag_guard.llm.base import LLMProvider, TransientLLMError
from rag_guard.utils.async_utils import retry_async


class AnthropicProvider(LLMProvider):
    """Claude via ``/v1/messages``."""

    name = "anthropic"

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        *,
        base_url: str = "https://api.anthropic.com",
        client: httpx.AsyncClient | None = None,
        retries: int = 3,
        **kwargs: Any,
    ) -> None:
        super().__init__(model, client=client, **kwargs)
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.retries = retries

    async def _call(
        self, system_prompt: str, user_prompt: str, max_tokens: int, temperature: float
    ) -> tuple[str, int, int, str, list[dict[str, Any]]]:
        headers = {"content-type": "application/json", "anthropic-version": "2023-06-01"}
        if self.api_key:
            headers["x-api-key"] = self.api_key
        payload: dict[str, Any] = {
            "model": self.model,
            "system": system_prompt,
            "messages": [{"role": "user", "content": user_prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        async def attempt() -> dict[str, Any]:
            return await self._post(f"{self.base_url}/v1/messages", headers, payload)

        body = await retry_async(
            attempt,
            attempts=self.retries,
            base_ms=50.0,
            max_ms=2000.0,
            retry_on=(TransientLLMError,),
        )
        try:
            blocks = body["content"]
            text = "".join(b.get("text", "") for b in blocks if b.get("type") == "text")
            tools = [b for b in blocks if b.get("type") == "tool_use"]
            usage = body.get("usage", {})
            return (
                text,
                int(usage.get("input_tokens", 0)),
                int(usage.get("output_tokens", 0)),
                str(body.get("stop_reason") or "end_turn"),
                tools,
            )
        except (KeyError, TypeError, AttributeError) as exc:
            raise LLMProviderError(f"malformed response: {exc}", self.name) from exc
