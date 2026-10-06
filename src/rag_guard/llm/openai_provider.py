"""OpenAI-compatible chat completion providers (OpenAI, Azure OpenAI, local servers)."""

from __future__ import annotations

from typing import Any

import httpx

from rag_guard.exceptions import LLMProviderError
from rag_guard.llm.base import LLMProvider, TransientLLMError
from rag_guard.utils.async_utils import retry_async


class OpenAIProvider(LLMProvider):
    """Chat Completions API (``/v1/chat/completions``)."""

    name = "openai"

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        *,
        base_url: str = "https://api.openai.com/v1",
        client: httpx.AsyncClient | None = None,
        retries: int = 3,
        **kwargs: Any,
    ) -> None:
        super().__init__(model, client=client, **kwargs)
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.retries = retries

    def _url(self) -> str:
        return f"{self.base_url}/chat/completions"

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    async def _call(
        self, system_prompt: str, user_prompt: str, max_tokens: int, temperature: float
    ) -> tuple[str, int, int, str, list[dict[str, Any]]]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        async def attempt() -> dict[str, Any]:
            return await self._post(self._url(), self._headers(), payload)

        body = await retry_async(
            attempt,
            attempts=self.retries,
            base_ms=50.0,
            max_ms=2000.0,
            retry_on=(TransientLLMError,),
        )
        try:
            choice = body["choices"][0]
            message = choice["message"]
            usage = body.get("usage", {})
            return (
                message.get("content") or "",
                int(usage.get("prompt_tokens", 0)),
                int(usage.get("completion_tokens", 0)),
                str(choice.get("finish_reason") or "stop"),
                list(message.get("tool_calls") or []),
            )
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMProviderError(f"malformed response: {exc}", self.name) from exc


class AzureOpenAIProvider(OpenAIProvider):
    """Azure OpenAI deployment; ``model`` is the deployment name."""

    name = "azure"

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        *,
        endpoint: str,
        api_version: str = "2024-06-01",
        **kwargs: Any,
    ) -> None:
        super().__init__(model, api_key, base_url=endpoint, **kwargs)
        self.api_version = api_version

    def _url(self) -> str:
        return f"{self.base_url}/openai/deployments/{self.model}/chat/completions?api-version={self.api_version}"

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["api-key"] = self.api_key
        return headers


class LocalProvider(OpenAIProvider):
    """Ollama / vLLM / any OpenAI-compatible local endpoint (no auth required)."""

    name = "local"

    def __init__(
        self, model: str, *, endpoint: str = "http://localhost:11434/v1", **kwargs: Any
    ) -> None:
        super().__init__(model, None, base_url=endpoint, **kwargs)
