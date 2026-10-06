"""Runtime provider resolution from settings."""

from __future__ import annotations

import httpx

from rag_guard.config import Settings
from rag_guard.exceptions import ConfigError
from rag_guard.llm.anthropic_provider import AnthropicProvider
from rag_guard.llm.base import LLMProvider
from rag_guard.llm.openai_provider import AzureOpenAIProvider, LocalProvider, OpenAIProvider

SUPPORTED_PROVIDERS = ("anthropic", "openai", "azure", "local")


def create_provider(
    name: str,
    settings: Settings,
    *,
    model: str | None = None,
    client: httpx.AsyncClient | None = None,
) -> LLMProvider:
    """Build the provider called ``name`` using credentials and endpoints from ``settings``."""
    chosen = model or settings.default_model
    timeout_s = settings.llm_timeout_ms / 1000.0
    if name == "anthropic":
        return AnthropicProvider(
            chosen, settings.anthropic_api_key, client=client, timeout_s=timeout_s
        )
    if name == "openai":
        return OpenAIProvider(chosen, settings.openai_api_key, client=client, timeout_s=timeout_s)
    if name == "azure":
        if not settings.azure_endpoint:
            raise ConfigError("azure provider requires AZURE_ENDPOINT")
        return AzureOpenAIProvider(
            chosen,
            settings.azure_api_key,
            endpoint=settings.azure_endpoint,
            client=client,
            timeout_s=timeout_s,
        )
    if name == "local":
        if not settings.local_llm_endpoint:
            raise ConfigError("local provider requires LOCAL_LLM_ENDPOINT")
        return LocalProvider(
            chosen, endpoint=settings.local_llm_endpoint, client=client, timeout_s=timeout_s
        )
    raise ConfigError(f"unsupported provider '{name}'; choose from {SUPPORTED_PROVIDERS}")
