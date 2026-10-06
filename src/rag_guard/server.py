"""ASGI entrypoint: ``uvicorn rag_guard.server:app``.

Serves a demo corpus. Uses the configured LLM provider when its API key is present; otherwise
falls back to the offline mock model so the service always starts.
"""

from __future__ import annotations

from fastapi import FastAPI

from mocks.mock_llms import ExtractiveMockLLM
from rag_guard.api import create_app
from rag_guard.config import Settings
from rag_guard.evaluation.dataset import QADataset
from rag_guard.llm import LLMProvider, create_provider
from rag_guard.logging import configure_logging, get_logger
from rag_guard.orchestration.builder import build_pipeline


def select_llm(settings: Settings) -> LLMProvider:
    """Real provider if credentials are configured, otherwise the offline mock."""
    name = settings.default_llm_provider
    if settings.use_mock_providers:
        return ExtractiveMockLLM()
    if name in {"anthropic", "openai", "azure"} and not settings.get_provider_key(name):
        return ExtractiveMockLLM()
    if name == "local" and not settings.local_llm_endpoint:
        return ExtractiveMockLLM()
    return create_provider(name, settings)


def create_default_app(settings: Settings | None = None) -> FastAPI:
    """Build the demo application from environment settings."""
    cfg = settings or Settings()
    configure_logging(cfg)
    llm = select_llm(cfg)
    get_logger(__name__).info("server_starting", llm=llm.name, model=llm.model)
    corpus = QADataset.synthetic().corpus
    return create_app(build_pipeline(corpus, llm, cfg))


app = create_default_app()
