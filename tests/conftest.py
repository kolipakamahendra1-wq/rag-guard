"""Shared fixtures. Every test runs fully offline."""

from __future__ import annotations

from collections.abc import Callable

import httpx
import pytest

from mocks.mock_data import SAMPLE_CORPUS
from mocks.mock_llms import ExtractiveMockLLM
from rag_guard.config import Settings
from rag_guard.types import RetrievalResult

QUERY = "What is the retention period of Orion?"


@pytest.fixture
def corpus() -> dict[str, str]:
    return dict(SAMPLE_CORPUS)


@pytest.fixture
def settings() -> Settings:
    return Settings(_env_file=None)


@pytest.fixture
def llm() -> ExtractiveMockLLM:
    return ExtractiveMockLLM()


@pytest.fixture
def good_chunk(corpus: dict[str, str]) -> RetrievalResult:
    return RetrievalResult(chunk_id="doc-orion", content=corpus["doc-orion"], score=0.9)


@pytest.fixture
def noise_chunk(corpus: dict[str, str]) -> RetrievalResult:
    return RetrievalResult(chunk_id="doc-noise", content=corpus["doc-noise"], score=0.9)


def mock_client(handler: Callable[[httpx.Request], httpx.Response]) -> httpx.AsyncClient:
    """AsyncClient whose requests are answered by ``handler`` (no network)."""
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))
