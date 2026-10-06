import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from mocks.mock_llms import ExtractiveMockLLM
from rag_guard.api import create_app
from rag_guard.config import Settings
from rag_guard.exceptions import ConfigError, RAGGuardException, RetryExhaustedError
from rag_guard.orchestration.builder import build_pipeline
from rag_guard.retrieval import BM25Retriever
from rag_guard.telemetry import Telemetry
from tests.conftest import QUERY


def test_settings_defaults_and_helpers() -> None:
    s = Settings(_env_file=None)
    assert s.is_development() and not s.is_production()
    assert s.get_provider_key("openai") is None
    assert Settings(_env_file=None, openai_api_key="k").get_provider_key("openai") == "k"


def test_settings_validation() -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, environment="moon")
    with pytest.raises(ValidationError):
        Settings(_env_file=None, default_llm_provider="nope")
    with pytest.raises(ValidationError):
        Settings(_env_file=None, backoff_base_ms=500.0, backoff_max_ms=100.0)


def test_settings_reads_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEGRADATION_THRESHOLD", "0.8")
    assert Settings(_env_file=None).degradation_threshold == 0.8


def test_exception_hierarchy() -> None:
    err = RetryExhaustedError("x", attempts=3)
    assert isinstance(err, RAGGuardException)
    assert err.code == "RETRY_EXHAUSTED_ERROR" and err.attempts == 3
    assert ConfigError("bad").code == "CONFIG_ERROR"


def test_telemetry_spans_and_counters() -> None:
    t = Telemetry()
    with t.span("a"):
        pass
    t.incr("hits")
    t.incr("hits", 2)
    assert t.spans[0].name == "a" and t.spans[0].trace_id == t.trace_id
    assert t.counters["hits"] == 3
    assert t.total_ms() >= 0.0


def test_api_query_and_health(corpus: dict[str, str], settings: Settings) -> None:
    pipeline = build_pipeline(corpus, ExtractiveMockLLM(), settings, primary=BM25Retriever(corpus))
    client = TestClient(create_app(pipeline))
    assert client.get("/health").json() == {"status": "ok"}
    body = client.post("/query", json={"query": QUERY}).json()
    assert body["success"] and "90 days" in body["answer"]
    assert body["sources"][0] == "doc-orion"
    assert body["degraded"] is False
    assert client.post("/query", json={"query": ""}).status_code == 422
