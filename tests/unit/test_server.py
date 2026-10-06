from fastapi.testclient import TestClient

from mocks.mock_llms import ExtractiveMockLLM
from rag_guard.config import Settings
from rag_guard.server import create_default_app, select_llm


def test_select_llm_falls_back_to_mock_without_credentials() -> None:
    for provider in ("anthropic", "openai", "azure", "local"):
        llm = select_llm(Settings(_env_file=None, default_llm_provider=provider))
        assert isinstance(llm, ExtractiveMockLLM)


def test_select_llm_honours_mock_flag_and_real_credentials() -> None:
    assert isinstance(
        select_llm(Settings(_env_file=None, use_mock_providers=True, anthropic_api_key="k")),
        ExtractiveMockLLM,
    )
    assert select_llm(Settings(_env_file=None, anthropic_api_key="k")).name == "anthropic"
    local = Settings(
        _env_file=None, default_llm_provider="local", local_llm_endpoint="http://localhost:1/v1"
    )
    assert select_llm(local).name == "local"


def test_default_app_serves_demo_corpus() -> None:
    client = TestClient(create_default_app(Settings(_env_file=None)))
    body = client.post(
        "/query", json={"query": "What is the replication factor of Torvex?"}
    ).json()
    assert body["success"]
    assert "31 units" in body["answer"]
