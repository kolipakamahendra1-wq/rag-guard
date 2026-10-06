import httpx
import pytest

from rag_guard.config import Settings
from rag_guard.exceptions import ConfigError, LLMProviderError
from rag_guard.llm import (
    AnthropicProvider,
    AzureOpenAIProvider,
    LocalProvider,
    OpenAIProvider,
    create_provider,
)
from tests.conftest import mock_client

OPENAI_BODY = {
    "choices": [{"message": {"content": "90 days"}, "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 100, "completion_tokens": 10},
}
ANTHROPIC_BODY = {
    "content": [
        {"type": "text", "text": "90 days"},
        {"type": "tool_use", "id": "t1", "name": "lookup", "input": {}},
    ],
    "usage": {"input_tokens": 100, "output_tokens": 10},
    "stop_reason": "end_turn",
}


async def test_anthropic_parses_response_and_cost() -> None:
    seen: dict[str, httpx.Request] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["req"] = request
        return httpx.Response(200, json=ANTHROPIC_BODY)

    provider = AnthropicProvider(
        "claude-test",
        "key-1",
        client=mock_client(handler),
        input_cost_per_1k=0.003,
        output_cost_per_1k=0.015,
    )
    out = await provider.generate("q?", "ctx", trace_id="t-1")
    assert out.text == "90 days"
    assert out.tokens_used.total_tokens == 110
    assert out.tokens_used.cost_usd == pytest.approx(0.1 * 0.003 + 0.01 * 0.015)
    assert out.tool_calls[0]["name"] == "lookup"
    assert out.trace_id == "t-1"
    assert seen["req"].headers["x-api-key"] == "key-1"
    assert seen["req"].url.path == "/v1/messages"


async def test_openai_parses_response() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["authorization"] == "Bearer sk-1"
        return httpx.Response(200, json=OPENAI_BODY)

    out = await OpenAIProvider("gpt-test", "sk-1", client=mock_client(handler)).generate("q", "c")
    assert out.text == "90 days"
    assert out.tokens_used.provider == "openai"
    assert out.stop_reason == "stop"


async def test_azure_uses_deployment_url_and_api_key_header() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert "/openai/deployments/my-deploy/chat/completions" in str(request.url)
        assert request.headers["api-key"] == "az-1"
        return httpx.Response(200, json=OPENAI_BODY)

    provider = AzureOpenAIProvider(
        "my-deploy", "az-1", endpoint="https://x.openai.azure.com", client=mock_client(handler)
    )
    assert (await provider.generate("q", "c")).text == "90 days"


async def test_local_provider_sends_no_auth() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert "authorization" not in request.headers
        return httpx.Response(200, json=OPENAI_BODY)

    provider = LocalProvider(
        "llama", endpoint="http://localhost:11434/v1", client=mock_client(handler)
    )
    assert (await provider.generate("q", "c")).text == "90 days"


async def test_retries_on_rate_limit_then_succeeds() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(429)
        return httpx.Response(200, json=OPENAI_BODY)

    out = await OpenAIProvider("m", "k", client=mock_client(handler)).generate("q", "c")
    assert out.text == "90 days"
    assert calls == 2


async def test_client_error_is_not_retried() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(400, text="bad request")

    with pytest.raises(LLMProviderError):
        await OpenAIProvider("m", "k", client=mock_client(handler)).generate("q", "c")
    assert calls == 1


async def test_persistent_server_error_exhausts_retries() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503)

    with pytest.raises(Exception) as info:
        await OpenAIProvider("m", "k", client=mock_client(handler), retries=2).generate("q", "c")
    assert "exhausted 2 attempts" in str(info.value)


async def test_network_error_is_retryable() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise httpx.ConnectError("down")
        return httpx.Response(200, json=ANTHROPIC_BODY)

    out = await AnthropicProvider("m", "k", client=mock_client(handler)).generate("q", "c")
    assert out.text == "90 days"


async def test_malformed_body_raises_provider_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"unexpected": True})

    with pytest.raises(LLMProviderError):
        await OpenAIProvider("m", "k", client=mock_client(handler)).generate("q", "c")
    with pytest.raises(LLMProviderError):
        await AnthropicProvider("m", "k", client=mock_client(handler)).generate("q", "c")


async def test_stream_generate_yields_ordered_tokens() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=OPENAI_BODY)

    provider = OpenAIProvider("m", "k", client=mock_client(handler))
    chunks = [c async for c in provider.stream_generate("q", "c")]
    assert [c.token for c in chunks] == ["90", "days"]
    assert [c.index for c in chunks] == [0, 1]


def test_factory_builds_each_provider() -> None:
    s = Settings(
        _env_file=None,
        azure_endpoint="https://x.openai.azure.com",
        local_llm_endpoint="http://localhost:11434/v1",
    )
    assert create_provider("anthropic", s).name == "anthropic"
    assert create_provider("openai", s, model="gpt-x").model == "gpt-x"
    assert create_provider("azure", s).name == "azure"
    assert create_provider("local", s).name == "local"


def test_factory_rejects_unknown_and_unconfigured(settings: Settings) -> None:
    with pytest.raises(ConfigError):
        create_provider("nope", settings)
    with pytest.raises(ConfigError):
        create_provider("azure", settings)
    with pytest.raises(ConfigError):
        create_provider("local", settings)
