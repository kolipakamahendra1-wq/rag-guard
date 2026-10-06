import json

import pytest

from rag_guard.config import Settings
from rag_guard.logging import bind_context, configure_logging, get_logger, unbind_context


@pytest.mark.parametrize("structured", [True, False])
def test_configure_and_log(structured: bool, capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging(
        Settings(_env_file=None, enable_structured_logging=structured, log_level="INFO")
    )
    get_logger("test").info("hello", key="value")
    out = capsys.readouterr().out
    assert "hello" in out and "value" in out
    if structured:
        assert json.loads(out.strip().splitlines()[-1])["key"] == "value"


def test_configure_is_idempotent(capsys: pytest.CaptureFixture[str]) -> None:
    settings = Settings(_env_file=None, log_level="INFO")
    configure_logging(settings)
    configure_logging(settings)
    get_logger("test").info("once")
    assert capsys.readouterr().out.count("once") == 1


def test_bound_context_appears_and_unbinds(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging(Settings(_env_file=None, log_level="INFO"))
    bind_context(request_id="r1")
    get_logger("test").info("with-ctx")
    assert json.loads(capsys.readouterr().out.strip().splitlines()[-1])["request_id"] == "r1"
    unbind_context("request_id", "never-bound")
    get_logger("test").info("without-ctx")
    assert "request_id" not in json.loads(capsys.readouterr().out.strip().splitlines()[-1])
