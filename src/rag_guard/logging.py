"""Structured logging setup using structlog on top of stdlib logging."""

from __future__ import annotations

import logging
import sys
from typing import Any

import structlog

from rag_guard.config import Settings

_HANDLER_MARK = "_rag_guard_handler"


def configure_logging(settings: Settings) -> None:
    """Configure structlog (JSON or console renderer) and the stdlib root logger. Idempotent."""
    renderer: Any = (
        structlog.processors.JSONRenderer()
        if settings.enable_structured_logging
        else structlog.dev.ConsoleRenderer(colors=False)
    )
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            renderer,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=False,
    )

    root = logging.getLogger()
    for existing in [h for h in root.handlers if getattr(h, _HANDLER_MARK, False)]:
        root.removeHandler(existing)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(message)s"))
    setattr(handler, _HANDLER_MARK, True)
    root.addHandler(handler)
    root.setLevel(getattr(logging, settings.log_level.upper()))
    logging.getLogger("httpx").setLevel(logging.WARNING)


def get_logger(name: str) -> Any:
    """Get a structured logger by name."""
    return structlog.get_logger(name)


def bind_context(**kwargs: Any) -> None:
    """Attach key/values to every subsequent log line in this async context."""
    structlog.contextvars.bind_contextvars(**kwargs)


def unbind_context(*keys: str) -> None:
    """Remove previously bound keys (missing keys are ignored)."""
    structlog.contextvars.unbind_contextvars(*keys)
