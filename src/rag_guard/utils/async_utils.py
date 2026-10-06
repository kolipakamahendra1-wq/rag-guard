"""Async helpers: exponential backoff retry and timeouts."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import TypeVar

from rag_guard.exceptions import OperationTimeoutError, RetryExhaustedError
from rag_guard.logging import get_logger

T = TypeVar("T")
_log = get_logger(__name__)

Sleeper = Callable[[float], Awaitable[None]]


def backoff_delay_ms(attempt: int, base_ms: float, max_ms: float) -> float:
    """Exponential backoff delay for a zero-based attempt index, capped at max_ms."""
    return float(min(max_ms, base_ms * (2**attempt)))


async def retry_async(
    func: Callable[[], Awaitable[T]],
    *,
    attempts: int = 3,
    base_ms: float = 100.0,
    max_ms: float = 5000.0,
    retry_on: tuple[type[BaseException], ...] = (Exception,),
    sleep: Sleeper = asyncio.sleep,
) -> T:
    """Run ``func`` with exponential backoff; raise RetryExhaustedError when exhausted."""
    if attempts < 1:
        raise ValueError("attempts must be >= 1")
    last: BaseException | None = None
    for attempt in range(attempts):
        try:
            return await func()
        except retry_on as exc:
            last = exc
            _log.warning("retry_attempt_failed", attempt=attempt + 1, of=attempts, error=str(exc))
            if attempt < attempts - 1:
                await sleep(backoff_delay_ms(attempt, base_ms, max_ms) / 1000.0)
    raise RetryExhaustedError(f"exhausted {attempts} attempts: {last}", attempts=attempts) from last


async def with_timeout(coro: Awaitable[T], timeout_ms: float) -> T:
    """Await ``coro`` and raise OperationTimeoutError if it exceeds ``timeout_ms``."""
    try:
        return await asyncio.wait_for(coro, timeout=timeout_ms / 1000.0)
    except TimeoutError as exc:
        raise OperationTimeoutError(f"timed out after {timeout_ms}ms", timeout_ms) from exc
