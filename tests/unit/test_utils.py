import asyncio

import pytest

from rag_guard.exceptions import OperationTimeoutError, RetryExhaustedError
from rag_guard.utils.async_utils import backoff_delay_ms, retry_async, with_timeout
from rag_guard.utils.text_utils import (
    BM25Index,
    cosine,
    hashed_embedding,
    percentile,
    split_sentences,
    token_overlap,
    tokenize,
)


def test_tokenize_drops_stopwords_and_lowercases() -> None:
    assert tokenize("What is the Retention Period?") == ["retention", "period"]


def test_tokenize_keeps_stopwords_when_asked() -> None:
    assert "the" in tokenize("the cat", drop_stopwords=False)


def test_split_sentences() -> None:
    assert split_sentences("One. Two! Three?") == ["One.", "Two!", "Three?"]
    assert split_sentences("   ") == []


def test_token_overlap() -> None:
    assert token_overlap(["a", "b"], ["a"]) == 0.5
    assert token_overlap([], ["a"]) == 0.0


def test_hashed_embedding_is_deterministic_and_unit_norm() -> None:
    a = hashed_embedding("retention period orion")
    assert a == hashed_embedding("retention period orion")
    assert sum(v * v for v in a) == pytest.approx(1.0)
    assert hashed_embedding("the of") == [0.0] * 256


def test_cosine_identity_and_orthogonal_and_mismatch() -> None:
    assert cosine([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)
    assert cosine([1.0, 0.0], [0.0, 1.0]) == 0.0
    assert cosine([0.0, 0.0], [1.0, 0.0]) == 0.0
    with pytest.raises(ValueError):
        cosine([1.0], [1.0, 2.0])


def test_percentile() -> None:
    assert percentile([], 50) == 0.0
    assert percentile([1.0, 2.0, 3.0, 4.0], 50) == pytest.approx(2.5)
    assert percentile([5.0], 99) == 5.0


def test_bm25_prefers_matching_document(corpus: dict[str, str]) -> None:
    index = BM25Index(corpus)
    q = "retention period of Orion"
    assert index.score(q, "doc-orion") > index.score(q, "doc-vega")
    assert index.score(q, "doc-noise") == 0.0


def test_backoff_is_exponential_and_capped() -> None:
    assert [backoff_delay_ms(i, 100, 500) for i in range(4)] == [100, 200, 400, 500]


async def test_retry_succeeds_after_transient_failures() -> None:
    calls = 0
    sleeps: list[float] = []

    async def flaky() -> str:
        nonlocal calls
        calls += 1
        if calls < 3:
            raise ValueError("boom")
        return "ok"

    async def fake_sleep(s: float) -> None:
        sleeps.append(s)

    assert await retry_async(flaky, attempts=3, base_ms=100, sleep=fake_sleep) == "ok"
    assert sleeps == [0.1, 0.2]


async def test_retry_exhausted_raises_typed_error() -> None:
    async def always() -> str:
        raise ValueError("nope")

    async def no_sleep(_: float) -> None:
        return None

    with pytest.raises(RetryExhaustedError) as info:
        await retry_async(always, attempts=2, sleep=no_sleep)
    assert info.value.attempts == 2


async def test_retry_does_not_catch_unlisted_exceptions() -> None:
    async def bad() -> str:
        raise KeyError("x")

    with pytest.raises(KeyError):
        await retry_async(bad, attempts=3, retry_on=(ValueError,))


async def test_retry_rejects_zero_attempts() -> None:
    async def f() -> int:
        return 1

    with pytest.raises(ValueError):
        await retry_async(f, attempts=0)


async def test_with_timeout() -> None:
    async def slow() -> None:
        await asyncio.sleep(1)

    with pytest.raises(OperationTimeoutError):
        await with_timeout(slow(), 10)

    async def fast() -> int:
        return 7

    assert await with_timeout(fast(), 1000) == 7
