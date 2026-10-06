import pytest

from rag_guard.detection import (
    FailureDetector,
    RetrievalScorer,
    faithfulness_score,
    hallucination_rate,
)
from rag_guard.types import RecoveryStrategy, RetrievalResult, SeverityLevel
from tests.conftest import QUERY


def test_scorer_empty_is_zero() -> None:
    assert RetrievalScorer().score(QUERY, []).relevance == 0.0


def test_scorer_relevant_beats_irrelevant(
    good_chunk: RetrievalResult, noise_chunk: RetrievalResult
) -> None:
    scorer = RetrievalScorer()
    good = scorer.score(QUERY, [good_chunk])
    bad = scorer.score(QUERY, [noise_chunk])
    assert good.relevance > 0.9
    assert bad.relevance < 0.3
    assert good.signals["coverage"] == 1.0
    assert bad.signals["coverage"] == 0.0


def test_scorer_rejects_bad_weights() -> None:
    with pytest.raises(ValueError):
        RetrievalScorer(0, 0, 0)


def test_detector_healthy(good_chunk: RetrievalResult) -> None:
    result = FailureDetector().detect(QUERY, [good_chunk])
    assert not result.degraded
    assert result.severity is SeverityLevel.HEALTHY
    assert result.recommended_strategy is RecoveryStrategy.NONE


def test_detector_critical_on_empty_and_irrelevant(noise_chunk: RetrievalResult) -> None:
    detector = FailureDetector()
    assert detector.detect(QUERY, []).severity is SeverityLevel.CRITICAL
    result = detector.detect(QUERY, [noise_chunk])
    assert result.severity is SeverityLevel.CRITICAL
    assert result.recommended_strategy is RecoveryStrategy.ALTERNATIVE_STRATEGY


def test_detector_degraded_partial_coverage_recommends_broader_search() -> None:
    chunk = RetrievalResult(chunk_id="c", content="retention policy overview", score=0.6)
    result = FailureDetector(threshold=0.6, critical_threshold=0.2).detect(QUERY, [chunk])
    assert result.severity is SeverityLevel.DEGRADED
    assert result.recommended_strategy is RecoveryStrategy.BROADER_SEARCH


def test_detector_drift_flags_drop_below_baseline(good_chunk: RetrievalResult) -> None:
    detector = FailureDetector(threshold=0.6, drift_threshold=0.15)
    for _ in range(3):
        detector.detect(QUERY, [good_chunk])
    assert detector.baseline() is not None
    partial = RetrievalResult(chunk_id="p", content="retention period policy", score=0.9)
    result = detector.detect(QUERY, [partial])
    assert result.details["drifted"] is True
    assert result.degraded


def test_detector_does_not_learn_from_degraded(noise_chunk: RetrievalResult) -> None:
    detector = FailureDetector()
    detector.detect(QUERY, [noise_chunk])
    assert detector.baseline() is None


def test_detector_rejects_invalid_thresholds() -> None:
    with pytest.raises(ValueError):
        FailureDetector(threshold=0.3, critical_threshold=0.5)


def test_faithfulness_grounded_vs_ungrounded() -> None:
    context = "Orion retention period is 90 days."
    assert faithfulness_score("Orion retention period is 90 days.", context) == 1.0
    assert faithfulness_score("The moon is made of green cheese entirely.", context) == 0.0
    assert hallucination_rate("The moon is made of green cheese entirely.", context) == 1.0
    assert faithfulness_score("", context) == 0.0


def test_faithfulness_mixed_answer() -> None:
    context = "Orion retention period is 90 days."
    answer = "Orion retention period is 90 days. Zebras migrate across continents annually."
    assert faithfulness_score(answer, context) == 0.5
