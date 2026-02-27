"""Tests for OutputEstimator — output token heuristics by task type."""
from __future__ import annotations

import pytest

from agent_energy_budget.prediction.output_estimator import (
    OutputEstimate,
    OutputEstimator,
    TaskType,
    _TASK_CONFIGS,
)


# ---------------------------------------------------------------------------
# OutputEstimate dataclass
# ---------------------------------------------------------------------------


class TestOutputEstimate:
    def test_valid_confidence(self) -> None:
        estimate = OutputEstimate(
            estimated_tokens=200,
            confidence=0.7,
            method="test",
            task_type=TaskType.CHAT,
            low_estimate=80,
            high_estimate=500,
        )
        assert estimate.confidence == 0.7

    def test_invalid_confidence_raises(self) -> None:
        with pytest.raises(ValueError, match="confidence must be in"):
            OutputEstimate(
                estimated_tokens=200,
                confidence=1.5,  # invalid
                method="test",
                task_type=TaskType.CHAT,
                low_estimate=80,
                high_estimate=500,
            )

    def test_negative_confidence_raises(self) -> None:
        with pytest.raises(ValueError):
            OutputEstimate(
                estimated_tokens=200,
                confidence=-0.1,
                method="test",
                task_type=TaskType.CHAT,
                low_estimate=80,
                high_estimate=500,
            )


# ---------------------------------------------------------------------------
# OutputEstimator construction
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_default_construction(self) -> None:
        estimator = OutputEstimator()
        assert estimator._default_task_type == TaskType.UNKNOWN

    def test_custom_default_task_type(self) -> None:
        estimator = OutputEstimator(default_task_type=TaskType.CHAT)
        assert estimator._default_task_type == TaskType.CHAT

    def test_repr_contains_type(self) -> None:
        estimator = OutputEstimator()
        assert "unknown" in repr(estimator)


# ---------------------------------------------------------------------------
# estimate() — task type heuristics
# ---------------------------------------------------------------------------


class TestEstimate:
    def test_chat_task_returns_reasonable_estimate(self) -> None:
        estimator = OutputEstimator()
        result = estimator.estimate(task_type=TaskType.CHAT)
        assert result.estimated_tokens == 200
        assert result.confidence >= 0.5
        assert result.task_type == TaskType.CHAT

    def test_code_gen_task(self) -> None:
        estimator = OutputEstimator()
        result = estimator.estimate(task_type=TaskType.CODE_GEN)
        assert result.estimated_tokens == 500

    def test_qa_task(self) -> None:
        estimator = OutputEstimator()
        result = estimator.estimate(task_type=TaskType.QA)
        assert result.estimated_tokens == 150

    def test_analysis_task(self) -> None:
        estimator = OutputEstimator()
        result = estimator.estimate(task_type=TaskType.ANALYSIS)
        assert result.estimated_tokens == 400

    def test_extraction_task(self) -> None:
        estimator = OutputEstimator()
        result = estimator.estimate(task_type=TaskType.EXTRACTION)
        assert result.estimated_tokens == 250

    def test_unknown_task_returns_fallback(self) -> None:
        estimator = OutputEstimator()
        result = estimator.estimate(task_type=TaskType.UNKNOWN)
        assert result.estimated_tokens > 0
        assert result.confidence <= 0.4

    def test_string_task_type_accepted(self) -> None:
        estimator = OutputEstimator()
        result = estimator.estimate(task_type="chat")
        assert result.task_type == TaskType.CHAT

    def test_unknown_string_falls_back_to_default(self) -> None:
        estimator = OutputEstimator(default_task_type=TaskType.CHAT)
        result = estimator.estimate(task_type="not_a_real_task_type")
        assert result.task_type == TaskType.CHAT

    def test_all_task_types_produce_valid_estimates(self) -> None:
        estimator = OutputEstimator()
        for task_type in TaskType:
            result = estimator.estimate(task_type=task_type, input_tokens=1000)
            assert result.estimated_tokens >= 1
            assert 0.0 <= result.confidence <= 1.0
            assert result.low_estimate >= 1
            assert result.high_estimate >= result.low_estimate

    def test_summary_uses_input_fraction(self) -> None:
        estimator = OutputEstimator()
        result_small = estimator.estimate(task_type=TaskType.SUMMARY, input_tokens=400)
        result_large = estimator.estimate(task_type=TaskType.SUMMARY, input_tokens=4000)
        # Summary output should scale with input
        assert result_large.estimated_tokens > result_small.estimated_tokens

    def test_summary_is_quarter_of_input(self) -> None:
        estimator = OutputEstimator()
        result = estimator.estimate(task_type=TaskType.SUMMARY, input_tokens=2000)
        assert result.estimated_tokens == pytest.approx(500, abs=50)

    def test_translation_is_same_length_as_input(self) -> None:
        estimator = OutputEstimator()
        result = estimator.estimate(task_type=TaskType.TRANSLATION, input_tokens=1000)
        assert result.estimated_tokens == pytest.approx(1000, abs=100)

    def test_max_tokens_caps_estimate(self) -> None:
        estimator = OutputEstimator()
        result = estimator.estimate(task_type=TaskType.CODE_GEN, max_tokens=100)
        assert result.estimated_tokens <= 100
        assert result.high_estimate <= 100

    def test_low_estimate_less_than_high(self) -> None:
        estimator = OutputEstimator()
        for task_type in TaskType:
            result = estimator.estimate(task_type=task_type)
            assert result.low_estimate <= result.high_estimate


# ---------------------------------------------------------------------------
# estimate_from_hint() — keyword-based classification
# ---------------------------------------------------------------------------


class TestEstimateFromHint:
    def test_code_keyword_detected(self) -> None:
        estimator = OutputEstimator()
        result = estimator.estimate_from_hint("Write a Python function to sort a list")
        assert result.task_type == TaskType.CODE_GEN

    def test_summary_keyword_detected(self) -> None:
        estimator = OutputEstimator()
        result = estimator.estimate_from_hint("Please summarize this document")
        assert result.task_type == TaskType.SUMMARY

    def test_translation_keyword_detected(self) -> None:
        estimator = OutputEstimator()
        result = estimator.estimate_from_hint("Translate this text to French")
        assert result.task_type == TaskType.TRANSLATION

    def test_extraction_keyword_detected(self) -> None:
        estimator = OutputEstimator()
        result = estimator.estimate_from_hint("Extract all entities from the text")
        assert result.task_type == TaskType.EXTRACTION

    def test_analysis_keyword_detected(self) -> None:
        estimator = OutputEstimator()
        result = estimator.estimate_from_hint("Analyze the sentiment of these reviews")
        assert result.task_type == TaskType.ANALYSIS

    def test_qa_keyword_detected(self) -> None:
        estimator = OutputEstimator()
        result = estimator.estimate_from_hint("What is the capital of France?")
        assert result.task_type == TaskType.QA

    def test_chat_keyword_detected(self) -> None:
        estimator = OutputEstimator()
        result = estimator.estimate_from_hint("Hi, how are you today?")
        assert result.task_type == TaskType.CHAT

    def test_unrecognized_hint_returns_unknown(self) -> None:
        estimator = OutputEstimator()
        result = estimator.estimate_from_hint("xyzzy plugh waldo frotz")
        assert result.task_type == TaskType.UNKNOWN

    def test_hint_with_input_tokens(self) -> None:
        estimator = OutputEstimator()
        result = estimator.estimate_from_hint("Summarize this", input_tokens=2000)
        assert result.task_type == TaskType.SUMMARY
        assert result.estimated_tokens == pytest.approx(500, abs=100)
