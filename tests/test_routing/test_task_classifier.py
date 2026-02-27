"""Tests for TaskClassifier and ClassificationResult (E10.2)."""
from __future__ import annotations

import pytest

from agent_energy_budget.routing.task_classifier import (
    ClassificationResult,
    TaskClassifier,
    TaskType,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def classifier() -> TaskClassifier:
    return TaskClassifier()


# ---------------------------------------------------------------------------
# ClassificationResult
# ---------------------------------------------------------------------------

class TestClassificationResult:
    def test_frozen_dataclass(self) -> None:
        result = ClassificationResult(
            task_type="simple",
            estimated_tokens=5,
            detected_keywords=(),
            is_factual_query=True,
            confidence="high",
        )
        with pytest.raises((AttributeError, TypeError)):
            result.task_type = "complex"  # type: ignore[misc]

    def test_fields_accessible(self) -> None:
        result = ClassificationResult(
            task_type="medium",
            estimated_tokens=200,
            detected_keywords=("explain",),
            is_factual_query=False,
            confidence="high",
        )
        assert result.task_type == "medium"
        assert result.estimated_tokens == 200
        assert "explain" in result.detected_keywords


# ---------------------------------------------------------------------------
# TaskClassifier — construction
# ---------------------------------------------------------------------------

class TestTaskClassifierConstruction:
    def test_default_thresholds(self) -> None:
        classifier = TaskClassifier()
        assert classifier._simple_threshold == 100
        assert classifier._complex_threshold == 500

    def test_custom_thresholds(self) -> None:
        classifier = TaskClassifier(
            simple_token_threshold=50, complex_token_threshold=200
        )
        assert classifier._simple_threshold == 50
        assert classifier._complex_threshold == 200

    def test_invalid_simple_threshold_raises(self) -> None:
        with pytest.raises(ValueError, match="simple_token_threshold"):
            TaskClassifier(simple_token_threshold=0, complex_token_threshold=500)

    def test_complex_not_greater_than_simple_raises(self) -> None:
        with pytest.raises(ValueError, match="complex_token_threshold"):
            TaskClassifier(
                simple_token_threshold=200, complex_token_threshold=100
            )

    def test_invalid_words_per_token_raises(self) -> None:
        with pytest.raises(ValueError, match="words_per_token"):
            TaskClassifier(words_per_token=0.0)


# ---------------------------------------------------------------------------
# TaskClassifier — estimate_tokens
# ---------------------------------------------------------------------------

class TestEstimateTokens:
    def test_empty_string_zero_tokens(self, classifier: TaskClassifier) -> None:
        assert classifier.estimate_tokens("") == 0

    def test_single_word(self, classifier: TaskClassifier) -> None:
        tokens = classifier.estimate_tokens("hello")
        assert tokens > 0

    def test_more_words_more_tokens(self, classifier: TaskClassifier) -> None:
        short = classifier.estimate_tokens("hello world")
        long_text = classifier.estimate_tokens("hello world " * 50)
        assert long_text > short

    def test_token_count_is_non_negative(self, classifier: TaskClassifier) -> None:
        assert classifier.estimate_tokens("   ") >= 0


# ---------------------------------------------------------------------------
# TaskClassifier — classify: simple
# ---------------------------------------------------------------------------

class TestClassifySimple:
    def test_short_factual_what_is(self, classifier: TaskClassifier) -> None:
        result = classifier.classify("What is the capital of France?")
        assert result.task_type == "simple"

    def test_short_factual_who_is(self, classifier: TaskClassifier) -> None:
        result = classifier.classify("Who is the President of the USA?")
        assert result.task_type == "simple"

    def test_very_short_prompt(self, classifier: TaskClassifier) -> None:
        result = classifier.classify("Hi there!")
        assert result.task_type == "simple"

    def test_simple_is_factual_true(self, classifier: TaskClassifier) -> None:
        result = classifier.classify("What is Python?")
        assert result.is_factual_query is True

    def test_simple_high_confidence(self, classifier: TaskClassifier) -> None:
        result = classifier.classify("What is 2 + 2?")
        assert result.confidence == "high"

    def test_simple_no_detected_keywords(self, classifier: TaskClassifier) -> None:
        result = classifier.classify("What time is it?")
        assert len(result.detected_keywords) == 0


# ---------------------------------------------------------------------------
# TaskClassifier — classify: medium
# ---------------------------------------------------------------------------

class TestClassifyMedium:
    def test_explain_keyword(self, classifier: TaskClassifier) -> None:
        result = classifier.classify("Explain how neural networks work in simple terms.")
        assert result.task_type == "medium"

    def test_describe_keyword(self, classifier: TaskClassifier) -> None:
        result = classifier.classify(
            "Describe the main features of the Python programming language."
        )
        assert result.task_type == "medium"

    def test_summarise_keyword(self, classifier: TaskClassifier) -> None:
        result = classifier.classify("Summarize the following article for me.")
        assert result.task_type == "medium"

    def test_medium_keyword_detected(self, classifier: TaskClassifier) -> None:
        result = classifier.classify("Explain the water cycle.")
        assert len(result.detected_keywords) > 0

    def test_medium_is_not_simple(self, classifier: TaskClassifier) -> None:
        result = classifier.classify(
            "Summarize the pros and cons of electric vehicles for commuters."
        )
        assert result.task_type != "simple"


# ---------------------------------------------------------------------------
# TaskClassifier — classify: complex
# ---------------------------------------------------------------------------

class TestClassifyComplex:
    def test_analyse_keyword(self, classifier: TaskClassifier) -> None:
        result = classifier.classify(
            "Analyse the economic impacts of climate change on coastal cities."
        )
        assert result.task_type == "complex"

    def test_comprehensive_keyword(self, classifier: TaskClassifier) -> None:
        result = classifier.classify(
            "Provide a comprehensive overview of quantum computing."
        )
        assert result.task_type == "complex"

    def test_step_by_step_keyword(self, classifier: TaskClassifier) -> None:
        result = classifier.classify(
            "Walk me step by step through setting up a Kubernetes cluster."
        )
        assert result.task_type == "complex"

    def test_long_prompt_is_complex(self, classifier: TaskClassifier) -> None:
        # Generate ~700+ words
        long_prompt = ("This is a test sentence for a very long prompt. " * 100)
        result = classifier.classify(long_prompt)
        assert result.task_type == "complex"

    def test_complex_high_confidence_long(self, classifier: TaskClassifier) -> None:
        long_prompt = "word " * 600
        result = classifier.classify(long_prompt)
        assert result.task_type == "complex"
        assert result.confidence == "high"

    def test_complex_keyword_detected(self, classifier: TaskClassifier) -> None:
        result = classifier.classify(
            "Evaluate the implications of this policy for small businesses."
        )
        assert len(result.detected_keywords) > 0
        assert result.task_type == "complex"

    def test_in_depth_keyword(self, classifier: TaskClassifier) -> None:
        result = classifier.classify(
            "Give me an in-depth analysis of modern cryptographic algorithms."
        )
        assert result.task_type == "complex"

    def test_research_keyword(self, classifier: TaskClassifier) -> None:
        result = classifier.classify(
            "Research the history and current state of renewable energy adoption."
        )
        assert result.task_type == "complex"


# ---------------------------------------------------------------------------
# TaskClassifier — classify_many
# ---------------------------------------------------------------------------

class TestClassifyMany:
    def test_returns_same_count(self, classifier: TaskClassifier) -> None:
        prompts = [
            "What is 2+2?",
            "Explain neural networks.",
            "Analyse the impact of AI on society.",
        ]
        results = classifier.classify_many(prompts)
        assert len(results) == 3

    def test_results_in_order(self, classifier: TaskClassifier) -> None:
        prompts = [
            "What is 2+2?",
            "Analyse the economic implications of AI.",
        ]
        results = classifier.classify_many(prompts)
        assert results[0].task_type == "simple"
        assert results[1].task_type == "complex"

    def test_empty_list_returns_empty(self, classifier: TaskClassifier) -> None:
        results = classifier.classify_many([])
        assert results == []


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestClassifierEdgeCases:
    def test_empty_prompt_is_simple(self, classifier: TaskClassifier) -> None:
        result = classifier.classify("")
        assert result.task_type == "simple"
        assert result.estimated_tokens == 0

    def test_whitespace_only_is_simple(self, classifier: TaskClassifier) -> None:
        result = classifier.classify("   \n\t  ")
        assert result.task_type == "simple"

    def test_result_task_type_is_valid_literal(self, classifier: TaskClassifier) -> None:
        for prompt in ["hello", "explain something", "analyse everything"]:
            result = classifier.classify(prompt)
            assert result.task_type in ("simple", "medium", "complex")

    def test_detected_keywords_is_tuple(self, classifier: TaskClassifier) -> None:
        result = classifier.classify("Explain quantum physics.")
        assert isinstance(result.detected_keywords, tuple)

    def test_estimated_tokens_non_negative(self, classifier: TaskClassifier) -> None:
        result = classifier.classify("Hello world.")
        assert result.estimated_tokens >= 0
