"""Tests for CostPredictor — pre-execution cost prediction engine."""
from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from agent_energy_budget.prediction.output_estimator import OutputEstimator, TaskType
from agent_energy_budget.prediction.predictor import (
    BatchPredictionResult,
    CostPredictor,
    PredictionResult,
)
from agent_energy_budget.prediction.pricing_table import ModelPricing, PricingTable
from agent_energy_budget.prediction.token_counter import TokenCounter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_predictor(
    buffer_size: int = 1_000,
) -> CostPredictor:
    """Create a CostPredictor that uses heuristic token counting."""
    counter = TokenCounter(model="gpt-4o", prefer_tiktoken=False)
    return CostPredictor(token_counter=counter)


# ---------------------------------------------------------------------------
# PredictionResult dataclass
# ---------------------------------------------------------------------------


class TestPredictionResult:
    def test_valid_result(self) -> None:
        result = PredictionResult(
            estimated_cost_usd=0.001,
            input_tokens=500,
            estimated_output_tokens=200,
            model="gpt-4o",
            confidence=0.7,
        )
        assert result.model == "gpt-4o"
        assert result.will_exceed_budget is None

    def test_invalid_confidence_raises(self) -> None:
        with pytest.raises(ValueError, match="confidence must be in"):
            PredictionResult(
                estimated_cost_usd=0.001,
                input_tokens=500,
                estimated_output_tokens=200,
                model="gpt-4o",
                confidence=1.5,
            )


# ---------------------------------------------------------------------------
# predict() — core
# ---------------------------------------------------------------------------


class TestPredict:
    def test_returns_prediction_result(self) -> None:
        predictor = make_predictor()
        result = predictor.predict(model="gpt-4o", prompt="Hello world")
        assert isinstance(result, PredictionResult)

    def test_model_set_correctly(self) -> None:
        predictor = make_predictor()
        result = predictor.predict(model="gpt-4o", prompt="Hello")
        assert result.model == "gpt-4o"

    def test_input_tokens_positive(self) -> None:
        predictor = make_predictor()
        result = predictor.predict(model="gpt-4o", prompt="Hello, world!")
        assert result.input_tokens >= 1

    def test_estimated_output_tokens_positive(self) -> None:
        predictor = make_predictor()
        result = predictor.predict(model="gpt-4o", prompt="Hello")
        assert result.estimated_output_tokens >= 1

    def test_estimated_cost_positive(self) -> None:
        predictor = make_predictor()
        result = predictor.predict(model="claude-sonnet-4", prompt="A" * 1000)
        assert result.estimated_cost_usd >= 0.0

    def test_task_type_affects_output_estimate(self) -> None:
        predictor = make_predictor()
        chat_result = predictor.predict(model="gpt-4o", prompt="Hello", task_type="chat")
        code_result = predictor.predict(model="gpt-4o", prompt="Hello", task_type="code_gen")
        # Code gen should estimate more output tokens than chat
        assert code_result.estimated_output_tokens > chat_result.estimated_output_tokens

    def test_system_prompt_increases_input_tokens(self) -> None:
        predictor = make_predictor()
        without_system = predictor.predict(model="gpt-4o", prompt="Hello")
        with_system = predictor.predict(
            model="gpt-4o",
            prompt="Hello",
            system="You are a helpful AI assistant for a Fortune 500 company.",
        )
        assert with_system.input_tokens > without_system.input_tokens

    def test_will_exceed_budget_true(self) -> None:
        predictor = make_predictor()
        result = predictor.predict(
            model="claude-opus-4",
            prompt="A" * 10_000,  # large prompt
            task_type="code_gen",
            budget_usd=0.000001,  # tiny budget
        )
        assert result.will_exceed_budget is True

    def test_will_exceed_budget_false(self) -> None:
        predictor = make_predictor()
        result = predictor.predict(
            model="gemini-2.0-flash",
            prompt="Hi",
            task_type="chat",
            budget_usd=1.0,  # generous budget
        )
        assert result.will_exceed_budget is False

    def test_will_exceed_budget_none_when_no_budget(self) -> None:
        predictor = make_predictor()
        result = predictor.predict(model="gpt-4o", prompt="Hello")
        assert result.will_exceed_budget is None

    def test_messages_list_prompt(self) -> None:
        predictor = make_predictor()
        messages = [
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "4"},
            {"role": "user", "content": "And 3+3?"},
        ]
        result = predictor.predict(model="gpt-4o", prompt=messages)
        assert result.input_tokens >= 4

    def test_max_tokens_caps_output_estimate(self) -> None:
        predictor = make_predictor()
        result = predictor.predict(
            model="gpt-4o",
            prompt="Write a very long essay",
            task_type="analysis",
            max_tokens=50,
        )
        assert result.estimated_output_tokens <= 50

    def test_unknown_model_raises_key_error(self) -> None:
        predictor = make_predictor()
        with pytest.raises(KeyError):
            predictor.predict(model="nonexistent-model-xyz", prompt="Hello")

    def test_confidence_in_valid_range(self) -> None:
        predictor = make_predictor()
        result = predictor.predict(model="gpt-4o", prompt="Hello")
        assert 0.0 <= result.confidence <= 1.0

    def test_output_estimate_attached(self) -> None:
        predictor = make_predictor()
        result = predictor.predict(model="gpt-4o", prompt="Hello")
        assert result.output_estimate is not None

    def test_low_cost_less_than_high_cost(self) -> None:
        predictor = make_predictor()
        result = predictor.predict(
            model="claude-sonnet-4",
            prompt="Write a story",
            task_type="analysis",
        )
        assert result.low_cost_usd <= result.estimated_cost_usd
        assert result.estimated_cost_usd <= result.high_cost_usd

    def test_cached_tokens_reduces_cost(self) -> None:
        predictor = make_predictor()
        result_no_cache = predictor.predict(
            model="claude-sonnet-4",
            prompt="A" * 2000,
            cached_tokens=0,
        )
        result_with_cache = predictor.predict(
            model="claude-sonnet-4",
            prompt="A" * 2000,
            cached_tokens=500,
        )
        assert result_with_cache.estimated_cost_usd <= result_no_cache.estimated_cost_usd


# ---------------------------------------------------------------------------
# predict_with_tokens() — known token counts
# ---------------------------------------------------------------------------


class TestPredictWithTokens:
    def test_basic_prediction(self) -> None:
        predictor = make_predictor()
        result = predictor.predict_with_tokens(
            model="gpt-4o",
            input_tokens=1000,
            output_tokens=200,
        )
        assert result.input_tokens == 1000
        assert result.estimated_output_tokens == 200
        assert result.confidence == 1.0

    def test_budget_check(self) -> None:
        predictor = make_predictor()
        result = predictor.predict_with_tokens(
            model="gpt-4o",
            input_tokens=1000,
            output_tokens=200,
            budget_usd=0.0,  # impossible budget
        )
        assert result.will_exceed_budget is True


# ---------------------------------------------------------------------------
# compare_models()
# ---------------------------------------------------------------------------


class TestCompareModels:
    def test_returns_sorted_cheapest_first(self) -> None:
        predictor = make_predictor()
        models = ["claude-opus-4", "gpt-4o-mini", "gemini-2.0-flash"]
        results = predictor.compare_models(models=models, prompt="Hello")
        costs = [r.estimated_cost_usd for r in results]
        assert costs == sorted(costs)

    def test_unknown_models_skipped(self) -> None:
        predictor = make_predictor()
        models = ["gpt-4o", "nonexistent-xyz", "gpt-4o-mini"]
        results = predictor.compare_models(models=models, prompt="Hello")
        returned_models = [r.model for r in results]
        assert "nonexistent-xyz" not in returned_models
        assert len(results) == 2

    def test_budget_applied_to_all(self) -> None:
        predictor = make_predictor()
        results = predictor.compare_models(
            models=["gpt-4o", "gpt-4o-mini"],
            prompt="Hello",
            budget_usd=1.0,
        )
        for result in results:
            assert result.will_exceed_budget is False


# ---------------------------------------------------------------------------
# cheapest_model_within_budget()
# ---------------------------------------------------------------------------


class TestCheapestModelWithinBudget:
    def test_returns_first_affordable(self) -> None:
        predictor = make_predictor()
        # All models should fit within $1
        result = predictor.cheapest_model_within_budget(
            models=["claude-sonnet-4", "gpt-4o-mini"],
            prompt="Hello",
            budget_usd=1.0,
        )
        assert result is not None
        assert result.will_exceed_budget is False

    def test_returns_none_when_nothing_fits(self) -> None:
        predictor = make_predictor()
        result = predictor.cheapest_model_within_budget(
            models=["claude-opus-4", "gpt-4o"],
            prompt="A" * 10_000,
            budget_usd=0.000001,  # impossibly small
        )
        assert result is None

    def test_preference_order_respected(self) -> None:
        predictor = make_predictor()
        # First model that fits wins (most preferred)
        result = predictor.cheapest_model_within_budget(
            models=["gpt-4o-mini", "gemini-2.0-flash"],
            prompt="Hello",
            budget_usd=1.0,
        )
        assert result is not None
        assert result.model == "gpt-4o-mini"


# ---------------------------------------------------------------------------
# predict_batch()
# ---------------------------------------------------------------------------


class TestPredictBatch:
    def test_batch_aggregates_costs(self) -> None:
        predictor = make_predictor()
        calls = [
            {"model": "gpt-4o", "prompt": "Hello", "task_type": "chat"},
            {"model": "gpt-4o-mini", "prompt": "Summarize this", "task_type": "summary"},
        ]
        result = predictor.predict_batch(calls)
        assert isinstance(result, BatchPredictionResult)
        assert len(result.predictions) == 2
        assert result.total_estimated_cost_usd >= 0.0

    def test_batch_total_equals_sum(self) -> None:
        predictor = make_predictor()
        calls = [
            {"model": "gpt-4o", "prompt": "A" * 100, "task_type": "chat"},
            {"model": "gpt-4o-mini", "prompt": "B" * 100, "task_type": "code_gen"},
        ]
        result = predictor.predict_batch(calls)
        expected = sum(r.estimated_cost_usd for r in result.predictions)
        assert result.total_estimated_cost_usd == pytest.approx(expected, rel=1e-6)

    def test_batch_budget_check(self) -> None:
        predictor = make_predictor()
        calls = [{"model": "gpt-4o", "prompt": "Hello", "task_type": "chat"}]
        result = predictor.predict_batch(calls, budget_usd=0.0)
        assert result.any_will_exceed_budget is True

    def test_batch_invalid_model_skipped(self) -> None:
        predictor = make_predictor()
        calls = [
            {"model": "gpt-4o", "prompt": "Hello", "task_type": "chat"},
            {"model": "nonexistent-xyz", "prompt": "Hello", "task_type": "chat"},
        ]
        result = predictor.predict_batch(calls)
        assert len(result.predictions) == 1

    def test_batch_no_budget_has_none(self) -> None:
        predictor = make_predictor()
        calls = [{"model": "gpt-4o", "prompt": "Hello", "task_type": "chat"}]
        result = predictor.predict_batch(calls)
        assert result.any_will_exceed_budget is None


# ---------------------------------------------------------------------------
# Repr
# ---------------------------------------------------------------------------


class TestRepr:
    def test_repr_is_string(self) -> None:
        predictor = make_predictor()
        assert isinstance(repr(predictor), str)
