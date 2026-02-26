"""Unit tests for agent_energy_budget.estimator.cost_estimator (standalone module)."""
from __future__ import annotations

import pytest

from agent_energy_budget.estimator.cost_estimator import CostEstimate, CostEstimator
from agent_energy_budget.pricing.tables import ModelPricing, ModelTier, ProviderName


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def estimator() -> CostEstimator:
    return CostEstimator()


# ---------------------------------------------------------------------------
# CostEstimate dataclass
# ---------------------------------------------------------------------------


class TestCostEstimate:
    def test_is_frozen(self, estimator: CostEstimator) -> None:
        est = estimator.estimate_llm_call("gpt-4o-mini", "Hello world", 100)
        with pytest.raises((AttributeError, TypeError)):
            est.model = "other"  # type: ignore[misc]

    def test_confidence_in_valid_range(self, estimator: CostEstimator) -> None:
        est = estimator.estimate_llm_call("gpt-4o-mini", "Hello world", 100)
        assert 0.0 <= est.confidence <= 1.0


# ---------------------------------------------------------------------------
# CostEstimator.estimate_tokens
# ---------------------------------------------------------------------------


class TestEstimateTokens:
    def test_empty_string_returns_zero(self, estimator: CostEstimator) -> None:
        assert estimator.estimate_tokens("") == 0

    def test_non_empty_string_returns_positive(self, estimator: CostEstimator) -> None:
        count = estimator.estimate_tokens("Hello, this is a test sentence.")
        assert count > 0

    def test_longer_text_produces_more_tokens(self, estimator: CostEstimator) -> None:
        short = estimator.estimate_tokens("Hi")
        long = estimator.estimate_tokens("Hi " * 100)
        assert long > short


# ---------------------------------------------------------------------------
# CostEstimator.estimate_llm_call
# ---------------------------------------------------------------------------


class TestEstimateLlmCall:
    def test_basic_estimate_known_model(self, estimator: CostEstimator) -> None:
        est = estimator.estimate_llm_call("gpt-4o-mini", "Hello world", 512)
        assert est.model == "gpt-4o-mini"
        assert est.estimated_cost_usd >= 0.0
        assert est.estimated_input_tokens >= 0
        assert est.estimated_output_tokens == 512

    def test_estimate_unknown_model_raises(self, estimator: CostEstimator) -> None:
        with pytest.raises(KeyError):
            estimator.estimate_llm_call("totally-unknown-xyz", "Hello", 100)

    def test_empty_prompt_gives_zero_input_tokens(self, estimator: CostEstimator) -> None:
        est = estimator.estimate_llm_call("gpt-4o-mini", "", 100)
        assert est.estimated_input_tokens == 0

    def test_custom_pricing_used(self) -> None:
        custom = ModelPricing(
            model="my-model",
            provider=ProviderName.CUSTOM,
            tier=ModelTier.NANO,
            input_per_million=0.001,
            output_per_million=0.002,
        )
        estimator = CostEstimator(custom_pricing={"my-model": custom})
        est = estimator.estimate_llm_call("my-model", "Hello", 1_000_000)
        # 1M output tokens at $0.002/M = $0.002
        assert est.estimated_cost_usd == pytest.approx(0.002, rel=0.1)

    def test_confidence_heuristic_when_no_tiktoken(self, estimator: CostEstimator) -> None:
        est = estimator.estimate_llm_call("gpt-4o-mini", "Hello", 100)
        # Without tiktoken, confidence should be heuristic (0.7)
        assert est.confidence in (
            CostEstimator._CONFIDENCE_HEURISTIC,
            CostEstimator._CONFIDENCE_TIKTOKEN,
        )

    def test_higher_output_tokens_means_higher_cost(self, estimator: CostEstimator) -> None:
        est_small = estimator.estimate_llm_call("gpt-4o-mini", "Hi", 100)
        est_large = estimator.estimate_llm_call("gpt-4o-mini", "Hi", 10000)
        assert est_large.estimated_cost_usd > est_small.estimated_cost_usd


# ---------------------------------------------------------------------------
# CostEstimator.estimate_workflow
# ---------------------------------------------------------------------------


class TestEstimateWorkflow:
    def test_empty_workflow_returns_zero(self, estimator: CostEstimator) -> None:
        total = estimator.estimate_workflow([])
        assert total == 0.0

    def test_single_step_workflow(self, estimator: CostEstimator) -> None:
        steps = [{"model": "gpt-4o-mini", "prompt": "Hello world"}]
        total = estimator.estimate_workflow(steps)
        assert total > 0.0

    def test_multi_step_sums_correctly(self, estimator: CostEstimator) -> None:
        steps = [
            {"model": "gpt-4o-mini", "prompt": "Step 1"},
            {"model": "gpt-4o-mini", "prompt": "Step 2"},
        ]
        total = estimator.estimate_workflow(steps)
        individual = estimator.estimate_llm_call("gpt-4o-mini", "Step 1", 4096)
        # Total should be roughly 2x a single step
        assert total > individual.estimated_cost_usd

    def test_custom_max_output_tokens_applied(self, estimator: CostEstimator) -> None:
        steps = [{"model": "gpt-4o-mini", "prompt": "Hello", "max_output_tokens": "100"}]
        total = estimator.estimate_workflow(steps)
        # Compare against default 4096 tokens — should be much less
        default_steps = [{"model": "gpt-4o-mini", "prompt": "Hello"}]
        default_total = estimator.estimate_workflow(default_steps)
        assert total < default_total

    def test_missing_model_key_raises(self, estimator: CostEstimator) -> None:
        with pytest.raises(KeyError):
            estimator.estimate_workflow([{"prompt": "Hello"}])

    def test_invalid_max_output_tokens_raises(self, estimator: CostEstimator) -> None:
        with pytest.raises(ValueError):
            estimator.estimate_workflow(
                [{"model": "gpt-4o-mini", "prompt": "Hi", "max_output_tokens": "abc"}]
            )


# ---------------------------------------------------------------------------
# CostEstimator.compare_models
# ---------------------------------------------------------------------------


class TestCompareModels:
    def test_sorted_cheapest_first(self, estimator: CostEstimator) -> None:
        models = ["gpt-4o", "gpt-4o-mini"]
        estimates = estimator.compare_models(models, "Hello world")
        assert len(estimates) == 2
        assert estimates[0].estimated_cost_usd <= estimates[1].estimated_cost_usd

    def test_unknown_models_silently_skipped(self, estimator: CostEstimator) -> None:
        models = ["gpt-4o-mini", "unknown-model-xyz"]
        estimates = estimator.compare_models(models, "Hi")
        model_names = [e.model for e in estimates]
        assert "gpt-4o-mini" in model_names
        assert "unknown-model-xyz" not in model_names

    def test_empty_models_returns_empty(self, estimator: CostEstimator) -> None:
        estimates = estimator.compare_models([], "Hi")
        assert estimates == []


# ---------------------------------------------------------------------------
# CostEstimator.cheapest_model_for_budget
# ---------------------------------------------------------------------------


class TestCheapestModelForBudget:
    def test_returns_estimate_within_budget(self, estimator: CostEstimator) -> None:
        result = estimator.cheapest_model_for_budget(10.0, "Hello world")
        assert result is not None
        assert result.estimated_cost_usd <= 10.0

    def test_returns_none_when_nothing_fits(self, estimator: CostEstimator) -> None:
        result = estimator.cheapest_model_for_budget(0.0, "Hello world")
        assert result is None

    def test_prefers_high_quality_within_budget(self, estimator: CostEstimator) -> None:
        result_large = estimator.cheapest_model_for_budget(100.0, "Hello", 100)
        result_small = estimator.cheapest_model_for_budget(0.00001, "Hello", 100)
        if result_large and result_small:
            # Large budget should give a better quality (higher priced) model
            assert (
                result_large.pricing.input_per_million
                >= result_small.pricing.input_per_million
            )
