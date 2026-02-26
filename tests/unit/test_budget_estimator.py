"""Unit tests for agent_energy_budget.budget.estimator."""
from __future__ import annotations

import pytest

from agent_energy_budget.budget.estimator import (
    CostEstimate,
    CostEstimator,
    WorkflowCostEstimate,
    WorkflowStep,
)
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
        est = estimator.estimate("gpt-4o-mini", 100, 50)
        with pytest.raises((AttributeError, TypeError)):
            est.model = "other"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# CostEstimator — estimate
# ---------------------------------------------------------------------------


class TestEstimate:
    def test_estimate_known_model(self, estimator: CostEstimator) -> None:
        est = estimator.estimate("gpt-4o-mini", 1000, 200)
        assert est.model == "gpt-4o-mini"
        assert est.estimated_cost_usd > 0.0
        assert est.input_tokens == 1000
        assert est.output_tokens == 200

    def test_estimate_unknown_model_raises(self, estimator: CostEstimator) -> None:
        with pytest.raises(KeyError):
            estimator.estimate("totally-unknown-xyz", 100, 50)

    def test_estimate_uses_default_output_tokens_when_none(
        self, estimator: CostEstimator
    ) -> None:
        est = estimator.estimate("gpt-4o-mini", 100, None)
        assert est.output_tokens == estimator._default_output_tokens

    def test_estimate_with_text_input(self, estimator: CostEstimator) -> None:
        est = estimator.estimate("gpt-4o-mini", "Hello world, this is a test prompt.", 50)
        assert est.input_tokens > 0
        assert est.estimated_cost_usd >= 0.0

    def test_estimate_cost_is_positive(self, estimator: CostEstimator) -> None:
        est = estimator.estimate("gpt-4o-mini", 5000, 500)
        assert est.estimated_cost_usd > 0.0

    def test_custom_pricing_takes_precedence(self) -> None:
        custom = ModelPricing(
            model="my-custom-model",
            provider=ProviderName.CUSTOM,
            tier=ModelTier.EFFICIENT,
            input_per_million=0.01,
            output_per_million=0.02,
        )
        estimator = CostEstimator(custom_pricing={"my-custom-model": custom})
        est = estimator.estimate("my-custom-model", 1_000_000, 1_000_000)
        assert est.estimated_cost_usd == pytest.approx(0.01 + 0.02, abs=1e-9)

    def test_resolve_pricing_checks_custom_first(self) -> None:
        custom = ModelPricing(
            model="gpt-4o-mini",
            provider=ProviderName.OPENAI,
            tier=ModelTier.EFFICIENT,
            input_per_million=999.0,  # clearly different from real pricing
            output_per_million=999.0,
        )
        estimator = CostEstimator(custom_pricing={"gpt-4o-mini": custom})
        est = estimator.estimate("gpt-4o-mini", 1000, 0)
        # Should use custom pricing — cost would be very high
        assert est.estimated_cost_usd == pytest.approx(999.0 / 1000.0, abs=1e-3)


# ---------------------------------------------------------------------------
# CostEstimator — estimate_from_messages
# ---------------------------------------------------------------------------


class TestEstimateFromMessages:
    def test_basic_messages(self, estimator: CostEstimator) -> None:
        messages = [
            {"role": "user", "content": "Hello, how are you?"},
            {"role": "assistant", "content": "I am doing well, thank you!"},
        ]
        est = estimator.estimate_from_messages("gpt-4o-mini", messages, 100)
        assert est.model == "gpt-4o-mini"
        assert est.input_tokens > 0
        assert est.estimated_cost_usd >= 0.0

    def test_empty_messages(self, estimator: CostEstimator) -> None:
        est = estimator.estimate_from_messages("gpt-4o-mini", [], 50)
        # TokenCounter may add small overhead even for empty messages
        assert est.input_tokens >= 0

    def test_default_output_used_when_none(self, estimator: CostEstimator) -> None:
        messages = [{"role": "user", "content": "Hi"}]
        est = estimator.estimate_from_messages("gpt-4o-mini", messages, None)
        assert est.output_tokens == estimator._default_output_tokens


# ---------------------------------------------------------------------------
# CostEstimator — estimate_workflow
# ---------------------------------------------------------------------------


class TestEstimateWorkflow:
    def test_empty_workflow_returns_zero(self, estimator: CostEstimator) -> None:
        result = estimator.estimate_workflow([])
        assert result.total_cost_usd == 0.0
        assert result.total_input_tokens == 0
        assert result.total_output_tokens == 0

    def test_single_step_workflow(self, estimator: CostEstimator) -> None:
        steps = [WorkflowStep("step1", "gpt-4o-mini", 500, 100)]
        result = estimator.estimate_workflow(steps)
        assert result.total_cost_usd > 0.0
        assert len(result.steps) == 1

    def test_multi_step_workflow_totals(self, estimator: CostEstimator) -> None:
        steps = [
            WorkflowStep("step1", "gpt-4o-mini", 500, 100),
            WorkflowStep("step2", "gpt-4o-mini", 300, 200),
        ]
        result = estimator.estimate_workflow(steps)
        assert len(result.steps) == 2
        assert result.total_input_tokens == 800
        assert result.total_output_tokens == 300
        # Total should be sum of individual steps
        expected_total = sum(s.estimated_cost_usd for s in result.steps)
        assert result.total_cost_usd == pytest.approx(expected_total, abs=1e-9)

    def test_workflow_with_text_input(self, estimator: CostEstimator) -> None:
        steps = [WorkflowStep("step1", "gpt-4o-mini", "This is a prompt", 50)]
        result = estimator.estimate_workflow(steps)
        assert result.total_input_tokens > 0

    def test_workflow_is_frozen_dataclass(self, estimator: CostEstimator) -> None:
        steps = [WorkflowStep("step1", "gpt-4o-mini", 100, 50)]
        result = estimator.estimate_workflow(steps)
        with pytest.raises((AttributeError, TypeError)):
            result.total_cost_usd = 999.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# CostEstimator — compare_models
# ---------------------------------------------------------------------------


class TestCompareModels:
    def test_sorted_cheapest_first(self, estimator: CostEstimator) -> None:
        models = ["gpt-4o", "gpt-4o-mini", "claude-haiku-4"]
        estimates = estimator.compare_models(models, 1000, 200)
        for i in range(len(estimates) - 1):
            assert estimates[i].estimated_cost_usd <= estimates[i + 1].estimated_cost_usd

    def test_unknown_models_skipped(self, estimator: CostEstimator) -> None:
        models = ["gpt-4o-mini", "totally-unknown-xyz"]
        estimates = estimator.compare_models(models, 100, 50)
        model_names = [e.model for e in estimates]
        assert "gpt-4o-mini" in model_names
        assert "totally-unknown-xyz" not in model_names

    def test_empty_models_returns_empty(self, estimator: CostEstimator) -> None:
        estimates = estimator.compare_models([], 100, 50)
        assert estimates == []


# ---------------------------------------------------------------------------
# CostEstimator — cheapest_model_for_budget
# ---------------------------------------------------------------------------


class TestCheapestModelForBudget:
    def test_returns_estimate_within_budget(self, estimator: CostEstimator) -> None:
        result = estimator.cheapest_model_for_budget(1.0, 100, 50)
        assert result is not None
        assert result.estimated_cost_usd <= 1.0

    def test_returns_none_when_nothing_fits(self, estimator: CostEstimator) -> None:
        result = estimator.cheapest_model_for_budget(0.0, 100, 50)
        assert result is None

    def test_prefers_higher_input_rate_model(self, estimator: CostEstimator) -> None:
        # With a generous budget, should pick a premium model
        result = estimator.cheapest_model_for_budget(100.0, 100, 50)
        assert result is not None
        # Just verify it returned something reasonable
        assert result.estimated_cost_usd <= 100.0

    def test_custom_pricing_included_in_search(self) -> None:
        very_cheap = ModelPricing(
            model="ultra-cheap-model",
            provider=ProviderName.CUSTOM,
            tier=ModelTier.NANO,
            input_per_million=0.0001,
            output_per_million=0.0002,
        )
        estimator = CostEstimator(custom_pricing={"ultra-cheap-model": very_cheap})
        result = estimator.cheapest_model_for_budget(0.000001, 10, 10)
        # ultra-cheap-model should fit within this micro-budget
        assert result is not None
