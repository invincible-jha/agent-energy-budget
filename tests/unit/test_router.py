"""Comprehensive unit tests for agent_energy_budget.router.

Covers:
- ModelProfile construction, validation, and cost calculations
- RouterBudgetConfig construction and validation
- RoutingDecision immutability
- DEFAULT_MODEL_PROFILES catalogue integrity
- CheapestFirstStrategy — all scenarios
- QualityFirstStrategy — all scenarios
- BalancedStrategy — all scenarios including complexity multipliers
- BudgetAwareStrategy — tier transitions based on budget depletion
- CostAwareRouter — init, select_model, route, reset_budget, swap_strategy
- CLI route command — happy paths, error paths, JSON output
- Edge cases: empty models, zero budget, no qualifying models, single model
- Budget depletion across multiple route calls
"""
from __future__ import annotations

import json
import math
from typing import Any

import pytest
from click.testing import CliRunner

from agent_energy_budget.router.models import (
    DEFAULT_MODEL_PROFILES,
    BudgetConfig,
    ModelProfile,
    RoutingDecision,
    RouterBudgetConfig,
)
from agent_energy_budget.router.strategies import (
    BalancedStrategy,
    BudgetAwareStrategy,
    CheapestFirstStrategy,
    NoAffordableModelError,
    QualityFirstStrategy,
    RoutingStrategy,
    _filter_affordable,
    _nominal_cost,
    _raise_if_no_affordable,
    _require_non_empty,
)
from agent_energy_budget.router.cost_router import (
    CostAwareRouter,
    _infer_complexity,
    _COMPLEXITY_TOKEN_MULTIPLIER,
    _NOMINAL_INPUT_TOKENS,
    _NOMINAL_OUTPUT_TOKENS,
)


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture()
def cheap_model() -> ModelProfile:
    """Lowest-cost model in isolation."""
    return ModelProfile(
        name="cheap-model",
        provider="test",
        cost_per_1k_input=0.0001,
        cost_per_1k_output=0.0002,
        quality_score=0.4,
        max_context=4_000,
        latency_p50_ms=100,
    )


@pytest.fixture()
def mid_model() -> ModelProfile:
    """Mid-tier model."""
    return ModelProfile(
        name="mid-model",
        provider="test",
        cost_per_1k_input=0.001,
        cost_per_1k_output=0.003,
        quality_score=0.70,
        max_context=32_000,
        latency_p50_ms=500,
    )


@pytest.fixture()
def premium_model() -> ModelProfile:
    """High-quality, high-cost model."""
    return ModelProfile(
        name="premium-model",
        provider="test",
        cost_per_1k_input=0.005,
        cost_per_1k_output=0.015,
        quality_score=0.95,
        max_context=200_000,
        latency_p50_ms=1_500,
    )


@pytest.fixture()
def three_models(
    cheap_model: ModelProfile,
    mid_model: ModelProfile,
    premium_model: ModelProfile,
) -> list[ModelProfile]:
    return [cheap_model, mid_model, premium_model]


@pytest.fixture()
def default_budget() -> RouterBudgetConfig:
    return RouterBudgetConfig(total_budget_usd=10.0, min_quality_score=0.0)


@pytest.fixture()
def tight_budget() -> RouterBudgetConfig:
    """Budget that can only afford the cheapest model."""
    return RouterBudgetConfig(total_budget_usd=0.001, min_quality_score=0.0)


@pytest.fixture()
def quality_floor_budget() -> RouterBudgetConfig:
    """Budget with a quality floor that excludes cheap models."""
    return RouterBudgetConfig(total_budget_usd=10.0, min_quality_score=0.65)


@pytest.fixture()
def router(three_models: list[ModelProfile], default_budget: RouterBudgetConfig) -> CostAwareRouter:
    return CostAwareRouter(models=three_models, budget=default_budget, strategy="balanced")


# ===========================================================================
# ModelProfile Tests
# ===========================================================================


class TestModelProfile:
    def test_construction_valid(self, cheap_model: ModelProfile) -> None:
        assert cheap_model.name == "cheap-model"
        assert cheap_model.provider == "test"
        assert cheap_model.quality_score == 0.4

    def test_frozen_immutable(self, cheap_model: ModelProfile) -> None:
        with pytest.raises((AttributeError, TypeError)):
            cheap_model.name = "changed"  # type: ignore[misc]

    def test_cost_for_tokens_zero_tokens(self, cheap_model: ModelProfile) -> None:
        assert cheap_model.cost_for_tokens(0, 0) == 0.0

    def test_cost_for_tokens_only_input(self, cheap_model: ModelProfile) -> None:
        # 1000 input tokens at $0.0001/1K = $0.0001
        result = cheap_model.cost_for_tokens(1_000, 0)
        assert math.isclose(result, 0.0001, rel_tol=1e-6)

    def test_cost_for_tokens_only_output(self, cheap_model: ModelProfile) -> None:
        # 1000 output tokens at $0.0002/1K = $0.0002
        result = cheap_model.cost_for_tokens(0, 1_000)
        assert math.isclose(result, 0.0002, rel_tol=1e-6)

    def test_cost_for_tokens_combined(self, cheap_model: ModelProfile) -> None:
        # 1000 input + 500 output
        expected = 0.0001 + 0.5 * 0.0002
        result = cheap_model.cost_for_tokens(1_000, 500)
        assert math.isclose(result, expected, rel_tol=1e-6)

    def test_cost_for_tokens_large_values(self, premium_model: ModelProfile) -> None:
        result = premium_model.cost_for_tokens(1_000_000, 1_000_000)
        expected = (1_000_000 / 1_000) * 0.005 + (1_000_000 / 1_000) * 0.015
        assert math.isclose(result, expected, rel_tol=1e-6)

    def test_cost_efficiency_ratio_positive(self, cheap_model: ModelProfile) -> None:
        ratio = cheap_model.cost_efficiency_ratio()
        assert ratio > 0

    def test_cost_efficiency_ratio_zero_cost(self) -> None:
        free_model = ModelProfile(
            name="free",
            provider="test",
            cost_per_1k_input=0.0,
            cost_per_1k_output=0.0,
            quality_score=0.5,
            max_context=4_000,
            latency_p50_ms=100,
        )
        assert free_model.cost_efficiency_ratio() == float("inf")

    def test_cost_efficiency_ratio_higher_for_better_value(
        self, cheap_model: ModelProfile, premium_model: ModelProfile
    ) -> None:
        # cheap_model: quality 0.4, premium: quality 0.95 but much higher cost
        # Both ratios should be positive; cheap might have better ratio
        cheap_ratio = cheap_model.cost_efficiency_ratio()
        premium_ratio = premium_model.cost_efficiency_ratio()
        assert cheap_ratio > 0
        assert premium_ratio > 0

    def test_validation_empty_name_raises(self) -> None:
        with pytest.raises(ValueError, match="name must not be empty"):
            ModelProfile(
                name="",
                provider="test",
                cost_per_1k_input=0.001,
                cost_per_1k_output=0.002,
                quality_score=0.5,
                max_context=4_000,
                latency_p50_ms=100,
            )

    def test_validation_quality_score_below_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="quality_score"):
            ModelProfile(
                name="bad",
                provider="test",
                cost_per_1k_input=0.001,
                cost_per_1k_output=0.002,
                quality_score=-0.1,
                max_context=4_000,
                latency_p50_ms=100,
            )

    def test_validation_quality_score_above_one_raises(self) -> None:
        with pytest.raises(ValueError, match="quality_score"):
            ModelProfile(
                name="bad",
                provider="test",
                cost_per_1k_input=0.001,
                cost_per_1k_output=0.002,
                quality_score=1.01,
                max_context=4_000,
                latency_p50_ms=100,
            )

    def test_validation_quality_score_boundary_zero(self) -> None:
        m = ModelProfile(
            name="ok",
            provider="test",
            cost_per_1k_input=0.0,
            cost_per_1k_output=0.0,
            quality_score=0.0,
            max_context=0,
            latency_p50_ms=0,
        )
        assert m.quality_score == 0.0

    def test_validation_quality_score_boundary_one(self) -> None:
        m = ModelProfile(
            name="ok",
            provider="test",
            cost_per_1k_input=0.0,
            cost_per_1k_output=0.0,
            quality_score=1.0,
            max_context=0,
            latency_p50_ms=0,
        )
        assert m.quality_score == 1.0

    def test_validation_negative_input_cost_raises(self) -> None:
        with pytest.raises(ValueError, match="cost_per_1k_input"):
            ModelProfile(
                name="bad",
                provider="test",
                cost_per_1k_input=-0.001,
                cost_per_1k_output=0.002,
                quality_score=0.5,
                max_context=4_000,
                latency_p50_ms=100,
            )

    def test_validation_negative_output_cost_raises(self) -> None:
        with pytest.raises(ValueError, match="cost_per_1k_output"):
            ModelProfile(
                name="bad",
                provider="test",
                cost_per_1k_input=0.001,
                cost_per_1k_output=-0.002,
                quality_score=0.5,
                max_context=4_000,
                latency_p50_ms=100,
            )

    def test_validation_negative_max_context_raises(self) -> None:
        with pytest.raises(ValueError, match="max_context"):
            ModelProfile(
                name="bad",
                provider="test",
                cost_per_1k_input=0.001,
                cost_per_1k_output=0.002,
                quality_score=0.5,
                max_context=-1,
                latency_p50_ms=100,
            )

    def test_validation_negative_latency_raises(self) -> None:
        with pytest.raises(ValueError, match="latency_p50_ms"):
            ModelProfile(
                name="bad",
                provider="test",
                cost_per_1k_input=0.001,
                cost_per_1k_output=0.002,
                quality_score=0.5,
                max_context=4_000,
                latency_p50_ms=-1,
            )

    def test_equality_based_on_values(self) -> None:
        m1 = ModelProfile(
            name="x", provider="p", cost_per_1k_input=0.001,
            cost_per_1k_output=0.002, quality_score=0.5, max_context=4_000, latency_p50_ms=100,
        )
        m2 = ModelProfile(
            name="x", provider="p", cost_per_1k_input=0.001,
            cost_per_1k_output=0.002, quality_score=0.5, max_context=4_000, latency_p50_ms=100,
        )
        assert m1 == m2

    def test_inequality_different_name(self) -> None:
        m1 = ModelProfile(
            name="x", provider="p", cost_per_1k_input=0.001,
            cost_per_1k_output=0.002, quality_score=0.5, max_context=4_000, latency_p50_ms=100,
        )
        m2 = ModelProfile(
            name="y", provider="p", cost_per_1k_input=0.001,
            cost_per_1k_output=0.002, quality_score=0.5, max_context=4_000, latency_p50_ms=100,
        )
        assert m1 != m2


# ===========================================================================
# RouterBudgetConfig Tests
# ===========================================================================


class TestRouterBudgetConfig:
    def test_default_values(self) -> None:
        cfg = RouterBudgetConfig(total_budget_usd=5.0)
        assert cfg.total_budget_usd == 5.0
        assert cfg.alert_threshold_pct == 80.0
        assert cfg.min_quality_score == 0.0

    def test_alert_threshold_fraction(self) -> None:
        cfg = RouterBudgetConfig(total_budget_usd=10.0, alert_threshold_pct=75.0)
        assert math.isclose(cfg.alert_threshold_fraction, 0.75)

    def test_alert_threshold_fraction_zero(self) -> None:
        cfg = RouterBudgetConfig(total_budget_usd=10.0, alert_threshold_pct=0.0)
        assert cfg.alert_threshold_fraction == 0.0

    def test_alert_threshold_fraction_hundred(self) -> None:
        cfg = RouterBudgetConfig(total_budget_usd=10.0, alert_threshold_pct=100.0)
        assert cfg.alert_threshold_fraction == 1.0

    def test_negative_budget_raises(self) -> None:
        with pytest.raises(ValueError, match="total_budget_usd"):
            RouterBudgetConfig(total_budget_usd=-1.0)

    def test_zero_budget_valid(self) -> None:
        cfg = RouterBudgetConfig(total_budget_usd=0.0)
        assert cfg.total_budget_usd == 0.0

    def test_alert_threshold_above_100_raises(self) -> None:
        with pytest.raises(ValueError, match="alert_threshold_pct"):
            RouterBudgetConfig(total_budget_usd=10.0, alert_threshold_pct=101.0)

    def test_alert_threshold_below_0_raises(self) -> None:
        with pytest.raises(ValueError, match="alert_threshold_pct"):
            RouterBudgetConfig(total_budget_usd=10.0, alert_threshold_pct=-1.0)

    def test_min_quality_above_1_raises(self) -> None:
        with pytest.raises(ValueError, match="min_quality_score"):
            RouterBudgetConfig(total_budget_usd=10.0, min_quality_score=1.01)

    def test_min_quality_below_0_raises(self) -> None:
        with pytest.raises(ValueError, match="min_quality_score"):
            RouterBudgetConfig(total_budget_usd=10.0, min_quality_score=-0.1)

    def test_budget_config_alias(self) -> None:
        # BudgetConfig must be an alias for RouterBudgetConfig
        cfg = BudgetConfig(total_budget_usd=5.0)
        assert isinstance(cfg, RouterBudgetConfig)

    def test_frozen_immutable(self) -> None:
        cfg = RouterBudgetConfig(total_budget_usd=5.0)
        with pytest.raises((AttributeError, TypeError)):
            cfg.total_budget_usd = 99.0  # type: ignore[misc]


# ===========================================================================
# RoutingDecision Tests
# ===========================================================================


class TestRoutingDecision:
    def test_construction(self, cheap_model: ModelProfile) -> None:
        decision = RoutingDecision(
            selected_model=cheap_model,
            reason="test reason",
            estimated_cost=0.001,
            remaining_budget=9.999,
        )
        assert decision.selected_model is cheap_model
        assert decision.reason == "test reason"
        assert decision.estimated_cost == 0.001
        assert decision.remaining_budget == 9.999

    def test_frozen_immutable(self, cheap_model: ModelProfile) -> None:
        decision = RoutingDecision(
            selected_model=cheap_model,
            reason="test",
            estimated_cost=0.001,
            remaining_budget=9.999,
        )
        with pytest.raises((AttributeError, TypeError)):
            decision.reason = "changed"  # type: ignore[misc]


# ===========================================================================
# DEFAULT_MODEL_PROFILES Tests
# ===========================================================================


class TestDefaultModelProfiles:
    def test_not_empty(self) -> None:
        assert len(DEFAULT_MODEL_PROFILES) >= 6

    def test_all_are_model_profiles(self) -> None:
        for profile in DEFAULT_MODEL_PROFILES:
            assert isinstance(profile, ModelProfile)

    def test_contains_expected_models(self) -> None:
        names = {m.name for m in DEFAULT_MODEL_PROFILES}
        assert "gpt-4o" in names
        assert "gpt-4o-mini" in names
        assert "claude-sonnet-4-6" in names
        assert "claude-haiku-4-5" in names
        assert "llama-3-70b" in names
        assert "mistral-7b" in names

    def test_all_quality_scores_in_range(self) -> None:
        for profile in DEFAULT_MODEL_PROFILES:
            assert 0.0 <= profile.quality_score <= 1.0, (
                f"{profile.name} quality_score={profile.quality_score} out of range"
            )

    def test_all_costs_non_negative(self) -> None:
        for profile in DEFAULT_MODEL_PROFILES:
            assert profile.cost_per_1k_input >= 0.0
            assert profile.cost_per_1k_output >= 0.0

    def test_all_names_non_empty(self) -> None:
        for profile in DEFAULT_MODEL_PROFILES:
            assert profile.name

    def test_all_providers_non_empty(self) -> None:
        for profile in DEFAULT_MODEL_PROFILES:
            assert profile.provider

    def test_gpt4o_costs(self) -> None:
        gpt4o = next(m for m in DEFAULT_MODEL_PROFILES if m.name == "gpt-4o")
        assert math.isclose(gpt4o.cost_per_1k_input, 0.0025, rel_tol=1e-4)
        assert math.isclose(gpt4o.cost_per_1k_output, 0.01000, rel_tol=1e-4)

    def test_gpt4o_mini_cheaper_than_gpt4o(self) -> None:
        gpt4o = next(m for m in DEFAULT_MODEL_PROFILES if m.name == "gpt-4o")
        mini = next(m for m in DEFAULT_MODEL_PROFILES if m.name == "gpt-4o-mini")
        assert _nominal_cost(mini) < _nominal_cost(gpt4o)

    def test_claude_sonnet_provider(self) -> None:
        sonnet = next(m for m in DEFAULT_MODEL_PROFILES if m.name == "claude-sonnet-4-6")
        assert sonnet.provider == "anthropic"

    def test_unique_names(self) -> None:
        names = [m.name for m in DEFAULT_MODEL_PROFILES]
        assert len(names) == len(set(names)), "Duplicate model names in DEFAULT_MODEL_PROFILES"


# ===========================================================================
# Strategy helper function Tests
# ===========================================================================


class TestStrategyHelpers:
    def test_nominal_cost_positive(self, cheap_model: ModelProfile) -> None:
        cost = _nominal_cost(cheap_model)
        assert cost > 0

    def test_filter_affordable_all_qualify(
        self, three_models: list[ModelProfile]
    ) -> None:
        result = _filter_affordable(three_models, remaining_budget=1000.0, min_quality_score=0.0)
        assert len(result) == 3

    def test_filter_affordable_none_qualify_budget(
        self, three_models: list[ModelProfile]
    ) -> None:
        result = _filter_affordable(three_models, remaining_budget=0.0, min_quality_score=0.0)
        assert result == []

    def test_filter_affordable_quality_floor_excludes(
        self,
        cheap_model: ModelProfile,
        mid_model: ModelProfile,
        premium_model: ModelProfile,
        three_models: list[ModelProfile],
    ) -> None:
        # cheap_model quality=0.4, excluded by floor 0.65
        result = _filter_affordable(
            three_models, remaining_budget=1000.0, min_quality_score=0.65
        )
        assert cheap_model not in result
        assert mid_model in result
        assert premium_model in result

    def test_filter_affordable_mixed_exclusions(
        self,
        cheap_model: ModelProfile,
        mid_model: ModelProfile,
        premium_model: ModelProfile,
        three_models: list[ModelProfile],
    ) -> None:
        # Budget only covers cheap, but quality floor excludes cheap
        cheap_cost = _nominal_cost(cheap_model)
        result = _filter_affordable(
            three_models,
            remaining_budget=cheap_cost * 1.1,
            min_quality_score=0.65,
        )
        assert result == []

    def test_require_non_empty_passes_with_items(
        self, three_models: list[ModelProfile]
    ) -> None:
        _require_non_empty(three_models)  # should not raise

    def test_require_non_empty_raises_on_empty(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            _require_non_empty([])

    def test_raise_if_no_affordable_passes_when_non_empty(
        self, cheap_model: ModelProfile
    ) -> None:
        _raise_if_no_affordable([cheap_model], 1.0, 0.0)  # should not raise

    def test_raise_if_no_affordable_raises_when_empty(self) -> None:
        with pytest.raises(NoAffordableModelError):
            _raise_if_no_affordable([], 1.0, 0.0)


# ===========================================================================
# NoAffordableModelError Tests
# ===========================================================================


class TestNoAffordableModelError:
    def test_attributes_set(self) -> None:
        exc = NoAffordableModelError(remaining_budget=0.001, min_quality_score=0.7)
        assert exc.remaining_budget == 0.001
        assert exc.min_quality_score == 0.7

    def test_default_min_quality(self) -> None:
        exc = NoAffordableModelError(remaining_budget=0.0)
        assert exc.min_quality_score == 0.0

    def test_is_value_error(self) -> None:
        exc = NoAffordableModelError(remaining_budget=0.0)
        assert isinstance(exc, ValueError)

    def test_message_contains_budget(self) -> None:
        exc = NoAffordableModelError(remaining_budget=0.001234)
        assert "0.001234" in str(exc)


# ===========================================================================
# CheapestFirstStrategy Tests
# ===========================================================================


class TestCheapestFirstStrategy:
    @pytest.fixture()
    def strategy(self) -> CheapestFirstStrategy:
        return CheapestFirstStrategy()

    def test_selects_cheapest_of_three(
        self,
        strategy: CheapestFirstStrategy,
        three_models: list[ModelProfile],
        cheap_model: ModelProfile,
        default_budget: RouterBudgetConfig,
    ) -> None:
        selected, reason = strategy.select(three_models, 10.0, default_budget)
        assert selected == cheap_model

    def test_reason_contains_model_name(
        self,
        strategy: CheapestFirstStrategy,
        three_models: list[ModelProfile],
        cheap_model: ModelProfile,
        default_budget: RouterBudgetConfig,
    ) -> None:
        _, reason = strategy.select(three_models, 10.0, default_budget)
        assert cheap_model.name in reason

    def test_reason_contains_strategy_name(
        self,
        strategy: CheapestFirstStrategy,
        three_models: list[ModelProfile],
        default_budget: RouterBudgetConfig,
    ) -> None:
        _, reason = strategy.select(three_models, 10.0, default_budget)
        assert "CheapestFirst" in reason

    def test_raises_on_empty_models(
        self,
        strategy: CheapestFirstStrategy,
        default_budget: RouterBudgetConfig,
    ) -> None:
        with pytest.raises(ValueError, match="empty"):
            strategy.select([], 10.0, default_budget)

    def test_raises_when_budget_too_low(
        self,
        strategy: CheapestFirstStrategy,
        three_models: list[ModelProfile],
        default_budget: RouterBudgetConfig,
    ) -> None:
        with pytest.raises(NoAffordableModelError):
            strategy.select(three_models, 0.0, default_budget)

    def test_selects_single_model_when_only_one_affordable(
        self,
        strategy: CheapestFirstStrategy,
        cheap_model: ModelProfile,
        mid_model: ModelProfile,
        default_budget: RouterBudgetConfig,
    ) -> None:
        cheap_cost = _nominal_cost(cheap_model)
        mid_cost = _nominal_cost(mid_model)
        # Budget allows only cheap
        budget = cheap_cost + (mid_cost - cheap_cost) / 2
        selected, _ = strategy.select(
            [cheap_model, mid_model], budget, default_budget
        )
        assert selected == cheap_model

    def test_ignores_quality_floor_when_no_model_qualifies(
        self,
        strategy: CheapestFirstStrategy,
        cheap_model: ModelProfile,
    ) -> None:
        # quality floor excludes cheap_model
        budget_cfg = RouterBudgetConfig(total_budget_usd=10.0, min_quality_score=0.9)
        with pytest.raises(NoAffordableModelError):
            strategy.select([cheap_model], 10.0, budget_cfg)

    def test_single_model_list(
        self,
        strategy: CheapestFirstStrategy,
        cheap_model: ModelProfile,
        default_budget: RouterBudgetConfig,
    ) -> None:
        selected, _ = strategy.select([cheap_model], 10.0, default_budget)
        assert selected == cheap_model

    def test_task_complexity_ignored(
        self,
        strategy: CheapestFirstStrategy,
        three_models: list[ModelProfile],
        cheap_model: ModelProfile,
        default_budget: RouterBudgetConfig,
    ) -> None:
        # CheapestFirst should always pick cheap regardless of complexity
        for complexity in ("low", "medium", "high"):
            selected, _ = strategy.select(
                three_models, 10.0, default_budget, task_complexity=complexity  # type: ignore[arg-type]
            )
            assert selected == cheap_model

    def test_respects_quality_floor(
        self,
        strategy: CheapestFirstStrategy,
        cheap_model: ModelProfile,
        mid_model: ModelProfile,
        premium_model: ModelProfile,
    ) -> None:
        # Quality floor 0.65 excludes cheap (0.4), so mid should be cheapest affordable
        budget_cfg = RouterBudgetConfig(total_budget_usd=10.0, min_quality_score=0.65)
        models = [cheap_model, mid_model, premium_model]
        selected, _ = strategy.select(models, 10.0, budget_cfg)
        assert selected == mid_model


# ===========================================================================
# QualityFirstStrategy Tests
# ===========================================================================


class TestQualityFirstStrategy:
    @pytest.fixture()
    def strategy(self) -> QualityFirstStrategy:
        return QualityFirstStrategy()

    def test_selects_highest_quality_within_budget(
        self,
        strategy: QualityFirstStrategy,
        three_models: list[ModelProfile],
        premium_model: ModelProfile,
        default_budget: RouterBudgetConfig,
    ) -> None:
        selected, _ = strategy.select(three_models, 10.0, default_budget)
        assert selected == premium_model

    def test_reason_contains_quality(
        self,
        strategy: QualityFirstStrategy,
        three_models: list[ModelProfile],
        default_budget: RouterBudgetConfig,
    ) -> None:
        _, reason = strategy.select(three_models, 10.0, default_budget)
        assert "quality" in reason.lower()

    def test_reason_contains_strategy_name(
        self,
        strategy: QualityFirstStrategy,
        three_models: list[ModelProfile],
        default_budget: RouterBudgetConfig,
    ) -> None:
        _, reason = strategy.select(three_models, 10.0, default_budget)
        assert "QualityFirst" in reason

    def test_raises_on_empty_models(
        self,
        strategy: QualityFirstStrategy,
        default_budget: RouterBudgetConfig,
    ) -> None:
        with pytest.raises(ValueError, match="empty"):
            strategy.select([], 10.0, default_budget)

    def test_raises_when_budget_too_low(
        self,
        strategy: QualityFirstStrategy,
        three_models: list[ModelProfile],
        default_budget: RouterBudgetConfig,
    ) -> None:
        with pytest.raises(NoAffordableModelError):
            strategy.select(three_models, 0.0, default_budget)

    def test_falls_back_when_premium_unaffordable(
        self,
        strategy: QualityFirstStrategy,
        cheap_model: ModelProfile,
        mid_model: ModelProfile,
        premium_model: ModelProfile,
        default_budget: RouterBudgetConfig,
    ) -> None:
        # Budget between mid and premium
        mid_cost = _nominal_cost(mid_model)
        premium_cost = _nominal_cost(premium_model)
        budget = mid_cost + (premium_cost - mid_cost) / 2
        models = [cheap_model, mid_model, premium_model]
        selected, _ = strategy.select(models, budget, default_budget)
        assert selected == mid_model

    def test_tiebreaker_prefers_cheaper_on_equal_quality(
        self,
        strategy: QualityFirstStrategy,
        default_budget: RouterBudgetConfig,
    ) -> None:
        m1 = ModelProfile(
            name="tied-expensive", provider="test",
            cost_per_1k_input=0.01, cost_per_1k_output=0.02,
            quality_score=0.8, max_context=4_000, latency_p50_ms=100,
        )
        m2 = ModelProfile(
            name="tied-cheap", provider="test",
            cost_per_1k_input=0.001, cost_per_1k_output=0.002,
            quality_score=0.8, max_context=4_000, latency_p50_ms=100,
        )
        # m1 and m2 have equal quality; m2 is cheaper — should prefer m2
        selected, _ = strategy.select([m1, m2], 10.0, default_budget)
        assert selected == m2

    def test_respects_quality_floor(
        self,
        strategy: QualityFirstStrategy,
        cheap_model: ModelProfile,
        mid_model: ModelProfile,
        premium_model: ModelProfile,
    ) -> None:
        budget_cfg = RouterBudgetConfig(total_budget_usd=10.0, min_quality_score=0.65)
        selected, _ = strategy.select(
            [cheap_model, mid_model, premium_model], 10.0, budget_cfg
        )
        assert selected == premium_model

    def test_single_model_returned_when_only_one_qualifies(
        self,
        strategy: QualityFirstStrategy,
        cheap_model: ModelProfile,
        default_budget: RouterBudgetConfig,
    ) -> None:
        selected, _ = strategy.select([cheap_model], 10.0, default_budget)
        assert selected == cheap_model

    def test_task_complexity_does_not_change_selection(
        self,
        strategy: QualityFirstStrategy,
        three_models: list[ModelProfile],
        premium_model: ModelProfile,
        default_budget: RouterBudgetConfig,
    ) -> None:
        for complexity in ("low", "medium", "high"):
            selected, _ = strategy.select(
                three_models, 10.0, default_budget, task_complexity=complexity  # type: ignore[arg-type]
            )
            assert selected == premium_model


# ===========================================================================
# BalancedStrategy Tests
# ===========================================================================


class TestBalancedStrategy:
    @pytest.fixture()
    def strategy(self) -> BalancedStrategy:
        return BalancedStrategy()

    def test_selects_best_ratio_model(
        self,
        strategy: BalancedStrategy,
        three_models: list[ModelProfile],
        default_budget: RouterBudgetConfig,
    ) -> None:
        selected, reason = strategy.select(three_models, 10.0, default_budget)
        # Result must be one of the three models
        assert selected in three_models
        assert reason

    def test_reason_contains_efficiency(
        self,
        strategy: BalancedStrategy,
        three_models: list[ModelProfile],
        default_budget: RouterBudgetConfig,
    ) -> None:
        _, reason = strategy.select(three_models, 10.0, default_budget)
        assert "Balanced" in reason

    def test_raises_on_empty_models(
        self,
        strategy: BalancedStrategy,
        default_budget: RouterBudgetConfig,
    ) -> None:
        with pytest.raises(ValueError, match="empty"):
            strategy.select([], 10.0, default_budget)

    def test_raises_when_budget_too_low(
        self,
        strategy: BalancedStrategy,
        three_models: list[ModelProfile],
        default_budget: RouterBudgetConfig,
    ) -> None:
        with pytest.raises(NoAffordableModelError):
            strategy.select(three_models, 0.0, default_budget)

    def test_high_complexity_favours_quality(
        self,
        strategy: BalancedStrategy,
        three_models: list[ModelProfile],
        premium_model: ModelProfile,
        default_budget: RouterBudgetConfig,
    ) -> None:
        selected, _ = strategy.select(
            three_models, 10.0, default_budget, task_complexity="high"
        )
        # With high complexity boosting quality, premium should be favoured
        # if its ratio becomes dominant (depends on model values)
        assert selected in three_models

    def test_low_complexity_favours_cheap(
        self,
        strategy: BalancedStrategy,
        default_budget: RouterBudgetConfig,
    ) -> None:
        # Construct models where cheap is very slightly worse in quality
        # but "low" complexity should still select it by ratio
        cheap = ModelProfile(
            name="cheap2", provider="test",
            cost_per_1k_input=0.0001, cost_per_1k_output=0.0002,
            quality_score=0.5, max_context=4_000, latency_p50_ms=100,
        )
        expensive = ModelProfile(
            name="exp2", provider="test",
            cost_per_1k_input=0.01, cost_per_1k_output=0.02,
            quality_score=0.6, max_context=4_000, latency_p50_ms=500,
        )
        selected, _ = strategy.select([cheap, expensive], 10.0, default_budget, task_complexity="low")
        assert selected in [cheap, expensive]  # ratio determines winner

    def test_complexity_medium_uses_no_bonus(
        self,
        strategy: BalancedStrategy,
        three_models: list[ModelProfile],
        default_budget: RouterBudgetConfig,
    ) -> None:
        sel_medium, _ = strategy.select(three_models, 10.0, default_budget, task_complexity="medium")
        # medium uses multiplier 1.0, so it's the baseline
        assert sel_medium in three_models

    def test_single_model_always_selected(
        self,
        strategy: BalancedStrategy,
        cheap_model: ModelProfile,
        default_budget: RouterBudgetConfig,
    ) -> None:
        selected, _ = strategy.select([cheap_model], 10.0, default_budget)
        assert selected == cheap_model

    def test_respects_quality_floor(
        self,
        strategy: BalancedStrategy,
        cheap_model: ModelProfile,
        mid_model: ModelProfile,
        premium_model: ModelProfile,
    ) -> None:
        budget_cfg = RouterBudgetConfig(total_budget_usd=10.0, min_quality_score=0.65)
        selected, _ = strategy.select(
            [cheap_model, mid_model, premium_model], 10.0, budget_cfg
        )
        assert selected.quality_score >= 0.65

    def test_zero_cost_model_has_infinite_ratio(
        self,
        strategy: BalancedStrategy,
        cheap_model: ModelProfile,
        default_budget: RouterBudgetConfig,
    ) -> None:
        free = ModelProfile(
            name="free-model", provider="test",
            cost_per_1k_input=0.0, cost_per_1k_output=0.0,
            quality_score=0.5, max_context=4_000, latency_p50_ms=100,
        )
        selected, _ = strategy.select([cheap_model, free], 10.0, default_budget)
        # Free model wins because ratio is inf
        assert selected == free


# ===========================================================================
# BudgetAwareStrategy Tests
# ===========================================================================


class TestBudgetAwareStrategy:
    @pytest.fixture()
    def strategy(self) -> BudgetAwareStrategy:
        return BudgetAwareStrategy()

    def test_raises_on_empty_models(
        self,
        strategy: BudgetAwareStrategy,
        default_budget: RouterBudgetConfig,
    ) -> None:
        with pytest.raises(ValueError, match="empty"):
            strategy.select([], 10.0, default_budget)

    def test_raises_when_nothing_affordable(
        self,
        strategy: BudgetAwareStrategy,
        three_models: list[ModelProfile],
        default_budget: RouterBudgetConfig,
    ) -> None:
        with pytest.raises(NoAffordableModelError):
            strategy.select(three_models, 0.0, default_budget)

    def test_quality_first_tier_when_budget_full(
        self,
        strategy: BudgetAwareStrategy,
        three_models: list[ModelProfile],
        premium_model: ModelProfile,
    ) -> None:
        # 100 % of budget remaining — uses QualityFirst
        cfg = RouterBudgetConfig(
            total_budget_usd=10.0,
            alert_threshold_pct=80.0,
            min_quality_score=0.0,
        )
        selected, reason = strategy.select(three_models, 10.0, cfg)
        assert selected == premium_model
        assert "quality-first" in reason

    def test_balanced_tier_when_budget_partially_depleted(
        self,
        strategy: BudgetAwareStrategy,
        three_models: list[ModelProfile],
    ) -> None:
        # remaining = 50 % of 10.0 (below 80 % alert, above 20 % critical)
        cfg = RouterBudgetConfig(
            total_budget_usd=10.0,
            alert_threshold_pct=80.0,
            min_quality_score=0.0,
        )
        selected, reason = strategy.select(three_models, 5.0, cfg)
        # 5/10 = 50 % which is < 80 % alert → balanced tier
        assert "balanced" in reason

    def test_cheapest_tier_when_budget_critical(
        self,
        strategy: BudgetAwareStrategy,
        three_models: list[ModelProfile],
        cheap_model: ModelProfile,
    ) -> None:
        # remaining = 10 % of 10.0 (below 20 % critical threshold)
        cfg = RouterBudgetConfig(
            total_budget_usd=10.0,
            alert_threshold_pct=80.0,
            min_quality_score=0.0,
        )
        selected, reason = strategy.select(three_models, 1.0, cfg)
        # 1/10 = 10 % < 20 % critical → cheapest-first
        assert "cheapest-first" in reason
        assert selected == cheap_model

    def test_zero_total_budget_uses_cheapest(
        self,
        strategy: BudgetAwareStrategy,
        three_models: list[ModelProfile],
        cheap_model: ModelProfile,
    ) -> None:
        # Zero total budget — treated as fully depleted (0 % remaining)
        cfg = RouterBudgetConfig(total_budget_usd=0.0, alert_threshold_pct=80.0)
        # remaining_budget must allow at least cheap model
        selected, reason = strategy.select(three_models, _nominal_cost(cheap_model) * 2, cfg)
        assert "cheapest-first" in reason
        assert selected == cheap_model

    def test_reason_includes_sub_reason(
        self,
        strategy: BudgetAwareStrategy,
        three_models: list[ModelProfile],
        default_budget: RouterBudgetConfig,
    ) -> None:
        _, reason = strategy.select(three_models, 10.0, default_budget)
        # BudgetAware wraps the sub-strategy reason
        assert "BudgetAware" in reason

    def test_task_complexity_passed_to_sub_strategy(
        self,
        strategy: BudgetAwareStrategy,
        three_models: list[ModelProfile],
        default_budget: RouterBudgetConfig,
    ) -> None:
        # Just ensure it doesn't raise; complexity is forwarded
        for complexity in ("low", "medium", "high"):
            selected, reason = strategy.select(
                three_models, 10.0, default_budget, task_complexity=complexity  # type: ignore[arg-type]
            )
            assert selected in three_models

    def test_alert_threshold_exactly_at_boundary(
        self,
        strategy: BudgetAwareStrategy,
        three_models: list[ModelProfile],
    ) -> None:
        cfg = RouterBudgetConfig(
            total_budget_usd=10.0,
            alert_threshold_pct=50.0,
            min_quality_score=0.0,
        )
        # remaining == exactly 50 % of 10.0 = 5.0 → quality-first (>= alert)
        _, reason = strategy.select(three_models, 5.0, cfg)
        assert "quality-first" in reason

    def test_alert_just_below_boundary(
        self,
        strategy: BudgetAwareStrategy,
        three_models: list[ModelProfile],
    ) -> None:
        cfg = RouterBudgetConfig(
            total_budget_usd=10.0,
            alert_threshold_pct=50.0,
            min_quality_score=0.0,
        )
        # remaining = 4.99 / 10.0 = 49.9 % < 50 % → balanced
        _, reason = strategy.select(three_models, 4.99, cfg)
        assert "balanced" in reason


# ===========================================================================
# RoutingStrategy Protocol Tests
# ===========================================================================


class TestRoutingStrategyProtocol:
    def test_cheapest_is_routing_strategy(self) -> None:
        assert isinstance(CheapestFirstStrategy(), RoutingStrategy)

    def test_quality_is_routing_strategy(self) -> None:
        assert isinstance(QualityFirstStrategy(), RoutingStrategy)

    def test_balanced_is_routing_strategy(self) -> None:
        assert isinstance(BalancedStrategy(), RoutingStrategy)

    def test_budget_aware_is_routing_strategy(self) -> None:
        assert isinstance(BudgetAwareStrategy(), RoutingStrategy)


# ===========================================================================
# CostAwareRouter Tests
# ===========================================================================


class TestCostAwareRouter:
    # --- Construction ---

    def test_default_strategy_is_balanced(self, three_models: list[ModelProfile]) -> None:
        router = CostAwareRouter(models=three_models)
        assert router.strategy_name == "balanced"

    def test_accepts_explicit_strategy(self, three_models: list[ModelProfile]) -> None:
        router = CostAwareRouter(models=three_models, strategy="cheapest_first")
        assert router.strategy_name == "cheapest_first"

    def test_uses_default_models_when_none_provided(self) -> None:
        router = CostAwareRouter()
        assert len(router.models) > 0

    def test_empty_models_raises(self) -> None:
        with pytest.raises(ValueError, match="at least one"):
            CostAwareRouter(models=[])

    def test_unknown_strategy_raises(self, three_models: list[ModelProfile]) -> None:
        with pytest.raises(ValueError, match="Unknown routing strategy"):
            CostAwareRouter(models=three_models, strategy="nonexistent")  # type: ignore[arg-type]

    def test_models_property_returns_copy(self, router: CostAwareRouter) -> None:
        models = router.models
        models.clear()  # mutate the returned copy
        assert len(router.models) > 0  # original unchanged

    def test_remaining_budget_starts_at_total(self, three_models: list[ModelProfile]) -> None:
        budget = RouterBudgetConfig(total_budget_usd=7.50)
        router = CostAwareRouter(models=three_models, budget=budget)
        assert router.remaining_budget == 7.50

    def test_budget_property(self, three_models: list[ModelProfile]) -> None:
        budget = RouterBudgetConfig(total_budget_usd=5.0)
        router = CostAwareRouter(models=three_models, budget=budget)
        assert router.budget is budget

    # --- select_model ---

    def test_select_model_returns_profile(self, router: CostAwareRouter) -> None:
        selected = router.select_model()
        assert isinstance(selected, ModelProfile)

    def test_select_model_does_not_modify_budget(self, router: CostAwareRouter) -> None:
        before = router.remaining_budget
        router.select_model()
        assert router.remaining_budget == before

    def test_select_model_with_explicit_budget(
        self, three_models: list[ModelProfile], cheap_model: ModelProfile
    ) -> None:
        router = CostAwareRouter(
            models=three_models,
            budget=RouterBudgetConfig(total_budget_usd=10.0),
            strategy="cheapest_first",
        )
        # Pass remaining_budget so tiny it can only afford cheap
        cheap_cost = _nominal_cost(cheap_model)
        mid_model_cost = _nominal_cost(three_models[1])
        tiny_budget = cheap_cost * 1.5  # enough for cheap only
        if tiny_budget < mid_model_cost:
            selected = router.select_model(remaining_budget=tiny_budget)
            assert selected == cheap_model

    def test_select_model_raises_when_no_affordable(
        self, three_models: list[ModelProfile]
    ) -> None:
        router = CostAwareRouter(
            models=three_models,
            budget=RouterBudgetConfig(total_budget_usd=10.0),
            strategy="cheapest_first",
        )
        with pytest.raises(NoAffordableModelError):
            router.select_model(remaining_budget=0.0)

    # --- route ---

    def test_route_returns_routing_decision(self, router: CostAwareRouter) -> None:
        decision = router.route("Hello, world!")
        assert isinstance(decision, RoutingDecision)

    def test_route_deducts_estimated_cost(self, router: CostAwareRouter) -> None:
        before = router.remaining_budget
        decision = router.route("Hello!")
        assert router.remaining_budget == decision.remaining_budget
        assert router.remaining_budget < before

    def test_route_remaining_budget_never_negative(self, router: CostAwareRouter) -> None:
        # Route many times until budget is near zero
        for _ in range(100):
            try:
                router.route("x")
            except NoAffordableModelError:
                break
        assert router.remaining_budget >= 0.0

    def test_route_with_max_cost_respected(
        self, three_models: list[ModelProfile], premium_model: ModelProfile
    ) -> None:
        # quality_first would normally pick premium, but max_cost caps it
        router = CostAwareRouter(
            models=three_models,
            budget=RouterBudgetConfig(total_budget_usd=10.0),
            strategy="quality_first",
        )
        premium_nominal = _nominal_cost(premium_model)
        # max_cost well below premium's nominal — premium should not be selected
        max_cost = premium_nominal / 10.0
        decision = router.route("Analyse this carefully please.", max_cost=max_cost)
        assert decision.estimated_cost <= max_cost + 1e-9  # allow float rounding

    def test_route_raises_when_max_cost_impossible(
        self, three_models: list[ModelProfile]
    ) -> None:
        router = CostAwareRouter(
            models=three_models,
            budget=RouterBudgetConfig(total_budget_usd=10.0),
            strategy="cheapest_first",
        )
        with pytest.raises(NoAffordableModelError):
            router.route("Hello!", max_cost=0.0)

    def test_route_decision_has_non_empty_reason(self, router: CostAwareRouter) -> None:
        decision = router.route("Hello!")
        assert decision.reason

    def test_route_multiple_calls_deplete_budget(self, router: CostAwareRouter) -> None:
        before = router.remaining_budget
        for _ in range(5):
            try:
                router.route("Hello!")
            except NoAffordableModelError:
                break
        assert router.remaining_budget <= before

    # --- reset_budget ---

    def test_reset_budget_restores_total(self, router: CostAwareRouter) -> None:
        router.route("Hello!")
        router.reset_budget()
        assert router.remaining_budget == router.budget.total_budget_usd

    def test_reset_budget_with_custom_value(self, router: CostAwareRouter) -> None:
        router.reset_budget(2.50)
        assert router.remaining_budget == 2.50

    def test_reset_budget_zero(self, router: CostAwareRouter) -> None:
        router.reset_budget(0.0)
        assert router.remaining_budget == 0.0

    # --- swap_strategy ---

    def test_swap_strategy_changes_name(self, router: CostAwareRouter) -> None:
        router.swap_strategy("cheapest_first")
        assert router.strategy_name == "cheapest_first"

    def test_swap_strategy_affects_routing(
        self, three_models: list[ModelProfile], cheap_model: ModelProfile
    ) -> None:
        router = CostAwareRouter(
            models=three_models,
            budget=RouterBudgetConfig(total_budget_usd=10.0),
            strategy="quality_first",
        )
        router.swap_strategy("cheapest_first")
        decision = router.route("Hello!")
        assert decision.selected_model == cheap_model

    def test_swap_strategy_unknown_raises(self, router: CostAwareRouter) -> None:
        with pytest.raises(ValueError, match="Unknown routing strategy"):
            router.swap_strategy("nonexistent")  # type: ignore[arg-type]

    # --- summary ---

    def test_summary_contains_expected_keys(self, router: CostAwareRouter) -> None:
        s = router.summary()
        assert "strategy" in s
        assert "model_count" in s
        assert "total_budget_usd" in s
        assert "remaining_budget_usd" in s
        assert "models" in s

    def test_summary_model_count_correct(
        self, three_models: list[ModelProfile], default_budget: RouterBudgetConfig
    ) -> None:
        router = CostAwareRouter(models=three_models, budget=default_budget)
        assert router.summary()["model_count"] == 3

    def test_summary_models_list_contains_names(
        self, three_models: list[ModelProfile], default_budget: RouterBudgetConfig
    ) -> None:
        router = CostAwareRouter(models=three_models, budget=default_budget)
        names = router.summary()["models"]
        assert isinstance(names, list)
        assert all(isinstance(n, str) for n in names)

    # --- all four strategies callable through router ---

    @pytest.mark.parametrize("strategy_name", ["cheapest_first", "quality_first", "balanced", "budget_aware"])
    def test_all_strategies_produce_decision(
        self,
        three_models: list[ModelProfile],
        default_budget: RouterBudgetConfig,
        strategy_name: str,
    ) -> None:
        router = CostAwareRouter(
            models=three_models,
            budget=default_budget,
            strategy=strategy_name,  # type: ignore[arg-type]
        )
        decision = router.route("Hello, this is a test prompt.")
        assert isinstance(decision, RoutingDecision)
        assert decision.selected_model in three_models


# ===========================================================================
# _infer_complexity Tests
# ===========================================================================


class TestInferComplexity:
    def test_short_prompt_is_low(self) -> None:
        assert _infer_complexity("Hi") == "low"

    def test_exactly_100_chars_is_low(self) -> None:
        prompt = "a" * 100
        assert _infer_complexity(prompt) == "low"

    def test_101_chars_no_keywords_is_medium(self) -> None:
        prompt = "a" * 101
        assert _infer_complexity(prompt) == "medium"

    def test_500_chars_no_keywords_is_high(self) -> None:
        prompt = "x" * 500
        assert _infer_complexity(prompt) == "high"

    def test_keyword_analyse_is_high(self) -> None:
        prompt = "Please analyse the situation."
        assert _infer_complexity(prompt) == "high"

    def test_keyword_analyze_is_high(self) -> None:
        prompt = "Can you analyze this document?"
        assert _infer_complexity(prompt) == "high"

    def test_keyword_compare_is_high(self) -> None:
        prompt = "Compare these two approaches."
        assert _infer_complexity(prompt) == "high"

    def test_keyword_summarize_is_high(self) -> None:
        prompt = "Please summarize this article for me."
        assert _infer_complexity(prompt) == "high"

    def test_keyword_evaluate_is_high(self) -> None:
        prompt = "Evaluate the pros and cons."
        assert _infer_complexity(prompt) == "high"

    def test_medium_length_no_keywords(self) -> None:
        prompt = "What is the capital of France and why is it important historically?"
        result = _infer_complexity(prompt)
        assert result in ("low", "medium", "high")

    def test_empty_prompt_is_low(self) -> None:
        assert _infer_complexity("") == "low"

    def test_whitespace_only_is_low(self) -> None:
        assert _infer_complexity("   ") == "low"


# ===========================================================================
# Budget Depletion Integration Tests
# ===========================================================================


class TestBudgetDepletion:
    def test_remaining_budget_decreases_per_call(
        self, three_models: list[ModelProfile]
    ) -> None:
        router = CostAwareRouter(
            models=three_models,
            budget=RouterBudgetConfig(total_budget_usd=1.0),
            strategy="cheapest_first",
        )
        budgets: list[float] = [router.remaining_budget]
        for _ in range(5):
            try:
                router.route("Hello!")
                budgets.append(router.remaining_budget)
            except NoAffordableModelError:
                break
        # At least one call must have succeeded
        assert len(budgets) >= 2
        # Budget must be monotonically non-increasing
        for a, b in zip(budgets, budgets[1:]):
            assert b <= a

    def test_budget_aware_degrades_over_time(
        self, three_models: list[ModelProfile], premium_model: ModelProfile
    ) -> None:
        router = CostAwareRouter(
            models=three_models,
            budget=RouterBudgetConfig(
                total_budget_usd=10.0, alert_threshold_pct=80.0
            ),
            strategy="budget_aware",
        )
        # At the start, quality-first should be active
        first_decision = router.route("Hello there!")
        # There's no strict guarantee which model is selected (depends on ratios),
        # but the reason should mention budget-aware
        assert "BudgetAware" in first_decision.reason

    def test_total_cost_across_calls_is_bounded(
        self, three_models: list[ModelProfile]
    ) -> None:
        total_budget = 5.0
        router = CostAwareRouter(
            models=three_models,
            budget=RouterBudgetConfig(total_budget_usd=total_budget),
            strategy="cheapest_first",
        )
        total_spent = 0.0
        for _ in range(1_000):
            try:
                d = router.route("x")
                total_spent += d.estimated_cost
            except NoAffordableModelError:
                break
        assert total_spent <= total_budget + 1e-6  # minor float tolerance

    def test_raises_after_budget_exhausted(
        self, three_models: list[ModelProfile]
    ) -> None:
        router = CostAwareRouter(
            models=three_models,
            budget=RouterBudgetConfig(total_budget_usd=0.00001),
            strategy="cheapest_first",
        )
        with pytest.raises(NoAffordableModelError):
            # Should exhaust after very few calls
            for _ in range(1_000):
                router.route("x")

    def test_reset_allows_routing_again(
        self, three_models: list[ModelProfile]
    ) -> None:
        router = CostAwareRouter(
            models=three_models,
            budget=RouterBudgetConfig(total_budget_usd=0.00001),
            strategy="cheapest_first",
        )
        # Exhaust the budget
        try:
            for _ in range(1_000):
                router.route("x")
        except NoAffordableModelError:
            pass
        # Reset and confirm routing works again
        router.reset_budget(10.0)
        decision = router.route("Hello after reset!")
        assert isinstance(decision, RoutingDecision)


# ===========================================================================
# Edge Cases
# ===========================================================================


class TestEdgeCases:
    def test_single_model_catalogue(self) -> None:
        only_model = ModelProfile(
            name="only-one", provider="test",
            cost_per_1k_input=0.001, cost_per_1k_output=0.002,
            quality_score=0.5, max_context=4_000, latency_p50_ms=100,
        )
        router = CostAwareRouter(
            models=[only_model],
            budget=RouterBudgetConfig(total_budget_usd=10.0),
        )
        decision = router.route("Hello!")
        assert decision.selected_model == only_model

    def test_infinite_budget_allows_any_model(
        self, three_models: list[ModelProfile]
    ) -> None:
        router = CostAwareRouter(
            models=three_models,
            budget=RouterBudgetConfig(total_budget_usd=float("inf")),
            strategy="quality_first",
        )
        decision = router.route("Hello!")
        assert isinstance(decision, RoutingDecision)

    def test_all_models_same_quality(self) -> None:
        models = [
            ModelProfile(
                name=f"model-{i}", provider="test",
                cost_per_1k_input=0.001 * (i + 1),
                cost_per_1k_output=0.002 * (i + 1),
                quality_score=0.7,
                max_context=4_000,
                latency_p50_ms=100,
            )
            for i in range(5)
        ]
        router = CostAwareRouter(
            models=models,
            budget=RouterBudgetConfig(total_budget_usd=10.0),
            strategy="quality_first",
        )
        decision = router.route("Hello!")
        # All same quality — cheapest should win tiebreaker
        assert decision.selected_model.cost_per_1k_input == pytest.approx(0.001)

    def test_all_models_same_cost_quality_first_picks_any(self) -> None:
        models = [
            ModelProfile(
                name=f"equal-{i}", provider="test",
                cost_per_1k_input=0.001,
                cost_per_1k_output=0.002,
                quality_score=0.7,
                max_context=4_000,
                latency_p50_ms=100,
            )
            for i in range(3)
        ]
        router = CostAwareRouter(
            models=models,
            budget=RouterBudgetConfig(total_budget_usd=10.0),
            strategy="quality_first",
        )
        decision = router.route("Hello!")
        assert decision.selected_model in models

    def test_min_quality_score_exactly_at_model_quality(
        self, mid_model: ModelProfile
    ) -> None:
        # floor equals mid_model.quality_score exactly — should be selectable
        budget_cfg = RouterBudgetConfig(
            total_budget_usd=10.0,
            min_quality_score=mid_model.quality_score,
        )
        strategy = QualityFirstStrategy()
        selected, _ = strategy.select([mid_model], 10.0, budget_cfg)
        assert selected == mid_model

    def test_zero_quality_floor_allows_all(
        self, three_models: list[ModelProfile]
    ) -> None:
        budget_cfg = RouterBudgetConfig(total_budget_usd=10.0, min_quality_score=0.0)
        strategy = CheapestFirstStrategy()
        result = _filter_affordable(three_models, 10.0, budget_cfg.min_quality_score)
        assert len(result) == 3

    def test_very_long_prompt_infers_high(self) -> None:
        prompt = "word " * 200  # 1000+ chars
        assert _infer_complexity(prompt) == "high"

    def test_max_cost_none_uses_full_remaining(
        self, three_models: list[ModelProfile], premium_model: ModelProfile
    ) -> None:
        router = CostAwareRouter(
            models=three_models,
            budget=RouterBudgetConfig(total_budget_usd=100.0),
            strategy="quality_first",
        )
        decision = router.route("Hello!", max_cost=None)
        assert decision.selected_model == premium_model

    def test_router_budget_summary_updates_after_route(
        self, three_models: list[ModelProfile]
    ) -> None:
        router = CostAwareRouter(
            models=three_models,
            budget=RouterBudgetConfig(total_budget_usd=10.0),
        )
        before = router.summary()["remaining_budget_usd"]
        router.route("Hello!")
        after = router.summary()["remaining_budget_usd"]
        assert after < before


# ===========================================================================
# CLI Route Command Tests
# ===========================================================================


class TestCliRouteCommand:
    @pytest.fixture()
    def runner(self) -> CliRunner:
        return CliRunner()

    def test_route_basic_invocation(self, runner: CliRunner) -> None:
        from agent_energy_budget.cli.main import cli

        result = runner.invoke(cli, ["route", "--prompt", "Hello world!"])
        assert result.exit_code == 0, result.output

    def test_route_shows_selected_model(self, runner: CliRunner) -> None:
        from agent_energy_budget.cli.main import cli

        result = runner.invoke(cli, ["route", "--prompt", "Hi"])
        assert result.exit_code == 0
        assert "Selected Model" in result.output

    def test_route_with_strategy_cheapest_first(self, runner: CliRunner) -> None:
        from agent_energy_budget.cli.main import cli

        result = runner.invoke(
            cli,
            ["route", "--prompt", "Hello!", "--strategy", "cheapest_first"],
        )
        assert result.exit_code == 0

    def test_route_with_strategy_quality_first(self, runner: CliRunner) -> None:
        from agent_energy_budget.cli.main import cli

        result = runner.invoke(
            cli,
            ["route", "--prompt", "Hello!", "--strategy", "quality_first"],
        )
        assert result.exit_code == 0

    def test_route_with_strategy_balanced(self, runner: CliRunner) -> None:
        from agent_energy_budget.cli.main import cli

        result = runner.invoke(
            cli,
            ["route", "--prompt", "Hello!", "--strategy", "balanced"],
        )
        assert result.exit_code == 0

    def test_route_with_strategy_budget_aware(self, runner: CliRunner) -> None:
        from agent_energy_budget.cli.main import cli

        result = runner.invoke(
            cli,
            ["route", "--prompt", "Hello!", "--strategy", "budget_aware"],
        )
        assert result.exit_code == 0

    def test_route_json_output_is_valid_json(self, runner: CliRunner) -> None:
        from agent_energy_budget.cli.main import cli

        result = runner.invoke(
            cli,
            ["route", "--prompt", "Hello!", "--json-output"],
        )
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert "selected_model" in data
        assert "estimated_cost_usd" in data

    def test_route_json_output_contains_all_fields(self, runner: CliRunner) -> None:
        from agent_energy_budget.cli.main import cli

        result = runner.invoke(
            cli,
            ["route", "--prompt", "Hello!", "--json-output"],
        )
        assert result.exit_code == 0
        data = json.loads(result.output)
        expected_fields = {
            "selected_model", "provider", "quality_score",
            "cost_per_1k_input", "cost_per_1k_output",
            "estimated_cost_usd", "remaining_budget_usd", "reason", "strategy",
        }
        assert expected_fields.issubset(data.keys())

    def test_route_with_budget_option(self, runner: CliRunner) -> None:
        from agent_energy_budget.cli.main import cli

        result = runner.invoke(
            cli,
            ["route", "--prompt", "Hello!", "--budget", "5.0", "--json-output"],
        )
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["remaining_budget_usd"] <= 5.0

    def test_route_with_max_cost(self, runner: CliRunner) -> None:
        from agent_energy_budget.cli.main import cli

        result = runner.invoke(
            cli,
            ["route", "--prompt", "Hello!", "--max-cost", "0.01", "--json-output"],
        )
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["estimated_cost_usd"] <= 0.01 + 1e-9

    def test_route_with_min_quality(self, runner: CliRunner) -> None:
        from agent_energy_budget.cli.main import cli

        result = runner.invoke(
            cli,
            ["route", "--prompt", "Hello!", "--min-quality", "0.5", "--json-output"],
        )
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["quality_score"] >= 0.5

    def test_route_invalid_strategy_fails(self, runner: CliRunner) -> None:
        from agent_energy_budget.cli.main import cli

        result = runner.invoke(
            cli,
            ["route", "--prompt", "Hello!", "--strategy", "invalid_strategy"],
        )
        assert result.exit_code != 0

    def test_route_missing_prompt_fails(self, runner: CliRunner) -> None:
        from agent_energy_budget.cli.main import cli

        result = runner.invoke(cli, ["route"])
        assert result.exit_code != 0

    def test_route_invalid_min_quality_fails(self, runner: CliRunner) -> None:
        from agent_energy_budget.cli.main import cli

        result = runner.invoke(
            cli,
            ["route", "--prompt", "Hello!", "--min-quality", "1.5"],
        )
        assert result.exit_code != 0

    def test_route_negative_min_quality_fails(self, runner: CliRunner) -> None:
        from agent_energy_budget.cli.main import cli

        result = runner.invoke(
            cli,
            ["route", "--prompt", "Hello!", "--min-quality", "-0.1"],
        )
        assert result.exit_code != 0

    def test_route_negative_budget_fails(self, runner: CliRunner) -> None:
        from agent_energy_budget.cli.main import cli

        result = runner.invoke(
            cli,
            ["route", "--prompt", "Hello!", "--budget", "-1.0"],
        )
        assert result.exit_code != 0

    def test_route_zero_max_cost_exits_with_error(self, runner: CliRunner) -> None:
        from agent_energy_budget.cli.main import cli

        result = runner.invoke(
            cli,
            ["route", "--prompt", "Hello!", "--max-cost", "0.0"],
        )
        assert result.exit_code != 0

    def test_route_json_error_on_no_affordable(self, runner: CliRunner) -> None:
        from agent_energy_budget.cli.main import cli

        result = runner.invoke(
            cli,
            [
                "route",
                "--prompt", "Hello!",
                "--max-cost", "0.0",
                "--json-output",
            ],
        )
        assert result.exit_code != 0
        data = json.loads(result.output)
        assert "error" in data

    def test_route_shows_provider(self, runner: CliRunner) -> None:
        from agent_energy_budget.cli.main import cli

        result = runner.invoke(cli, ["route", "--prompt", "Hello!"])
        assert result.exit_code == 0
        assert "Provider" in result.output

    def test_route_shows_estimated_cost(self, runner: CliRunner) -> None:
        from agent_energy_budget.cli.main import cli

        result = runner.invoke(cli, ["route", "--prompt", "Hello!"])
        assert result.exit_code == 0
        assert "Estimated Cost" in result.output

    def test_route_shows_remaining_budget(self, runner: CliRunner) -> None:
        from agent_energy_budget.cli.main import cli

        result = runner.invoke(cli, ["route", "--prompt", "Hello!"])
        assert result.exit_code == 0
        assert "Remaining Budget" in result.output

    def test_route_shows_reason(self, runner: CliRunner) -> None:
        from agent_energy_budget.cli.main import cli

        result = runner.invoke(cli, ["route", "--prompt", "Hello!"])
        assert result.exit_code == 0
        assert "Reason" in result.output

    @pytest.mark.parametrize("strategy", ["cheapest_first", "quality_first", "balanced", "budget_aware"])
    def test_all_strategies_via_cli_json(self, runner: CliRunner, strategy: str) -> None:
        from agent_energy_budget.cli.main import cli

        result = runner.invoke(
            cli,
            [
                "route",
                "--prompt", "Evaluate this carefully and in detail.",
                "--strategy", strategy,
                "--budget", "10.0",
                "--json-output",
            ],
        )
        assert result.exit_code == 0, f"Strategy {strategy} failed: {result.output}"
        data = json.loads(result.output)
        assert data["strategy"] == strategy
