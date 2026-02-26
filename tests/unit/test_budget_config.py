"""Unit tests for agent_energy_budget.budget.config."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from agent_energy_budget.budget.config import (
    AlertThresholds,
    BudgetConfig,
    DegradationStrategy,
    ModelPreferences,
)


# ---------------------------------------------------------------------------
# AlertThresholds
# ---------------------------------------------------------------------------


class TestAlertThresholds:
    def test_defaults_are_valid(self) -> None:
        thresholds = AlertThresholds()
        assert thresholds.warning == 50.0
        assert thresholds.critical == 80.0
        assert thresholds.exhausted == 100.0

    def test_custom_valid_ascending_thresholds(self) -> None:
        thresholds = AlertThresholds(warning=30.0, critical=60.0, exhausted=90.0)
        assert thresholds.warning == 30.0
        assert thresholds.critical == 60.0
        assert thresholds.exhausted == 90.0

    def test_equal_thresholds_are_valid(self) -> None:
        # warning == critical == exhausted is allowed
        thresholds = AlertThresholds(warning=50.0, critical=50.0, exhausted=50.0)
        assert thresholds.warning == 50.0

    def test_descending_thresholds_raises(self) -> None:
        with pytest.raises(ValidationError, match="ascending"):
            AlertThresholds(warning=80.0, critical=50.0, exhausted=100.0)

    def test_value_below_zero_raises(self) -> None:
        with pytest.raises(ValidationError):
            AlertThresholds(warning=-1.0)

    def test_value_above_hundred_raises(self) -> None:
        with pytest.raises(ValidationError):
            AlertThresholds(exhausted=101.0)


# ---------------------------------------------------------------------------
# ModelPreferences
# ---------------------------------------------------------------------------


class TestModelPreferences:
    def test_defaults(self) -> None:
        prefs = ModelPreferences()
        assert prefs.preferred_models == []
        assert prefs.fallback_model == "gpt-4o-mini"
        assert prefs.blocked_models == []
        assert prefs.require_vision is False

    def test_custom_values(self) -> None:
        prefs = ModelPreferences(
            preferred_models=["claude-opus-4", "gpt-4o"],
            fallback_model="gpt-4o-mini",
            blocked_models=["very-expensive-model"],
            require_vision=True,
        )
        assert "claude-opus-4" in prefs.preferred_models
        assert prefs.require_vision is True


# ---------------------------------------------------------------------------
# BudgetConfig
# ---------------------------------------------------------------------------


class TestBudgetConfig:
    def test_minimal_valid_config(self) -> None:
        config = BudgetConfig(agent_id="my-agent")
        assert config.agent_id == "my-agent"
        assert config.daily_limit == 0.0

    def test_agent_id_with_spaces_raises(self) -> None:
        with pytest.raises(ValidationError, match="spaces"):
            BudgetConfig(agent_id="my agent")

    def test_agent_id_empty_raises(self) -> None:
        with pytest.raises(ValidationError):
            BudgetConfig(agent_id="")

    def test_currency_uppercased_automatically(self) -> None:
        config = BudgetConfig(agent_id="agent", currency="usd")
        assert config.currency == "USD"

    def test_negative_daily_limit_raises(self) -> None:
        with pytest.raises(ValidationError):
            BudgetConfig(agent_id="agent", daily_limit=-1.0)

    def test_default_degradation_strategy(self) -> None:
        config = BudgetConfig(agent_id="agent")
        assert config.degradation_strategy == DegradationStrategy.TOKEN_REDUCTION

    def test_custom_degradation_strategy(self) -> None:
        config = BudgetConfig(
            agent_id="agent",
            degradation_strategy=DegradationStrategy.BLOCK_WITH_ERROR,
        )
        assert config.degradation_strategy == DegradationStrategy.BLOCK_WITH_ERROR

    def test_tags_stored_correctly(self) -> None:
        config = BudgetConfig(agent_id="agent", tags={"env": "prod", "team": "ml"})
        assert config.tags["env"] == "prod"

    def test_active_limit_returns_tightest_limit(self) -> None:
        config = BudgetConfig(
            agent_id="agent",
            daily_limit=1.0,
            weekly_limit=5.0,
            monthly_limit=20.0,
        )
        assert config.active_limit() == 1.0

    def test_active_limit_returns_none_when_all_zero(self) -> None:
        config = BudgetConfig(agent_id="agent")
        assert config.active_limit() is None

    def test_active_limit_with_only_monthly_set(self) -> None:
        config = BudgetConfig(agent_id="agent", monthly_limit=100.0)
        assert config.active_limit() == 100.0

    def test_active_limit_skips_zero_values(self) -> None:
        config = BudgetConfig(
            agent_id="agent",
            daily_limit=0.0,
            weekly_limit=10.0,
            monthly_limit=40.0,
        )
        assert config.active_limit() == 10.0

    def test_agent_id_with_hyphens_is_valid(self) -> None:
        config = BudgetConfig(agent_id="my-cool-agent-123")
        assert config.agent_id == "my-cool-agent-123"

    def test_agent_id_with_underscores_is_valid(self) -> None:
        config = BudgetConfig(agent_id="my_agent")
        assert config.agent_id == "my_agent"

    def test_all_degradation_strategy_values_valid(self) -> None:
        for strategy in DegradationStrategy:
            config = BudgetConfig(agent_id="agent", degradation_strategy=strategy)
            assert config.degradation_strategy == strategy
