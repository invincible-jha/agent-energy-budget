"""Pydantic v2 configuration schema for budget tracking.

BudgetConfig is validated at system boundaries (CLI, SDK init, YAML load).
All fields have sensible defaults so minimal configuration is required.
"""
from __future__ import annotations

from enum import Enum
from typing import Annotated

from pydantic import BaseModel, Field, field_validator, model_validator


class DegradationStrategy(str, Enum):
    """Action to take when a budget limit is approached or exceeded.

    MODEL_DOWNGRADE   — switch to a cheaper model automatically.
    TOKEN_REDUCTION   — reduce max_output_tokens to fit the budget.
    BLOCK_WITH_ERROR  — raise BudgetExceededError immediately.
    CACHED_FALLBACK   — return a cached previous response if available.
    """

    MODEL_DOWNGRADE = "model_downgrade"
    TOKEN_REDUCTION = "token_reduction"
    BLOCK_WITH_ERROR = "block_with_error"
    CACHED_FALLBACK = "cached_fallback"


class AlertThresholds(BaseModel):
    """Utilisation percentages at which budget alerts are fired.

    Parameters
    ----------
    warning:
        First alert threshold (default 50 %).
    critical:
        Second alert threshold (default 80 %).
    exhausted:
        Third alert threshold (default 100 %).
    """

    warning: Annotated[float, Field(ge=0.0, le=100.0)] = 50.0
    critical: Annotated[float, Field(ge=0.0, le=100.0)] = 80.0
    exhausted: Annotated[float, Field(ge=0.0, le=100.0)] = 100.0

    @model_validator(mode="after")
    def thresholds_must_be_ascending(self) -> "AlertThresholds":
        if not (self.warning <= self.critical <= self.exhausted):
            raise ValueError(
                "Alert thresholds must be ascending: warning <= critical <= exhausted. "
                f"Got warning={self.warning}, critical={self.critical}, "
                f"exhausted={self.exhausted}."
            )
        return self


class ModelPreferences(BaseModel):
    """Model selection preferences for degradation and estimation.

    Parameters
    ----------
    preferred_models:
        Ordered list of model IDs to try first (most preferred first).
    fallback_model:
        Model to use when all others are exhausted or over budget.
    blocked_models:
        Model IDs that must never be used (e.g., too expensive).
    require_vision:
        When True, restrict model selection to vision-capable models only.
    """

    preferred_models: list[str] = Field(default_factory=list)
    fallback_model: str = "gpt-4o-mini"
    blocked_models: list[str] = Field(default_factory=list)
    require_vision: bool = False


class BudgetConfig(BaseModel):
    """Top-level budget configuration for a single agent or agent group.

    Parameters
    ----------
    agent_id:
        Unique identifier for the agent this budget belongs to.
    daily_limit:
        Maximum USD spend per calendar day. 0 = disabled.
    weekly_limit:
        Maximum USD spend per calendar week (Mon–Sun). 0 = disabled.
    monthly_limit:
        Maximum USD spend per calendar month. 0 = disabled.
    degradation_strategy:
        Strategy to apply when a limit is breached.
    alert_thresholds:
        Utilisation % levels that trigger alerts.
    model_preferences:
        Model selection configuration for degradation.
    currency:
        ISO currency code for display purposes (does not affect math).
    tags:
        Arbitrary key/value metadata attached to this budget.
    """

    agent_id: str = Field(min_length=1)
    daily_limit: Annotated[float, Field(ge=0.0)] = 0.0
    weekly_limit: Annotated[float, Field(ge=0.0)] = 0.0
    monthly_limit: Annotated[float, Field(ge=0.0)] = 0.0
    degradation_strategy: DegradationStrategy = DegradationStrategy.TOKEN_REDUCTION
    alert_thresholds: AlertThresholds = Field(default_factory=AlertThresholds)
    model_preferences: ModelPreferences = Field(default_factory=ModelPreferences)
    currency: str = "USD"
    tags: dict[str, str] = Field(default_factory=dict)

    @field_validator("agent_id")
    @classmethod
    def agent_id_no_spaces(cls, value: str) -> str:
        if " " in value:
            raise ValueError(
                f"agent_id must not contain spaces; got {value!r}. "
                "Use hyphens or underscores as word separators."
            )
        return value

    @field_validator("currency")
    @classmethod
    def currency_uppercase(cls, value: str) -> str:
        return value.upper()

    @model_validator(mode="after")
    def at_least_one_limit(self) -> "BudgetConfig":
        if self.daily_limit == 0.0 and self.weekly_limit == 0.0 and self.monthly_limit == 0.0:
            # Not an error — unlimited budgets are valid; just log a note.
            pass
        return self

    def active_limit(self) -> float | None:
        """Return the tightest active limit in USD, or None if all disabled.

        Returns the smallest non-zero limit across daily, weekly, and
        monthly values since the daily limit is the most restrictive.

        Returns
        -------
        float | None
            Tightest limit in USD, or None if no limits are set.
        """
        candidates = [
            v for v in (self.daily_limit, self.weekly_limit, self.monthly_limit) if v > 0.0
        ]
        return min(candidates) if candidates else None
