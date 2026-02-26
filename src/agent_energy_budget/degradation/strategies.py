"""Unified degradation strategy facade.

Provides a high-level DegradationManager that wraps the individual
strategy implementations and exposes a single :meth:`check` method.
Also re-exports the strategy enum and action dataclass for callers
that prefer the simplified API over the full strategy class hierarchy.

Strategy tiers
--------------
premium  — top models (claude-sonnet-4, gpt-4o, gemini-2.5-pro)
standard — mainstream models (claude-haiku-4, gpt-4o-mini, gemini-2.0-flash)
efficient — good-value models (mistral-large, llama-3.3-70b)
nano     — ultra-cheap models (deepseek-v3, mistral-small)
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

from agent_energy_budget.budget.tracker import BudgetStatus
from agent_energy_budget.pricing.tables import (
    PROVIDER_PRICING,
    ModelTier,
    TIER_ORDER,
    get_pricing,
)


class DegradationStrategy(str, Enum):
    """High-level strategy selection for the DegradationManager.

    NONE           — no degradation; always proceed unchanged.
    MODEL_DOWNGRADE — switch to a cheaper model when budget is low.
    TOKEN_LIMIT    — reduce max output tokens proportional to remaining budget.
    RATE_LIMIT     — apply a calls-per-minute cap proportional to remaining budget.
    SUSPEND        — block all calls when budget is exhausted.
    """

    NONE = "none"
    MODEL_DOWNGRADE = "model_downgrade"
    TOKEN_LIMIT = "token_limit"
    RATE_LIMIT = "rate_limit"
    SUSPEND = "suspend"


@dataclass(frozen=True)
class DegradationAction:
    """Recommended action returned by DegradationManager.check().

    Parameters
    ----------
    strategy:
        The strategy that produced this action.
    original_model:
        Model that was originally requested.
    degraded_model:
        Model that should be used (may equal original_model if no downgrade).
    token_limit:
        Maximum output tokens to request, or None if no limit was applied.
    rate_limit_per_minute:
        Maximum calls per minute, or None if no rate limit was applied.
    reason:
        Human-readable explanation of the action taken.
    """

    strategy: DegradationStrategy
    original_model: str
    degraded_model: str
    token_limit: Optional[int]
    rate_limit_per_minute: Optional[int]
    reason: str


# ---------------------------------------------------------------------------
# Tier model mappings for the simplified DegradationManager
# ---------------------------------------------------------------------------

# Ordered from highest quality to lowest quality within each conceptual tier
MODEL_TIERS: dict[str, list[str]] = {
    "premium": ["claude-sonnet-4", "gpt-4o", "gemini-2.5-pro"],
    "standard": ["claude-haiku-4", "gpt-4o-mini", "gemini-2.0-flash"],
    "efficient": ["mistral-large", "llama-3.3-70b"],
    "nano": ["deepseek-v3", "mistral-small"],
}

# Tier names ordered from highest to lowest quality
_TIER_QUALITY_ORDER: list[str] = ["premium", "standard", "efficient", "nano"]

# Default rate limit at 100% budget (calls per minute)
_DEFAULT_FULL_RATE_LIMIT: int = 60
# Default max output tokens at 100% budget
_DEFAULT_FULL_TOKEN_LIMIT: int = 4096


def _tier_for_model(model: str) -> str | None:
    """Return the MODEL_TIERS key for *model*, or None if not found."""
    normalised = model.strip().lower()
    for tier_name, models in MODEL_TIERS.items():
        for tier_model in models:
            if tier_model.lower() == normalised or normalised in tier_model.lower():
                return tier_name
    # Fall back to pricing tables tier classification
    try:
        pricing = get_pricing(model)
        tier_map: dict[ModelTier, str] = {
            ModelTier.PREMIUM: "premium",
            ModelTier.STANDARD: "standard",
            ModelTier.EFFICIENT: "efficient",
            ModelTier.NANO: "nano",
        }
        return tier_map.get(pricing.tier)
    except KeyError:
        return None


def _next_lower_tier_model(model: str) -> str | None:
    """Return the first model in the next lower quality tier, or None."""
    current_tier = _tier_for_model(model)
    if current_tier is None:
        # Unknown model — fall straight to the cheapest known tier
        for tier_name in reversed(_TIER_QUALITY_ORDER):
            if MODEL_TIERS[tier_name]:
                return MODEL_TIERS[tier_name][0]
        return None

    current_index = _TIER_QUALITY_ORDER.index(current_tier)
    # Iterate towards lower quality tiers
    for tier_name in _TIER_QUALITY_ORDER[current_index + 1:]:
        if MODEL_TIERS[tier_name]:
            return MODEL_TIERS[tier_name][0]
    return None


class DegradationManager:
    """Facade that applies a single degradation strategy.

    Parameters
    ----------
    strategy:
        The :class:`DegradationStrategy` to apply. Defaults to
        ``MODEL_DOWNGRADE``.

    Examples
    --------
    >>> manager = DegradationManager()
    >>> action = manager.check(budget_status, current_model="gpt-4o")
    >>> action.degraded_model
    'claude-haiku-4'
    """

    # When utilisation exceeds this threshold, degradation is activated
    _DEFAULT_WARNING_THRESHOLD: float = 80.0
    _DEFAULT_CRITICAL_THRESHOLD: float = 95.0

    def __init__(
        self,
        strategy: DegradationStrategy = DegradationStrategy.MODEL_DOWNGRADE,
    ) -> None:
        self._strategy = strategy
        self._warning_threshold: float = self._DEFAULT_WARNING_THRESHOLD
        self._critical_threshold: float = self._DEFAULT_CRITICAL_THRESHOLD

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def configure(
        self,
        strategy: DegradationStrategy,
        thresholds: Optional[dict[str, float]] = None,
    ) -> None:
        """Update the active strategy and optional threshold overrides.

        Parameters
        ----------
        strategy:
            New degradation strategy to apply.
        thresholds:
            Optional dict with keys ``"warning"`` and/or ``"critical"``
            mapping to utilisation percentage thresholds (0–100).
        """
        self._strategy = strategy
        if thresholds:
            if "warning" in thresholds:
                self._warning_threshold = float(thresholds["warning"])
            if "critical" in thresholds:
                self._critical_threshold = float(thresholds["critical"])

    # ------------------------------------------------------------------
    # Main check
    # ------------------------------------------------------------------

    def check(
        self,
        budget_status: BudgetStatus,
        current_model: str,
    ) -> DegradationAction:
        """Evaluate the budget status and return the recommended action.

        Parameters
        ----------
        budget_status:
            Current snapshot of budget utilisation from BudgetTracker.
        current_model:
            The model identifier about to be used.

        Returns
        -------
        DegradationAction
            Recommended action, potentially with a degraded model or limits.
        """
        utilisation = budget_status.utilisation_pct
        is_exhausted = (
            budget_status.remaining_usd <= 0
            or budget_status.limit_usd <= 0
            and utilisation >= 100.0
        )

        # NONE strategy — always proceed unchanged
        if self._strategy == DegradationStrategy.NONE:
            return DegradationAction(
                strategy=self._strategy,
                original_model=current_model,
                degraded_model=current_model,
                token_limit=None,
                rate_limit_per_minute=None,
                reason="No degradation strategy active; proceeding normally.",
            )

        # Below warning threshold — no action needed
        if utilisation < self._warning_threshold and not is_exhausted:
            return DegradationAction(
                strategy=self._strategy,
                original_model=current_model,
                degraded_model=current_model,
                token_limit=None,
                rate_limit_per_minute=None,
                reason=(
                    f"Budget utilisation {utilisation:.1f}% is below "
                    f"warning threshold {self._warning_threshold:.1f}%."
                ),
            )

        if self._strategy == DegradationStrategy.SUSPEND:
            return self._apply_suspend(
                budget_status, current_model, is_exhausted
            )

        if self._strategy == DegradationStrategy.MODEL_DOWNGRADE:
            return self._apply_model_downgrade(
                budget_status, current_model, is_exhausted
            )

        if self._strategy == DegradationStrategy.TOKEN_LIMIT:
            return self._apply_token_limit(
                budget_status, current_model, is_exhausted
            )

        if self._strategy == DegradationStrategy.RATE_LIMIT:
            return self._apply_rate_limit(
                budget_status, current_model, is_exhausted
            )

        # Fallback — suspend when budget is exhausted, else no action
        if is_exhausted:
            return DegradationAction(
                strategy=self._strategy,
                original_model=current_model,
                degraded_model=current_model,
                token_limit=0,
                rate_limit_per_minute=0,
                reason="Budget exhausted; all calls suspended.",
            )
        return DegradationAction(
            strategy=self._strategy,
            original_model=current_model,
            degraded_model=current_model,
            token_limit=None,
            rate_limit_per_minute=None,
            reason="Unknown strategy; proceeding without degradation.",
        )

    # ------------------------------------------------------------------
    # Strategy implementations
    # ------------------------------------------------------------------

    def _apply_suspend(
        self,
        budget_status: BudgetStatus,
        current_model: str,
        is_exhausted: bool,
    ) -> DegradationAction:
        """Return suspend action when budget is exhausted."""
        if is_exhausted:
            return DegradationAction(
                strategy=DegradationStrategy.SUSPEND,
                original_model=current_model,
                degraded_model=current_model,
                token_limit=0,
                rate_limit_per_minute=0,
                reason=(
                    f"Budget for agent '{budget_status.agent_id}' is exhausted "
                    f"(spent ${budget_status.spent_usd:.4f} of "
                    f"${budget_status.limit_usd:.4f}). All calls suspended."
                ),
            )
        # At or above warning but not yet exhausted — warn but allow
        return DegradationAction(
            strategy=DegradationStrategy.SUSPEND,
            original_model=current_model,
            degraded_model=current_model,
            token_limit=None,
            rate_limit_per_minute=None,
            reason=(
                f"Budget utilisation at {budget_status.utilisation_pct:.1f}%. "
                "Approaching suspension threshold."
            ),
        )

    def _apply_model_downgrade(
        self,
        budget_status: BudgetStatus,
        current_model: str,
        is_exhausted: bool,
    ) -> DegradationAction:
        """Downgrade to the next cheaper model tier."""
        if is_exhausted:
            return DegradationAction(
                strategy=DegradationStrategy.MODEL_DOWNGRADE,
                original_model=current_model,
                degraded_model=current_model,
                token_limit=0,
                rate_limit_per_minute=None,
                reason=(
                    f"Budget exhausted for agent '{budget_status.agent_id}'. "
                    "Cannot proceed even with cheapest model."
                ),
            )

        degraded = _next_lower_tier_model(current_model)
        if degraded is None or degraded == current_model:
            # Already at cheapest tier or unknown model — proceed as-is
            return DegradationAction(
                strategy=DegradationStrategy.MODEL_DOWNGRADE,
                original_model=current_model,
                degraded_model=current_model,
                token_limit=None,
                rate_limit_per_minute=None,
                reason=(
                    f"Model '{current_model}' is already at the cheapest available tier."
                ),
            )

        return DegradationAction(
            strategy=DegradationStrategy.MODEL_DOWNGRADE,
            original_model=current_model,
            degraded_model=degraded,
            token_limit=None,
            rate_limit_per_minute=None,
            reason=(
                f"Budget at {budget_status.utilisation_pct:.1f}% utilisation; "
                f"downgraded from '{current_model}' to '{degraded}'."
            ),
        )

    def _apply_token_limit(
        self,
        budget_status: BudgetStatus,
        current_model: str,
        is_exhausted: bool,
    ) -> DegradationAction:
        """Reduce max output tokens proportional to remaining budget."""
        if is_exhausted:
            return DegradationAction(
                strategy=DegradationStrategy.TOKEN_LIMIT,
                original_model=current_model,
                degraded_model=current_model,
                token_limit=0,
                rate_limit_per_minute=None,
                reason=(
                    f"Budget exhausted for agent '{budget_status.agent_id}'. "
                    "Token limit set to 0; call blocked."
                ),
            )

        # Scale token limit linearly: remaining_fraction * default_limit
        if budget_status.limit_usd > 0:
            remaining_fraction = max(
                0.0,
                budget_status.remaining_usd / budget_status.limit_usd,
            )
        else:
            remaining_fraction = 1.0

        token_limit = max(1, round(_DEFAULT_FULL_TOKEN_LIMIT * remaining_fraction))

        return DegradationAction(
            strategy=DegradationStrategy.TOKEN_LIMIT,
            original_model=current_model,
            degraded_model=current_model,
            token_limit=token_limit,
            rate_limit_per_minute=None,
            reason=(
                f"Budget at {budget_status.utilisation_pct:.1f}% utilisation; "
                f"max output tokens reduced to {token_limit} "
                f"({remaining_fraction * 100:.1f}% of {_DEFAULT_FULL_TOKEN_LIMIT})."
            ),
        )

    def _apply_rate_limit(
        self,
        budget_status: BudgetStatus,
        current_model: str,
        is_exhausted: bool,
    ) -> DegradationAction:
        """Reduce calls-per-minute proportional to remaining budget."""
        if is_exhausted:
            return DegradationAction(
                strategy=DegradationStrategy.RATE_LIMIT,
                original_model=current_model,
                degraded_model=current_model,
                token_limit=None,
                rate_limit_per_minute=0,
                reason=(
                    f"Budget exhausted for agent '{budget_status.agent_id}'. "
                    "Rate limit set to 0; call blocked."
                ),
            )

        if budget_status.limit_usd > 0:
            remaining_fraction = max(
                0.0,
                budget_status.remaining_usd / budget_status.limit_usd,
            )
        else:
            remaining_fraction = 1.0

        rate_limit = max(1, round(_DEFAULT_FULL_RATE_LIMIT * remaining_fraction))

        return DegradationAction(
            strategy=DegradationStrategy.RATE_LIMIT,
            original_model=current_model,
            degraded_model=current_model,
            token_limit=None,
            rate_limit_per_minute=rate_limit,
            reason=(
                f"Budget at {budget_status.utilisation_pct:.1f}% utilisation; "
                f"rate limited to {rate_limit} calls/min "
                f"({remaining_fraction * 100:.1f}% of {_DEFAULT_FULL_RATE_LIMIT})."
            ),
        )
