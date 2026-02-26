"""Model downgrade degradation strategy.

Searches MODEL_TIERS from cheapest tier upward and selects the most
capable (highest input price) model whose cost for the requested token
counts fits within the remaining budget.
"""
from __future__ import annotations

from agent_energy_budget.degradation.base import DegradationResult, DegradationStrategyBase
from agent_energy_budget.pricing.tables import (
    PROVIDER_PRICING,
    ModelPricing,
    ModelTier,
    TIER_ORDER,
)


class ModelDowngradeStrategy(DegradationStrategyBase):
    """Find the cheapest model that can afford the requested call.

    Preference is for the highest-quality (most expensive per-token)
    model that still fits within the remaining budget — i.e., the strategy
    degrades as little as possible.

    Parameters
    ----------
    custom_pricing:
        Optional additional pricing entries to include in model search.
    """

    def __init__(
        self,
        custom_pricing: dict[str, ModelPricing] | None = None,
    ) -> None:
        self._custom_pricing = custom_pricing or {}

    def apply(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        remaining_budget: float,
    ) -> DegradationResult:
        """Select the best model within budget.

        Parameters
        ----------
        model:
            Originally requested model.
        input_tokens:
            Expected input token count.
        output_tokens:
            Expected output token count.
        remaining_budget:
            Available USD for this call.

        Returns
        -------
        DegradationResult
            Recommended model and proceed/block decision.
        """
        combined = {**PROVIDER_PRICING, **self._custom_pricing}

        best: ModelPricing | None = None

        # Iterate cheapest tier first — keep upgrading as long as budget allows
        for tier in TIER_ORDER:
            tier_candidates = [p for p in combined.values() if p.tier == tier]
            # Sort cheapest first within tier
            tier_candidates.sort(key=lambda p: p.input_per_million + p.output_per_million)

            for candidate in tier_candidates:
                cost = candidate.cost_for_tokens(input_tokens, output_tokens)
                if cost <= remaining_budget:
                    # This fits — check if it's better than current best
                    if best is None or candidate.input_per_million > best.input_per_million:
                        best = candidate

        if best is None:
            return DegradationResult(
                can_proceed=False,
                recommended_model=model,
                max_tokens=0,
                action="block",
                message=(
                    f"No model found within budget ${remaining_budget:.6f} for "
                    f"{input_tokens} input + {output_tokens} output tokens."
                ),
            )

        downgrade_cost = best.cost_for_tokens(input_tokens, output_tokens)
        if best.model == model:
            return DegradationResult(
                can_proceed=True,
                recommended_model=model,
                max_tokens=output_tokens,
                action="proceed",
                message=f"Requested model '{model}' fits within budget.",
            )

        return DegradationResult(
            can_proceed=True,
            recommended_model=best.model,
            max_tokens=output_tokens,
            action="model_downgrade",
            message=(
                f"Downgraded from '{model}' to '{best.model}' "
                f"(${downgrade_cost:.6f}) to fit ${remaining_budget:.6f} budget."
            ),
        )

    def models_within_budget(
        self,
        input_tokens: int,
        output_tokens: int,
        remaining_budget: float,
    ) -> list[ModelPricing]:
        """Return all models affordable within the given budget, best first.

        Parameters
        ----------
        input_tokens:
            Input token count.
        output_tokens:
            Output token count.
        remaining_budget:
            Available USD.

        Returns
        -------
        list[ModelPricing]
            Affordable models sorted by quality (input price) descending.
        """
        combined = {**PROVIDER_PRICING, **self._custom_pricing}
        affordable: list[ModelPricing] = [
            p
            for p in combined.values()
            if p.cost_for_tokens(input_tokens, output_tokens) <= remaining_budget
        ]
        return sorted(affordable, key=lambda p: p.input_per_million, reverse=True)
