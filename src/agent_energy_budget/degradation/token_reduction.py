"""Token reduction degradation strategy.

Reduces the requested output token count to the maximum that fits within
the remaining budget, keeping the original model unchanged.
"""
from __future__ import annotations

from agent_energy_budget.degradation.base import DegradationResult, DegradationStrategyBase
from agent_energy_budget.pricing.tables import PROVIDER_PRICING, ModelPricing, get_pricing


class TokenReductionStrategy(DegradationStrategyBase):
    """Reduce max output tokens so the call fits within remaining budget.

    Parameters
    ----------
    absolute_minimum_tokens:
        The smallest acceptable output token count. If even this does not
        fit, the call is blocked. Defaults to 50.
    custom_pricing:
        Optional additional pricing entries.
    """

    def __init__(
        self,
        absolute_minimum_tokens: int = 50,
        custom_pricing: dict[str, ModelPricing] | None = None,
    ) -> None:
        if absolute_minimum_tokens < 1:
            raise ValueError("absolute_minimum_tokens must be >= 1")
        self._min_tokens = absolute_minimum_tokens
        self._custom_pricing = custom_pricing or {}

    def apply(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        remaining_budget: float,
    ) -> DegradationResult:
        """Compute the maximum affordable output token count.

        Parameters
        ----------
        model:
            The model to use (unchanged by this strategy).
        input_tokens:
            Expected input tokens.
        output_tokens:
            Originally requested output tokens.
        remaining_budget:
            Available USD.

        Returns
        -------
        DegradationResult
            Proceed with reduced tokens, or block if even minimum doesn't fit.
        """
        try:
            pricing = self._custom_pricing.get(model) or get_pricing(model)
        except KeyError:
            # Unknown model pricing — cannot reduce safely; block
            return DegradationResult(
                can_proceed=False,
                recommended_model=model,
                max_tokens=0,
                action="block",
                message=f"Unknown model '{model}'; cannot compute token reduction.",
            )

        # Check if original request fits
        original_cost = pricing.cost_for_tokens(input_tokens, output_tokens)
        if original_cost <= remaining_budget:
            return DegradationResult(
                can_proceed=True,
                recommended_model=model,
                max_tokens=output_tokens,
                action="proceed",
                message="Original token count fits within budget.",
            )

        # Calculate max affordable output tokens
        max_output = pricing.max_output_for_budget(remaining_budget, input_tokens)

        if max_output < self._min_tokens:
            return DegradationResult(
                can_proceed=False,
                recommended_model=model,
                max_tokens=0,
                action="block",
                message=(
                    f"Maximum affordable output ({max_output} tokens) is below "
                    f"minimum threshold ({self._min_tokens} tokens). "
                    f"Remaining budget: ${remaining_budget:.6f}."
                ),
            )

        reduced_cost = pricing.cost_for_tokens(input_tokens, max_output)
        reduction_pct = round((1 - max_output / max(output_tokens, 1)) * 100, 1)

        return DegradationResult(
            can_proceed=True,
            recommended_model=model,
            max_tokens=max_output,
            action="reduce_tokens",
            message=(
                f"Output reduced {reduction_pct}% from {output_tokens} to "
                f"{max_output} tokens (${reduced_cost:.6f}) to fit "
                f"${remaining_budget:.6f} budget."
            ),
        )

    def calculate_max_tokens(
        self, model: str, input_tokens: int, remaining_budget: float
    ) -> int:
        """Calculate max output tokens for a given budget without applying strategy.

        Parameters
        ----------
        model:
            Model identifier.
        input_tokens:
            Input token count.
        remaining_budget:
            Available USD.

        Returns
        -------
        int
            Maximum affordable output tokens.
        """
        try:
            pricing = self._custom_pricing.get(model) or get_pricing(model)
        except KeyError:
            return 0
        return pricing.max_output_for_budget(remaining_budget, input_tokens)
