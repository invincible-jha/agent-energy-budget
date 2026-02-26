"""Block-with-error degradation strategy.

Returns an informative error result when a budget would be exceeded.
The caller is responsible for raising an exception if desired; this
strategy returns a DegradationResult with can_proceed=False.
"""
from __future__ import annotations

from agent_energy_budget.degradation.base import DegradationResult, DegradationStrategyBase
from agent_energy_budget.pricing.tables import get_pricing


class BlockStrategy(DegradationStrategyBase):
    """Immediately block the call when budget would be exceeded.

    Parameters
    ----------
    include_pricing_detail:
        When True, the error message includes pricing and budget figures.
    """

    def __init__(self, include_pricing_detail: bool = True) -> None:
        self._include_detail = include_pricing_detail

    def apply(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        remaining_budget: float,
    ) -> DegradationResult:
        """Block the call and return an informative result.

        Parameters
        ----------
        model:
            Requested model.
        input_tokens:
            Expected input tokens.
        output_tokens:
            Expected output tokens.
        remaining_budget:
            Available USD.

        Returns
        -------
        DegradationResult
            Always has can_proceed=False.
        """
        if not self._include_detail:
            return DegradationResult(
                can_proceed=False,
                recommended_model=model,
                max_tokens=0,
                action="block",
                message="Budget exceeded.",
            )

        try:
            pricing = get_pricing(model)
            estimated_cost = pricing.cost_for_tokens(input_tokens, output_tokens)
            overage = estimated_cost - remaining_budget
            message = (
                f"Budget exceeded for model '{model}': "
                f"estimated cost ${estimated_cost:.6f} exceeds "
                f"remaining ${remaining_budget:.6f} by ${overage:.6f}. "
                f"Input rate: ${pricing.input_per_million}/M tokens. "
                f"Output rate: ${pricing.output_per_million}/M tokens."
            )
        except KeyError:
            message = (
                f"Budget exceeded for unknown model '{model}'. "
                f"Remaining budget: ${remaining_budget:.6f}."
            )

        return DegradationResult(
            can_proceed=False,
            recommended_model=model,
            max_tokens=0,
            action="block",
            message=message,
        )
