"""Pre-call cost estimation using pricing tables and token counting.

CostEstimator provides offline cost estimates before making LLM API calls,
allowing budget checks to run without network I/O.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Union

from agent_energy_budget.pricing.tables import (
    PROVIDER_PRICING,
    ModelPricing,
    get_pricing,
)
from agent_energy_budget.pricing.token_counter import TokenCounter


@dataclass(frozen=True)
class CostEstimate:
    """Result of a pre-call cost estimation.

    Parameters
    ----------
    model:
        Model identifier used in the estimate.
    input_tokens:
        Estimated input token count.
    output_tokens:
        Estimated output token count.
    estimated_cost_usd:
        Estimated total cost in USD.
    pricing:
        The pricing record used for this estimate.
    """

    model: str
    input_tokens: int
    output_tokens: int
    estimated_cost_usd: float
    pricing: ModelPricing


@dataclass(frozen=True)
class WorkflowStep:
    """A single step in a multi-step workflow cost estimate.

    Parameters
    ----------
    step_name:
        Human-readable name for this workflow step.
    model:
        Model to use for this step.
    input_tokens_or_text:
        Either an integer token count or raw text to be tokenised.
    output_tokens:
        Expected output token count.
    """

    step_name: str
    model: str
    input_tokens_or_text: Union[int, str]
    output_tokens: int


@dataclass(frozen=True)
class WorkflowCostEstimate:
    """Aggregated cost estimate for a multi-step workflow.

    Parameters
    ----------
    steps:
        Per-step cost estimates.
    total_cost_usd:
        Sum of all step costs.
    total_input_tokens:
        Total input tokens across all steps.
    total_output_tokens:
        Total output tokens across all steps.
    """

    steps: list[CostEstimate]
    total_cost_usd: float
    total_input_tokens: int
    total_output_tokens: int


class CostEstimator:
    """Pre-call LLM cost estimator.

    Parameters
    ----------
    default_output_tokens:
        Assumed output token count when not specified (default 512).
    token_counter:
        Optional custom TokenCounter. A new one is created if not provided.
    custom_pricing:
        Optional extra pricing entries to consider alongside PROVIDER_PRICING.
    """

    def __init__(
        self,
        default_output_tokens: int = 512,
        token_counter: TokenCounter | None = None,
        custom_pricing: dict[str, ModelPricing] | None = None,
    ) -> None:
        self._default_output_tokens = default_output_tokens
        self._counter = token_counter or TokenCounter()
        self._custom_pricing = custom_pricing or {}

    # ------------------------------------------------------------------
    # Core estimation
    # ------------------------------------------------------------------

    def _resolve_pricing(self, model: str) -> ModelPricing:
        """Look up pricing in custom table first, then standard table."""
        if model in self._custom_pricing:
            return self._custom_pricing[model]
        return get_pricing(model)

    def estimate(
        self,
        model: str,
        input_tokens_or_text: Union[int, str],
        output_tokens: int | None = None,
    ) -> CostEstimate:
        """Estimate the cost of a single LLM call.

        Parameters
        ----------
        model:
            Model identifier string.
        input_tokens_or_text:
            Either an integer token count or raw text that will be tokenised.
        output_tokens:
            Expected output token count. Uses ``default_output_tokens`` if None.

        Returns
        -------
        CostEstimate
            Estimated cost details.

        Raises
        ------
        KeyError
            If the model cannot be resolved to a pricing record.
        """
        pricing = self._resolve_pricing(model)
        input_tokens = self._counter.estimate_from_tokens_or_text(input_tokens_or_text)
        effective_output = output_tokens if output_tokens is not None else self._default_output_tokens
        cost = pricing.cost_for_tokens(input_tokens, effective_output)
        return CostEstimate(
            model=model,
            input_tokens=input_tokens,
            output_tokens=effective_output,
            estimated_cost_usd=cost,
            pricing=pricing,
        )

    def estimate_from_messages(
        self,
        model: str,
        messages: list[dict[str, str]],
        output_tokens: int | None = None,
    ) -> CostEstimate:
        """Estimate cost for a chat-format messages list.

        Parameters
        ----------
        model:
            Model identifier string.
        messages:
            List of role/content dicts (OpenAI / Anthropic message format).
        output_tokens:
            Expected output tokens. Uses ``default_output_tokens`` if None.

        Returns
        -------
        CostEstimate
            Estimated cost details.
        """
        input_tokens = self._counter.count_messages(messages)
        return self.estimate(model, input_tokens, output_tokens)

    # ------------------------------------------------------------------
    # Workflow estimation
    # ------------------------------------------------------------------

    def estimate_workflow(self, steps: list[WorkflowStep]) -> WorkflowCostEstimate:
        """Estimate the total cost of a multi-step LLM workflow.

        Parameters
        ----------
        steps:
            Ordered list of workflow steps.

        Returns
        -------
        WorkflowCostEstimate
            Aggregated costs and per-step breakdown.
        """
        step_estimates: list[CostEstimate] = []
        for step in steps:
            estimate = self.estimate(
                model=step.model,
                input_tokens_or_text=step.input_tokens_or_text,
                output_tokens=step.output_tokens,
            )
            step_estimates.append(estimate)

        total_cost = sum(e.estimated_cost_usd for e in step_estimates)
        total_input = sum(e.input_tokens for e in step_estimates)
        total_output = sum(e.output_tokens for e in step_estimates)

        return WorkflowCostEstimate(
            steps=step_estimates,
            total_cost_usd=round(total_cost, 8),
            total_input_tokens=total_input,
            total_output_tokens=total_output,
        )

    # ------------------------------------------------------------------
    # Comparison helpers
    # ------------------------------------------------------------------

    def compare_models(
        self,
        models: list[str],
        input_tokens_or_text: Union[int, str],
        output_tokens: int | None = None,
    ) -> list[CostEstimate]:
        """Estimate and compare costs across multiple models.

        Parameters
        ----------
        models:
            List of model identifiers to compare.
        input_tokens_or_text:
            Input tokens (int) or raw text.
        output_tokens:
            Expected output tokens.

        Returns
        -------
        list[CostEstimate]
            Estimates sorted cheapest first.
        """
        estimates: list[CostEstimate] = []
        for model in models:
            try:
                estimates.append(self.estimate(model, input_tokens_or_text, output_tokens))
            except KeyError:
                continue
        return sorted(estimates, key=lambda e: e.estimated_cost_usd)

    def cheapest_model_for_budget(
        self,
        budget_usd: float,
        input_tokens_or_text: Union[int, str],
        output_tokens: int | None = None,
    ) -> CostEstimate | None:
        """Return the most capable model that fits within the budget.

        Searches all known models (PROVIDER_PRICING + custom), returns the
        one with the highest per-token quality (highest input price) that
        still costs <= budget_usd.

        Parameters
        ----------
        budget_usd:
            Available budget in USD.
        input_tokens_or_text:
            Input tokens or raw text.
        output_tokens:
            Expected output tokens.

        Returns
        -------
        CostEstimate | None
            Best affordable estimate, or None if nothing fits.
        """
        combined = {**PROVIDER_PRICING, **self._custom_pricing}
        candidates: list[CostEstimate] = []

        for model_name in combined:
            try:
                estimate = self.estimate(model_name, input_tokens_or_text, output_tokens)
                if estimate.estimated_cost_usd <= budget_usd:
                    candidates.append(estimate)
            except KeyError:
                continue

        if not candidates:
            return None

        # Return the most capable (highest input rate) that still fits
        return max(candidates, key=lambda e: e.pricing.input_per_million)
