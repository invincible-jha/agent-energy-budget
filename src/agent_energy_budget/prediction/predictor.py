"""CostPredictor — pre-execution LLM call cost prediction engine.

CostPredictor combines PricingTable, TokenCounter, and OutputEstimator
to produce a PredictionResult before a single token is sent to an API.
This enables agents to make budget-aware decisions (route to a cheaper
model, reduce context, abort) without paying for the call first.

Usage
-----
::

    from agent_energy_budget.prediction import CostPredictor

    predictor = CostPredictor()
    result = predictor.predict(
        model="claude-sonnet-4",
        prompt="Write a Python quicksort implementation.",
        task_type="code_gen",
        budget_usd=0.01,
    )
    if result.will_exceed_budget:
        # route to cheaper model or truncate prompt
        ...
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional, Union

from agent_energy_budget.prediction.output_estimator import OutputEstimate, OutputEstimator, TaskType
from agent_energy_budget.prediction.pricing_table import ModelPricing, PricingTable
from agent_energy_budget.prediction.token_counter import TokenCounter

logger = logging.getLogger(__name__)


@dataclass
class PredictionResult:
    """Result of a pre-execution cost prediction.

    Parameters
    ----------
    estimated_cost_usd:
        Central cost estimate in USD.
    input_tokens:
        Estimated input token count.
    estimated_output_tokens:
        Estimated output token count.
    model:
        The model identifier used for this prediction.
    will_exceed_budget:
        True when the estimated cost exceeds the provided budget, False
        when it fits, None when no budget was specified.
    confidence:
        Overall confidence score in [0.0, 1.0] — a combination of the
        output estimation confidence and whether pricing is exact.
    low_cost_usd:
        Conservative (low-output) cost estimate.
    high_cost_usd:
        Generous (high-output) cost estimate.
    output_estimate:
        The OutputEstimate used to produce this result.
    """

    estimated_cost_usd: float
    input_tokens: int
    estimated_output_tokens: int
    model: str
    will_exceed_budget: Optional[bool] = None
    confidence: float = 0.0
    low_cost_usd: float = 0.0
    high_cost_usd: float = 0.0
    output_estimate: Optional[OutputEstimate] = None

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence must be in [0, 1], got {self.confidence}")


@dataclass
class BatchPredictionResult:
    """Aggregate result for a batch of PredictionResult items.

    Parameters
    ----------
    predictions:
        Individual prediction results.
    total_estimated_cost_usd:
        Sum of all central cost estimates.
    total_low_cost_usd:
        Sum of all low cost estimates.
    total_high_cost_usd:
        Sum of all high cost estimates.
    total_input_tokens:
        Sum of all input token counts.
    total_output_tokens:
        Sum of all estimated output tokens.
    any_will_exceed_budget:
        True if any individual result has ``will_exceed_budget=True``.
    """

    predictions: list[PredictionResult]
    total_estimated_cost_usd: float
    total_low_cost_usd: float
    total_high_cost_usd: float
    total_input_tokens: int
    total_output_tokens: int
    any_will_exceed_budget: Optional[bool] = None


class CostPredictor:
    """Pre-execution cost prediction combining pricing, token counting, and output estimation.

    Parameters
    ----------
    pricing_table:
        Pricing source. Defaults to built-in PricingTable.
    token_counter:
        Token counting backend. Defaults to TokenCounter().
    output_estimator:
        Output token heuristic. Defaults to OutputEstimator().

    Examples
    --------
    >>> predictor = CostPredictor()
    >>> result = predictor.predict(model="gpt-4o", prompt="Hello", task_type="chat")
    >>> result.estimated_cost_usd
    0.0  # (approximately)
    """

    def __init__(
        self,
        pricing_table: Optional[PricingTable] = None,
        token_counter: Optional[TokenCounter] = None,
        output_estimator: Optional[OutputEstimator] = None,
    ) -> None:
        self._pricing = pricing_table or PricingTable()
        self._counter = token_counter or TokenCounter()
        self._estimator = output_estimator or OutputEstimator()

    # ------------------------------------------------------------------
    # Core prediction
    # ------------------------------------------------------------------

    def predict(
        self,
        model: str,
        prompt: Union[str, list[dict[str, object]]],
        task_type: str | TaskType = TaskType.UNKNOWN,
        system: str = "",
        max_tokens: Optional[int] = None,
        budget_usd: Optional[float] = None,
        cached_tokens: int = 0,
    ) -> PredictionResult:
        """Predict cost for a single LLM call before it executes.

        Parameters
        ----------
        model:
            Target model identifier.
        prompt:
            Either a plain string prompt or a list of chat message dicts.
        task_type:
            Task category for output token estimation.
        system:
            Optional system prompt (included in input token count).
        max_tokens:
            Hard cap on output tokens (API parameter). Caps the estimate.
        budget_usd:
            Optional budget. When provided, sets ``will_exceed_budget``.
        cached_tokens:
            Number of input tokens served from cache (cheaper rate).

        Returns
        -------
        PredictionResult

        Raises
        ------
        KeyError
            If *model* is not found in the pricing table.
        """
        pricing = self._pricing.get_pricing(model)

        # Count input tokens
        input_tokens = self._counter.count_prompt(prompt, system=system)

        # Estimate output tokens
        output_estimate = self._estimator.estimate(
            task_type=task_type,
            input_tokens=input_tokens,
            max_tokens=max_tokens,
        )

        # Compute costs
        cost = pricing.cost_for_tokens(
            input_tokens=max(0, input_tokens - cached_tokens),
            output_tokens=output_estimate.estimated_tokens,
            cached_tokens=cached_tokens,
        )
        low_cost = pricing.cost_for_tokens(
            input_tokens=max(0, input_tokens - cached_tokens),
            output_tokens=output_estimate.low_estimate,
            cached_tokens=cached_tokens,
        )
        high_cost = pricing.cost_for_tokens(
            input_tokens=max(0, input_tokens - cached_tokens),
            output_tokens=output_estimate.high_estimate,
            cached_tokens=cached_tokens,
        )

        will_exceed = None
        if budget_usd is not None:
            will_exceed = cost > budget_usd

        return PredictionResult(
            estimated_cost_usd=cost,
            input_tokens=input_tokens,
            estimated_output_tokens=output_estimate.estimated_tokens,
            model=model,
            will_exceed_budget=will_exceed,
            confidence=output_estimate.confidence,
            low_cost_usd=low_cost,
            high_cost_usd=high_cost,
            output_estimate=output_estimate,
        )

    def predict_with_tokens(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        budget_usd: Optional[float] = None,
        cached_tokens: int = 0,
    ) -> PredictionResult:
        """Predict cost when token counts are already known.

        Parameters
        ----------
        model:
            Target model identifier.
        input_tokens:
            Known input token count.
        output_tokens:
            Known (or estimated) output token count.
        budget_usd:
            Optional budget for will_exceed_budget calculation.
        cached_tokens:
            Cached input tokens.

        Returns
        -------
        PredictionResult
        """
        pricing = self._pricing.get_pricing(model)
        cost = pricing.cost_for_tokens(
            input_tokens=max(0, input_tokens - cached_tokens),
            output_tokens=output_tokens,
            cached_tokens=cached_tokens,
        )

        will_exceed = None
        if budget_usd is not None:
            will_exceed = cost > budget_usd

        return PredictionResult(
            estimated_cost_usd=cost,
            input_tokens=input_tokens,
            estimated_output_tokens=output_tokens,
            model=model,
            will_exceed_budget=will_exceed,
            confidence=1.0,  # high confidence when tokens are known
        )

    # ------------------------------------------------------------------
    # Comparison helpers
    # ------------------------------------------------------------------

    def compare_models(
        self,
        models: list[str],
        prompt: Union[str, list[dict[str, object]]],
        task_type: str | TaskType = TaskType.UNKNOWN,
        system: str = "",
        budget_usd: Optional[float] = None,
    ) -> list[PredictionResult]:
        """Predict costs across multiple models and return sorted results.

        Parameters
        ----------
        models:
            List of model identifiers to compare.
        prompt:
            Prompt string or message list.
        task_type:
            Task type for output estimation.
        system:
            System prompt.
        budget_usd:
            Optional budget threshold.

        Returns
        -------
        list[PredictionResult]
            Results sorted cheapest first. Models not found in the
            pricing table are silently skipped.
        """
        results: list[PredictionResult] = []
        for model in models:
            try:
                result = self.predict(
                    model=model,
                    prompt=prompt,
                    task_type=task_type,
                    system=system,
                    budget_usd=budget_usd,
                )
                results.append(result)
            except KeyError:
                logger.debug("Model %r not found in pricing table, skipping", model)
                continue
        return sorted(results, key=lambda r: r.estimated_cost_usd)

    def cheapest_model_within_budget(
        self,
        models: list[str],
        prompt: Union[str, list[dict[str, object]]],
        budget_usd: float,
        task_type: str | TaskType = TaskType.UNKNOWN,
        system: str = "",
    ) -> Optional[PredictionResult]:
        """Return the most capable model whose predicted cost fits the budget.

        Parameters
        ----------
        models:
            Candidate model identifiers (in preference order — first is
            most preferred when multiple fit).
        prompt:
            Prompt string or message list.
        budget_usd:
            Maximum allowed cost in USD.
        task_type:
            Task type for output estimation.
        system:
            System prompt.

        Returns
        -------
        PredictionResult | None
            The first model in *models* that fits the budget, or None.
        """
        for model in models:
            try:
                result = self.predict(
                    model=model,
                    prompt=prompt,
                    task_type=task_type,
                    system=system,
                    budget_usd=budget_usd,
                )
                if not result.will_exceed_budget:
                    return result
            except KeyError:
                continue
        return None

    # ------------------------------------------------------------------
    # Batch prediction
    # ------------------------------------------------------------------

    def predict_batch(
        self,
        calls: list[dict[str, object]],
        budget_usd: Optional[float] = None,
    ) -> BatchPredictionResult:
        """Predict costs for a list of planned LLM calls.

        Parameters
        ----------
        calls:
            List of dicts, each containing keyword arguments for
            ``predict()`` (``model``, ``prompt``, ``task_type``, etc.).
        budget_usd:
            Optional total budget. When provided, ``any_will_exceed_budget``
            reflects whether the aggregate cost exceeds it.

        Returns
        -------
        BatchPredictionResult
        """
        predictions: list[PredictionResult] = []
        for call_kwargs in calls:
            try:
                result = self.predict(**call_kwargs)
                predictions.append(result)
            except (KeyError, TypeError) as exc:
                logger.warning("Skipping batch call due to error: %s", exc)

        total_cost = sum(r.estimated_cost_usd for r in predictions)
        total_low = sum(r.low_cost_usd for r in predictions)
        total_high = sum(r.high_cost_usd for r in predictions)
        total_input = sum(r.input_tokens for r in predictions)
        total_output = sum(r.estimated_output_tokens for r in predictions)

        any_exceed: Optional[bool] = None
        if budget_usd is not None:
            any_exceed = total_cost > budget_usd

        return BatchPredictionResult(
            predictions=predictions,
            total_estimated_cost_usd=round(total_cost, 8),
            total_low_cost_usd=round(total_low, 8),
            total_high_cost_usd=round(total_high, 8),
            total_input_tokens=total_input,
            total_output_tokens=total_output,
            any_will_exceed_budget=any_exceed,
        )

    def __repr__(self) -> str:
        return (
            f"CostPredictor("
            f"pricing={self._pricing!r}, "
            f"counter={self._counter!r}, "
            f"estimator={self._estimator!r})"
        )
