"""LiteLLM adapter — wraps litellm.completion() with AumOS budget enforcement.

Provides pre-execution cost prediction via the agent-energy-budget CostPredictor
before any tokens are sent to an LLM API. If the predicted cost exceeds the
configured budget limit, BudgetExceededError is raised without making the call.

Install the extra to use this module::

    pip install agent-energy-budget[litellm]

Usage
-----
::

    from agent_energy_budget.integrations.litellm_adapter import LiteLLMBudgetWrapper

    wrapper = LiteLLMBudgetWrapper(session_budget_usd=1.0)
    response = wrapper.completion_with_budget(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Hello"}],
        budget_limit=0.05,
    )
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

from agent_energy_budget.prediction.predictor import CostPredictor, PredictionResult

try:
    import litellm  # type: ignore[import-untyped]
except ImportError as _import_error:
    raise ImportError(
        "LiteLLM is required for this adapter. "
        "Install it with: pip install agent-energy-budget[litellm]"
    ) from _import_error

logger = logging.getLogger(__name__)


class BudgetExceededError(Exception):
    """Raised when a predicted LLM call cost exceeds the configured budget.

    Parameters
    ----------
    message:
        Human-readable description of the budget breach.
    predicted_cost_usd:
        The cost estimate that triggered the error.
    budget_limit_usd:
        The limit that was exceeded.
    prediction:
        The full PredictionResult for inspection.
    """

    def __init__(
        self,
        message: str,
        predicted_cost_usd: float,
        budget_limit_usd: float,
        prediction: PredictionResult,
    ) -> None:
        super().__init__(message)
        self.predicted_cost_usd = predicted_cost_usd
        self.budget_limit_usd = budget_limit_usd
        self.prediction = prediction


@dataclass
class CallRecord:
    """Record of a single completed LiteLLM call with cost data.

    Parameters
    ----------
    model:
        The model identifier used.
    predicted_cost_usd:
        Pre-call cost estimate.
    actual_input_tokens:
        Input token count from the API response.
    actual_output_tokens:
        Output token count from the API response.
    actual_cost_usd:
        Post-call actual cost estimate (derived from usage).
    """

    model: str
    predicted_cost_usd: float
    actual_input_tokens: int
    actual_output_tokens: int
    actual_cost_usd: float


@dataclass
class SessionSummary:
    """Aggregate cost summary for the current wrapper session.

    Parameters
    ----------
    total_predicted_cost_usd:
        Sum of all pre-call predictions.
    total_actual_cost_usd:
        Sum of all post-call actual cost estimates.
    total_calls:
        Number of completed calls.
    total_input_tokens:
        Cumulative input token count.
    total_output_tokens:
        Cumulative output token count.
    call_records:
        Individual records for each call.
    """

    total_predicted_cost_usd: float
    total_actual_cost_usd: float
    total_calls: int
    total_input_tokens: int
    total_output_tokens: int
    call_records: list[CallRecord] = field(default_factory=list)


class LiteLLMBudgetWrapper:
    """Wraps litellm.completion() with pre-execution cost prediction and budget enforcement.

    Parameters
    ----------
    session_budget_usd:
        Optional cumulative session budget. When set, individual call budgets
        are also checked against remaining session budget.
    predictor:
        Custom CostPredictor instance. Defaults to the standard predictor.

    Examples
    --------
    ::

        wrapper = LiteLLMBudgetWrapper(session_budget_usd=2.0)
        response = wrapper.completion_with_budget(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Summarise this text."}],
            budget_limit=0.10,
        )
        summary = wrapper.session_summary()
        print(f"Session cost so far: ${summary.total_actual_cost_usd:.6f}")
    """

    def __init__(
        self,
        session_budget_usd: Optional[float] = None,
        predictor: Optional[CostPredictor] = None,
    ) -> None:
        self._session_budget = session_budget_usd
        self._predictor = predictor or CostPredictor()
        self._call_records: list[CallRecord] = []
        self._cumulative_actual_cost: float = 0.0
        self._cumulative_predicted_cost: float = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict_before_call(
        self,
        model: str,
        messages: list[dict[str, Any]],
        task_type: str = "chat",
        budget_usd: Optional[float] = None,
    ) -> PredictionResult:
        """Predict the cost of a LiteLLM call before it executes.

        Parameters
        ----------
        model:
            The LiteLLM model string (e.g. ``"gpt-4o-mini"``, ``"claude-3-haiku"``).
        messages:
            OpenAI-format message list.
        task_type:
            Task category hint for output token estimation.
        budget_usd:
            Optional single-call budget for ``will_exceed_budget`` flag.

        Returns
        -------
        PredictionResult
            Pre-execution cost estimate.
        """
        return self._predictor.predict(
            model=model,
            prompt=messages,
            task_type=task_type,
            budget_usd=budget_usd,
        )

    def completion_with_budget(
        self,
        model: str,
        messages: list[dict[str, Any]],
        budget_limit: float,
        task_type: str = "chat",
        **litellm_kwargs: Any,
    ) -> Any:
        """Execute a LiteLLM completion with pre-call budget enforcement.

        Predicts the call cost before sending any tokens. If the prediction
        exceeds ``budget_limit``, raises ``BudgetExceededError`` immediately
        without making the API call.

        When a session budget is configured, the remaining session budget is
        also checked — the effective limit is ``min(budget_limit, remaining)``.

        Parameters
        ----------
        model:
            LiteLLM model string.
        messages:
            OpenAI-format message list.
        budget_limit:
            Maximum allowed cost for this single call, in USD.
        task_type:
            Task hint for output token estimation.
        **litellm_kwargs:
            Additional keyword arguments forwarded to ``litellm.completion()``.

        Returns
        -------
        litellm.ModelResponse
            The raw LiteLLM response object.

        Raises
        ------
        BudgetExceededError
            If the predicted cost exceeds ``budget_limit`` or the remaining
            session budget.
        """
        effective_limit = budget_limit
        if self._session_budget is not None:
            remaining = self._session_budget - self._cumulative_actual_cost
            effective_limit = min(budget_limit, max(0.0, remaining))
            logger.debug(
                "Session budget remaining: $%.6f; effective call limit: $%.6f",
                remaining,
                effective_limit,
            )

        prediction = self.predict_before_call(
            model=model,
            messages=messages,
            task_type=task_type,
            budget_usd=effective_limit,
        )

        if prediction.will_exceed_budget:
            raise BudgetExceededError(
                message=(
                    f"Predicted cost ${prediction.estimated_cost_usd:.6f} exceeds "
                    f"budget limit ${effective_limit:.6f} for model '{model}'"
                ),
                predicted_cost_usd=prediction.estimated_cost_usd,
                budget_limit_usd=effective_limit,
                prediction=prediction,
            )

        logger.debug(
            "Budget check passed: predicted=$%.6f limit=$%.6f model=%s",
            prediction.estimated_cost_usd,
            effective_limit,
            model,
        )

        response = litellm.completion(model=model, messages=messages, **litellm_kwargs)

        self._record_call(model=model, prediction=prediction, response=response)
        return response

    def session_summary(self) -> SessionSummary:
        """Return cumulative cost summary for all calls in this session.

        Returns
        -------
        SessionSummary
            Aggregate statistics for the session.
        """
        return SessionSummary(
            total_predicted_cost_usd=round(self._cumulative_predicted_cost, 8),
            total_actual_cost_usd=round(self._cumulative_actual_cost, 8),
            total_calls=len(self._call_records),
            total_input_tokens=sum(r.actual_input_tokens for r in self._call_records),
            total_output_tokens=sum(r.actual_output_tokens for r in self._call_records),
            call_records=list(self._call_records),
        )

    def reset_session(self) -> None:
        """Reset cumulative session tracking without changing the budget limit."""
        self._call_records.clear()
        self._cumulative_actual_cost = 0.0
        self._cumulative_predicted_cost = 0.0
        logger.debug("LiteLLMBudgetWrapper session reset")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _record_call(
        self,
        model: str,
        prediction: PredictionResult,
        response: Any,
    ) -> None:
        """Extract usage from the LiteLLM response and update session totals."""
        actual_input = 0
        actual_output = 0
        actual_cost = prediction.estimated_cost_usd  # fallback

        usage = getattr(response, "usage", None)
        if usage is not None:
            actual_input = int(getattr(usage, "prompt_tokens", 0) or 0)
            actual_output = int(getattr(usage, "completion_tokens", 0) or 0)
            # Re-predict with known token counts for a tighter actual cost
            try:
                actual_result = self._predictor.predict_with_tokens(
                    model=model,
                    input_tokens=actual_input,
                    output_tokens=actual_output,
                )
                actual_cost = actual_result.estimated_cost_usd
            except KeyError:
                actual_cost = prediction.estimated_cost_usd

        record = CallRecord(
            model=model,
            predicted_cost_usd=prediction.estimated_cost_usd,
            actual_input_tokens=actual_input,
            actual_output_tokens=actual_output,
            actual_cost_usd=actual_cost,
        )
        self._call_records.append(record)
        self._cumulative_predicted_cost += prediction.estimated_cost_usd
        self._cumulative_actual_cost += actual_cost

        logger.info(
            "Call recorded: model=%s predicted=$%.6f actual=$%.6f "
            "tokens=(%d in, %d out)",
            model,
            prediction.estimated_cost_usd,
            actual_cost,
            actual_input,
            actual_output,
        )

    def __repr__(self) -> str:
        return (
            f"LiteLLMBudgetWrapper("
            f"session_budget={self._session_budget!r}, "
            f"calls={len(self._call_records)})"
        )
