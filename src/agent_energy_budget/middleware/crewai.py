"""CrewAI budget middleware.

CrewAIBudgetMiddleware wraps CrewAI LLM invocations with budget
enforcement. It uses the same before_call / after_call pattern so it
can be composed with CrewAI's callback hooks or used standalone.

CrewAI is not a required dependency — the import is guarded so the
package installs without it.
"""
from __future__ import annotations

import logging
from typing import Optional

from agent_energy_budget.budget.tracker import BudgetExceededError, BudgetTracker

logger = logging.getLogger(__name__)


class CrewAIBudgetMiddleware:
    """Budget enforcement middleware for CrewAI agent LLM calls.

    Parameters
    ----------
    tracker:
        BudgetTracker to check and record against.
    default_model:
        Model name used when not explicitly provided (default "gpt-4o-mini").
    default_input_tokens:
        Fallback input token estimate (default 1500).
    default_output_tokens:
        Fallback output token estimate (default 512).
    raise_on_budget_exceeded:
        When True (default), raises BudgetExceededError to abort the task.

    Examples
    --------
    >>> middleware = CrewAIBudgetMiddleware(tracker=my_tracker)
    >>> params = middleware.before_call(model="gpt-4o-mini", prompt="Research topic X.")
    >>> if not params.get("_budget_blocked"):
    ...     # invoke crew task
    ...     middleware.after_call(model=params["model"], input_tokens=200, output_tokens=100)
    """

    def __init__(
        self,
        tracker: BudgetTracker,
        default_model: str = "gpt-4o-mini",
        default_input_tokens: int = 1500,
        default_output_tokens: int = 512,
        raise_on_budget_exceeded: bool = True,
    ) -> None:
        self._tracker = tracker
        self._default_model = default_model
        self._default_input_tokens = default_input_tokens
        self._default_output_tokens = default_output_tokens
        self._raise_on_exceed = raise_on_budget_exceeded

    # ------------------------------------------------------------------
    # Middleware hooks
    # ------------------------------------------------------------------

    def before_call(
        self,
        model: str,
        prompt: str = "",
        max_tokens: Optional[int] = None,
        **kwargs: object,
    ) -> dict[str, object]:
        """Run pre-call budget check and return call parameters.

        Parameters
        ----------
        model:
            CrewAI LLM model identifier.
        prompt:
            Task prompt (used for input token estimation).
        max_tokens:
            Max tokens to generate. Defaults to ``default_output_tokens``.
        **kwargs:
            Additional keyword arguments passed through unchanged.

        Returns
        -------
        dict[str, object]
            Call parameters with ``"model"``, ``"max_tokens"``, and
            ``"_budget_blocked"`` keys.

        Raises
        ------
        BudgetExceededError
            When budget is insufficient and raise_on_budget_exceeded is True.
        """
        estimated_input = self._estimate_input_tokens(prompt)
        effective_output = max_tokens if max_tokens is not None else self._default_output_tokens

        can_afford, recommendation = self._tracker.check(
            model, estimated_input, effective_output
        )

        if not can_afford:
            msg = f"CrewAI call blocked by budget: {recommendation.message}"
            if self._raise_on_exceed:
                raise BudgetExceededError(
                    agent_id=self._tracker.agent_id,
                    remaining_usd=recommendation.remaining_usd,
                    estimated_cost_usd=recommendation.estimated_cost_usd,
                )
            logger.warning(msg)
            return {
                "model": model,
                "max_tokens": 0,
                "_budget_blocked": True,
                **kwargs,
            }

        final_output = min(effective_output, recommendation.max_output_tokens)
        return {
            "model": recommendation.model,
            "max_tokens": final_output,
            "_budget_blocked": False,
            "_estimated_input_tokens": estimated_input,
            **kwargs,
        }

    def after_call(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        cost: Optional[float] = None,
    ) -> None:
        """Record actual cost after a CrewAI LLM call completes.

        Parameters
        ----------
        model:
            Model identifier used.
        input_tokens:
            Actual input tokens consumed.
        output_tokens:
            Actual output tokens generated.
        cost:
            Actual cost in USD (computed from pricing tables when None).
        """
        self._tracker.record(
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost=cost,
        )

    # ------------------------------------------------------------------
    # CrewAI task callback helpers
    # ------------------------------------------------------------------

    def on_task_start(self, task_description: str, agent_role: str, model: str) -> None:
        """Called when a CrewAI task starts; runs a pre-call budget check."""
        params = self.before_call(model=model, prompt=task_description)
        if params.get("_budget_blocked"):
            logger.warning(
                "CrewAI task for agent '%s' is budget-blocked.", agent_role
            )

    def on_task_end(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
    ) -> None:
        """Called when a CrewAI task ends; records cost."""
        self.after_call(
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _estimate_input_tokens(self, prompt: str) -> int:
        """Estimate input token count from prompt text."""
        if not prompt:
            return self._default_input_tokens
        try:
            from agent_energy_budget.pricing.token_counter import TokenCounter

            counter = TokenCounter()
            return counter.count(prompt)
        except Exception:
            return max(1, len(prompt) // 4)
