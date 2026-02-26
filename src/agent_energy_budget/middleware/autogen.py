"""AutoGen budget middleware.

AutoGenBudgetMiddleware integrates with Microsoft AutoGen's
ConversableAgent / AssistantAgent pattern to enforce budget limits
on every LLM call made by an agent.

AutoGen is not a required dependency — the import is guarded by
try/except so the package installs without it.
"""
from __future__ import annotations

import logging
from typing import Optional

from agent_energy_budget.budget.tracker import BudgetExceededError, BudgetTracker

logger = logging.getLogger(__name__)


class AutoGenBudgetHook:
    """Budget enforcement hook for Microsoft AutoGen agents.

    Intercepts message generation to check budgets before each LLM call
    and record costs after. Works with AutoGen's ConversableAgent pattern.

    Parameters
    ----------
    tracker:
        BudgetTracker to check and record against.
    default_model:
        Fallback model when the agent does not expose its model (default "gpt-4o-mini").
    default_input_tokens:
        Fallback input token estimate (default 1000).
    default_output_tokens:
        Fallback output token estimate (default 512).
    raise_on_budget_exceeded:
        When True (default), raises BudgetExceededError to abort the call.

    Examples
    --------
    >>> hook = AutoGenBudgetHook(tracker=my_tracker)
    >>> params = hook.before_call(
    ...     model="gpt-4o-mini",
    ...     prompt="Write a Python script.",
    ...     max_tokens=1024,
    ... )
    >>> if not params.get("_budget_blocked"):
    ...     # pass params to AutoGen agent
    ...     hook.after_call(model=params["model"], input_tokens=300, output_tokens=400)
    """

    def __init__(
        self,
        tracker: BudgetTracker,
        default_model: str = "gpt-4o-mini",
        default_input_tokens: int = 1000,
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
            AutoGen model identifier (matches OpenAI naming conventions).
        prompt:
            The message text (used for input token estimation).
        max_tokens:
            Maximum tokens to generate.
        **kwargs:
            Additional parameters forwarded unchanged.

        Returns
        -------
        dict[str, object]
            Updated parameters including ``"model"``, ``"max_tokens"``,
            and ``"_budget_blocked"`` indicator.

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
            msg = f"AutoGen call blocked by budget: {recommendation.message}"
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


# Alias for backward compatibility
AutoGenBudgetMiddleware = AutoGenBudgetHook

    def after_call(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        cost: Optional[float] = None,
    ) -> None:
        """Record actual cost after an AutoGen agent's LLM call.

        Parameters
        ----------
        model:
            Model identifier that was used.
        input_tokens:
            Actual input tokens consumed.
        output_tokens:
            Actual output tokens generated.
        cost:
            Explicit cost in USD; derived from pricing tables when None.
        """
        self._tracker.record(
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost=cost,
        )

    # ------------------------------------------------------------------
    # AutoGen message hook integration
    # ------------------------------------------------------------------

    def generate_reply_hook(
        self,
        messages: list[dict[str, object]],
        model: str,
        max_tokens: int = 512,
    ) -> dict[str, object]:
        """AutoGen generate_reply pre-hook: check budget before generation.

        Parameters
        ----------
        messages:
            AutoGen message history list.
        model:
            Model to use for generation.
        max_tokens:
            Maximum tokens to generate.

        Returns
        -------
        dict[str, object]
            Params dict; caller should check ``"_budget_blocked"`` key.
        """
        prompt_parts: list[str] = []
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, str):
                prompt_parts.append(content)
        combined_prompt = " ".join(prompt_parts)
        return self.before_call(
            model=model, prompt=combined_prompt, max_tokens=max_tokens
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
