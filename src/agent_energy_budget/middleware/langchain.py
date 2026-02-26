"""LangChain callback handler for budget enforcement.

LangChainBudgetCallback hooks into LangChain's callback system to check
budgets before LLM calls and record costs after. LangChain is not required
at install time; it is imported lazily if present.

Usage
-----
Install as a callback on any LangChain LLM, chain, or agent::

    llm = ChatOpenAI(callbacks=[LangChainBudgetCallback(tracker)])

Or use the before/after hooks directly::

    callback = LangChainBudgetCallback(tracker)
    params = callback.before_call(model="gpt-4o-mini", prompt="Hello")
    # ... make the LLM call ...
    callback.after_call(model="gpt-4o-mini", input_tokens=50, output_tokens=80)
"""
from __future__ import annotations

import logging
import uuid
from typing import Union

from agent_energy_budget.budget.tracker import BudgetExceededError, BudgetTracker
from agent_energy_budget.pricing.tables import PROVIDER_PRICING

logger = logging.getLogger(__name__)


class LangChainBudgetCallback:
    """LangChain callback that enforces a BudgetTracker on every LLM call.

    Parameters
    ----------
    tracker:
        BudgetTracker to check and record against.
    default_model:
        Model name to use when LangChain does not expose it in the serialised
        LLM object. Defaults to "gpt-4o-mini".
    default_input_tokens:
        Fallback input token estimate (default 1000).
    default_output_tokens:
        Fallback output token estimate (default 512).
    raise_on_budget_exceeded:
        When True (default), raises :class:`BudgetExceededError` inside
        ``on_llm_start`` / ``before_call`` to prevent the API call.
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
        # run_id -> (model, input_tokens, output_tokens)
        self._pending: dict[str, tuple[str, int, int]] = {}

    # ------------------------------------------------------------------
    # Explicit before/after hooks (framework-agnostic)
    # ------------------------------------------------------------------

    def before_call(
        self,
        model: str,
        prompt: str = "",
        max_tokens: int | None = None,
    ) -> dict[str, object]:
        """Pre-call budget check; returns updated call params.

        Parameters
        ----------
        model:
            The model to use.
        prompt:
            Prompt text (tokenised for estimation).
        max_tokens:
            Requested max output tokens.

        Returns
        -------
        dict[str, object]
            Updated params with ``model``, ``max_tokens``,
            and ``_budget_blocked`` keys.
        """
        input_tokens = self._estimate_tokens(prompt)
        output_tokens = max_tokens if max_tokens is not None else self._default_output_tokens

        can_afford, recommendation = self._tracker.check(model, input_tokens, output_tokens)

        if not can_afford:
            msg = f"LangChain call blocked by budget: {recommendation.message}"
            if self._raise_on_exceed:
                raise BudgetExceededError(
                    agent_id=self._tracker.agent_id,
                    remaining_usd=recommendation.remaining_usd,
                    estimated_cost_usd=recommendation.estimated_cost_usd,
                )
            logger.warning(msg)
            return {"model": model, "max_tokens": 0, "_budget_blocked": True}

        return {
            "model": recommendation.model,
            "max_tokens": recommendation.max_output_tokens,
            "_budget_blocked": False,
        }

    def after_call(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        cost: float | None = None,
    ) -> None:
        """Record actual cost after call completion.

        Parameters
        ----------
        model:
            Model that was used.
        input_tokens:
            Actual input tokens consumed.
        output_tokens:
            Actual output tokens generated.
        cost:
            Actual cost in USD. If None, derived from pricing tables.
        """
        self._tracker.record(
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost=cost,
        )

    # ------------------------------------------------------------------
    # LangChain BaseCallbackHandler protocol (duck-typed)
    # ------------------------------------------------------------------

    def on_llm_start(
        self,
        serialized: dict[str, object],
        prompts: list[str],
        *,
        run_id: Union[str, uuid.UUID, None] = None,
        **kwargs: object,
    ) -> None:
        """LangChain hook: fires before an LLM generates a response."""
        model = _extract_model(serialized, self._default_model)
        run_key = str(run_id) if run_id else str(uuid.uuid4())

        input_tokens = sum(self._estimate_tokens(p) for p in prompts)
        output_tokens = self._default_output_tokens

        can_afford, recommendation = self._tracker.check(model, input_tokens, output_tokens)
        self._pending[run_key] = (model, input_tokens, output_tokens)

        if not can_afford:
            msg = f"LangChain LLM call blocked by budget: {recommendation.message}"
            if self._raise_on_exceed:
                raise BudgetExceededError(
                    agent_id=self._tracker.agent_id,
                    remaining_usd=recommendation.remaining_usd,
                    estimated_cost_usd=recommendation.estimated_cost_usd,
                )
            logger.warning(msg)

    def on_llm_end(
        self,
        response: object,
        *,
        run_id: Union[str, uuid.UUID, None] = None,
        **kwargs: object,
    ) -> None:
        """LangChain hook: fires after a successful LLM response."""
        run_key = str(run_id) if run_id else ""
        model, default_input, default_output = self._pending.pop(
            run_key, (self._default_model, self._default_input_tokens, self._default_output_tokens)
        )

        actual_input = default_input
        actual_output = default_output
        llm_output = getattr(response, "llm_output", None)
        if isinstance(llm_output, dict):
            token_usage = llm_output.get("token_usage", {})
            if isinstance(token_usage, dict):
                actual_input = int(token_usage.get("prompt_tokens", default_input))
                actual_output = int(token_usage.get("completion_tokens", default_output))

        self._tracker.record(model=model, input_tokens=actual_input, output_tokens=actual_output)

    def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: Union[str, uuid.UUID, None] = None,
        **kwargs: object,
    ) -> None:
        """LangChain hook: fires when an LLM call errors."""
        run_key = str(run_id) if run_id else ""
        self._pending.pop(run_key, None)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _estimate_tokens(self, text: str) -> int:
        if not text:
            return self._default_input_tokens
        from agent_energy_budget.pricing.token_counter import TokenCounter

        return TokenCounter().count(text)


def _extract_model(serialized: dict[str, object], default: str) -> str:
    """Extract a model name from a LangChain serialized LLM dict."""
    kwargs = serialized.get("kwargs", {})
    if isinstance(kwargs, dict):
        for key in ("model", "model_name", "model_id"):
            value = kwargs.get(key)
            if isinstance(value, str) and value:
                return value
    class_name = str(serialized.get("name", "")).lower()
    for model_name in PROVIDER_PRICING:
        if any(part in class_name for part in model_name.split("-")[:2]):
            return model_name
    return default


# Alias — the middleware __init__.py exports LangChainBudgetMiddleware
LangChainBudgetMiddleware = LangChainBudgetCallback
