"""Anthropic SDK budget middleware.

Two integration styles are provided:

1. ``AnthropicBudgetWrapper`` — wraps an ``anthropic.Anthropic`` client
   instance so that every ``messages.create()`` call is automatically
   checked and recorded. This is a drop-in client replacement.

2. ``AnthropicBudgetMiddleware`` — explicit before_call / after_call hooks
   for callers that prefer to manage the lifecycle themselves or need to
   compose with other frameworks.

The ``anthropic`` package is not a required dependency. It is imported
lazily where needed; the module itself always imports successfully.
"""
from __future__ import annotations

import logging
from typing import Optional

from agent_energy_budget.budget.tracker import BudgetExceededError, BudgetTracker

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Middleware (before_call / after_call style)
# ---------------------------------------------------------------------------


class AnthropicBudgetMiddleware:
    """Budget enforcement middleware for Anthropic SDK calls.

    Parameters
    ----------
    tracker:
        BudgetTracker to check and record against.
    degradation_manager:
        Optional DegradationManager for model downgrade / token limit logic.
    default_input_tokens:
        Fallback input token estimate (default 1000).
    default_output_tokens:
        Fallback output token estimate (default 512).

    Examples
    --------
    >>> middleware = AnthropicBudgetMiddleware(tracker=my_tracker)
    >>> params = middleware.before_call(
    ...     model="claude-haiku-4",
    ...     prompt="Summarise this document.",
    ...     max_tokens=1024,
    ... )
    >>> if not params.get("_budget_blocked"):
    ...     # call anthropic.Anthropic().messages.create(**params)
    ...     middleware.after_call(
    ...         model=params["model"], input_tokens=100, output_tokens=200
    ...     )
    """

    def __init__(
        self,
        tracker: BudgetTracker,
        default_input_tokens: int = 1000,
        default_output_tokens: int = 512,
    ) -> None:
        self._tracker = tracker
        self._default_input_tokens = default_input_tokens
        self._default_output_tokens = default_output_tokens

    def before_call(
        self,
        model: str,
        prompt: str = "",
        max_tokens: Optional[int] = None,
        messages: Optional[list[dict[str, object]]] = None,
        **kwargs: object,
    ) -> dict[str, object]:
        """Run pre-call budget check and return (possibly modified) params.

        Parameters
        ----------
        model:
            Anthropic model identifier (e.g. ``"claude-haiku-4"``).
        prompt:
            Prompt string for token estimation when *messages* is not given.
        max_tokens:
            Maximum tokens to generate. Defaults to ``default_output_tokens``.
        messages:
            Anthropic messages list (overrides *prompt* for token estimation).
        **kwargs:
            Additional call parameters forwarded unchanged.

        Returns
        -------
        dict[str, object]
            Updated call parameters; check ``"_budget_blocked"`` key.
        """
        estimated_input = self._estimate_input_tokens(prompt, messages)
        effective_output = max_tokens if max_tokens is not None else self._default_output_tokens
        effective_model = model

        can_afford, recommendation = self._tracker.check(
            effective_model, estimated_input, effective_output
        )

        if not can_afford:
            logger.warning(
                "Anthropic call to '%s' blocked by budget: %s",
                effective_model,
                recommendation.message,
            )
            return {
                "model": effective_model,
                "max_tokens": 0,
                "_budget_blocked": True,
                **kwargs,
            }

        final_output = min(effective_output, recommendation.max_output_tokens)
        result: dict[str, object] = {
            "model": recommendation.model,
            "max_tokens": final_output,
            "_budget_blocked": False,
            "_estimated_input_tokens": estimated_input,
        }
        if messages is not None:
            result["messages"] = messages
        result.update(kwargs)
        return result

    def after_call(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        cost: Optional[float] = None,
    ) -> None:
        """Record actual cost after an Anthropic API call completes.

        Parameters
        ----------
        model:
            Model identifier that was used.
        input_tokens:
            Actual input tokens from the API response.
        output_tokens:
            Actual output tokens from the API response.
        cost:
            Explicit cost; derived from pricing tables when None.
        """
        self._tracker.record(
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost=cost,
        )

    def _estimate_input_tokens(
        self,
        prompt: str,
        messages: Optional[list[dict[str, object]]],
    ) -> int:
        """Estimate input tokens from prompt or messages list."""
        try:
            from agent_energy_budget.pricing.token_counter import TokenCounter

            counter = TokenCounter()
            if messages:
                text_msgs: list[dict[str, str]] = []
                for msg in messages:
                    if isinstance(msg, dict):
                        text_msgs.append(
                            {str(k): str(v) for k, v in msg.items()}
                        )
                return counter.count_messages(text_msgs)
            if prompt:
                return counter.count(prompt)
        except Exception:
            pass
        if prompt:
            return max(1, len(prompt) // 4)
        return self._default_input_tokens


# ---------------------------------------------------------------------------
# Drop-in client wrapper (legacy / convenience style)
# ---------------------------------------------------------------------------


class AnthropicBudgetWrapper:
    """Wrap an ``anthropic.Anthropic`` client with automatic budget enforcement.

    This is a drop-in replacement for the standard Anthropic client in code
    that uses ``client.messages.create(...)``.

    Parameters
    ----------
    client:
        An ``anthropic.Anthropic`` instance.
    tracker:
        BudgetTracker to check and record against.
    default_input_tokens:
        Fallback input estimate for pre-call check (default 1000).
    raise_on_budget_exceeded:
        When True (default), raises :class:`BudgetExceededError` when
        the budget check blocks a call.

    Examples
    --------
    >>> import anthropic
    >>> wrapped = AnthropicBudgetWrapper(anthropic.Anthropic(), tracker)
    >>> response = wrapped.messages.create(
    ...     model="claude-haiku-4", max_tokens=512,
    ...     messages=[{"role": "user", "content": "Hello"}],
    ... )
    """

    def __init__(
        self,
        client: object,
        tracker: BudgetTracker,
        default_input_tokens: int = 1000,
        raise_on_budget_exceeded: bool = True,
    ) -> None:
        self._client = client
        self._tracker = tracker
        self._default_input_tokens = default_input_tokens
        self._raise_on_exceed = raise_on_budget_exceeded
        self.messages = _MessagesProxy(self)

    def __getattr__(self, name: str) -> object:
        """Delegate unknown attribute access to the wrapped client."""
        return getattr(self._client, name)


class _MessagesProxy:
    """Proxy for ``client.messages`` that intercepts ``create`` calls."""

    def __init__(self, wrapper: AnthropicBudgetWrapper) -> None:
        self._wrapper = wrapper

    def create(self, **kwargs: object) -> object:
        """Budget-checked wrapper around ``messages.create``.

        Parameters
        ----------
        **kwargs:
            All keyword arguments are forwarded to the original
            ``messages.create`` method after budget adjustment.

        Returns
        -------
        object
            The Anthropic API response.

        Raises
        ------
        BudgetExceededError
            When the budget check blocks the call and
            ``raise_on_budget_exceeded`` is True.
        """
        model = str(kwargs.get("model", ""))
        max_tokens = int(kwargs.get("max_tokens", 512))  # type: ignore[arg-type]

        messages = kwargs.get("messages", [])
        input_tokens = self._wrapper._default_input_tokens
        if isinstance(messages, list):
            from agent_energy_budget.pricing.token_counter import TokenCounter

            counter = TokenCounter()
            text_msgs: list[dict[str, str]] = []
            for m in messages:
                if isinstance(m, dict):
                    text_msgs.append({str(k): str(v) for k, v in m.items()})
            input_tokens = counter.count_messages(text_msgs)

        can_afford, recommendation = self._wrapper._tracker.check(
            model, input_tokens, max_tokens
        )

        if not can_afford:
            msg = f"Anthropic SDK call blocked by budget: {recommendation.message}"
            if self._wrapper._raise_on_exceed:
                raise BudgetExceededError(
                    agent_id=self._wrapper._tracker.agent_id,
                    remaining_usd=recommendation.remaining_usd,
                    estimated_cost_usd=recommendation.estimated_cost_usd,
                )
            logger.warning(msg)
            return None  # type: ignore[return-value]

        if recommendation.max_output_tokens < max_tokens:
            kwargs = {**kwargs, "max_tokens": recommendation.max_output_tokens}

        underlying_messages = getattr(self._wrapper._client, "messages")
        response = underlying_messages.create(**kwargs)

        actual_input = input_tokens
        actual_output = max_tokens
        usage = getattr(response, "usage", None)
        if usage is not None:
            actual_input = getattr(usage, "input_tokens", input_tokens)
            actual_output = getattr(usage, "output_tokens", max_tokens)

        self._wrapper._tracker.record(
            model=model,
            input_tokens=int(actual_input),
            output_tokens=int(actual_output),
        )
        return response

    def __getattr__(self, name: str) -> object:
        """Delegate other methods to the underlying messages object."""
        underlying_messages = getattr(self._wrapper._client, "messages")
        return getattr(underlying_messages, name)
