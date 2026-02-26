"""OpenAI SDK budget wrapper.

OpenAIBudgetWrapper wraps the ``openai.OpenAI`` client so that every
``chat.completions.create()`` call is automatically checked against the
budget and recorded after completion.

Requires the ``openai`` package to be installed. The wrapper is a
drop-in replacement for the standard client.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from agent_energy_budget.budget.tracker import BudgetExceededError, BudgetTracker

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class OpenAIBudgetWrapper:
    """Wrap an OpenAI client with automatic budget checking and recording.

    Parameters
    ----------
    client:
        An ``openai.OpenAI`` (or ``openai.AsyncOpenAI``) instance.
    tracker:
        BudgetTracker to check and record against.
    default_input_tokens:
        Input token estimate for pre-call budget check. Defaults to 1000.
    raise_on_budget_exceeded:
        When True (default), raises :class:`BudgetExceededError` when the
        budget check blocks a call.

    Examples
    --------
    >>> import openai
    >>> from agent_energy_budget.budget import BudgetConfig, BudgetTracker
    >>> config = BudgetConfig(agent_id="my-agent", daily_limit=5.0)
    >>> tracker = BudgetTracker(config)
    >>> client = openai.OpenAI()
    >>> wrapped = OpenAIBudgetWrapper(client, tracker)
    >>> response = wrapped.chat.completions.create(
    ...     model="gpt-4o-mini",
    ...     max_tokens=512,
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
        self.chat = _ChatProxy(self)

    def __getattr__(self, name: str) -> object:
        """Delegate unknown attribute access to the wrapped client."""
        return getattr(self._client, name)


class _ChatProxy:
    """Proxy for ``client.chat``."""

    def __init__(self, wrapper: OpenAIBudgetWrapper) -> None:
        self._wrapper = wrapper
        self.completions = _CompletionsProxy(wrapper)

    def __getattr__(self, name: str) -> object:
        underlying_chat = getattr(self._wrapper._client, "chat")
        return getattr(underlying_chat, name)


class _CompletionsProxy:
    """Proxy for ``client.chat.completions`` that intercepts ``create``."""

    def __init__(self, wrapper: OpenAIBudgetWrapper) -> None:
        self._wrapper = wrapper

    def create(self, **kwargs: object) -> object:
        """Budget-checked wrapper around ``chat.completions.create``.

        Parameters
        ----------
        **kwargs:
            All keyword arguments are passed through unchanged.

        Returns
        -------
        object
            The OpenAI API response.

        Raises
        ------
        BudgetExceededError
            When the budget check blocks the call and raise_on_budget_exceeded
            is True.
        """
        model = str(kwargs.get("model", ""))
        max_tokens = int(kwargs.get("max_tokens", 512))  # type: ignore[arg-type]

        # Estimate input tokens
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
            msg = f"OpenAI SDK call blocked by budget: {recommendation.message}"
            if self._wrapper._raise_on_exceed:
                raise BudgetExceededError(
                    agent_id=self._wrapper._tracker.agent_id,
                    remaining_usd=recommendation.remaining_usd,
                    estimated_cost_usd=recommendation.estimated_cost_usd,
                )
            logger.warning(msg)
            return None

        # Cap max_tokens to recommendation
        if recommendation.max_output_tokens < max_tokens and recommendation.max_output_tokens > 0:
            kwargs = {**kwargs, "max_tokens": recommendation.max_output_tokens}

        # Make the actual API call
        underlying = getattr(self._wrapper._client, "chat")
        underlying_completions = getattr(underlying, "completions")
        response = underlying_completions.create(**kwargs)

        # Record actual usage
        actual_input = input_tokens
        actual_output = max_tokens
        usage = getattr(response, "usage", None)
        if usage is not None:
            actual_input = getattr(usage, "prompt_tokens", input_tokens)
            actual_output = getattr(usage, "completion_tokens", max_tokens)

        self._wrapper._tracker.record(
            model=model,
            input_tokens=int(actual_input),
            output_tokens=int(actual_output),
        )
        return response

    def __getattr__(self, name: str) -> object:
        underlying = getattr(self._wrapper._client, "chat")
        underlying_completions = getattr(underlying, "completions")
        return getattr(underlying_completions, name)


# ---------------------------------------------------------------------------
# Middleware (before_call / after_call style)
# ---------------------------------------------------------------------------


class OpenAIBudgetMiddleware:
    """Budget enforcement middleware for OpenAI SDK calls.

    Parameters
    ----------
    tracker:
        BudgetTracker to check and record against.
    default_input_tokens:
        Fallback input token estimate (default 1000).
    default_output_tokens:
        Fallback output token estimate (default 512).

    Examples
    --------
    >>> middleware = OpenAIBudgetMiddleware(tracker=my_tracker)
    >>> params = middleware.before_call(
    ...     model="gpt-4o-mini",
    ...     prompt="Explain quantum entanglement.",
    ...     max_tokens=256,
    ... )
    >>> if not params.get("_budget_blocked"):
    ...     middleware.after_call(
    ...         model=params["model"], input_tokens=80, output_tokens=200
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
        max_tokens: int | None = None,
        messages: list[dict[str, object]] | None = None,
        **kwargs: object,
    ) -> dict[str, object]:
        """Run pre-call budget check and return (possibly modified) params.

        Parameters
        ----------
        model:
            OpenAI model identifier (e.g. ``"gpt-4o-mini"``).
        prompt:
            Plain prompt string for token estimation when *messages* not given.
        max_tokens:
            Maximum tokens to generate. Defaults to ``default_output_tokens``.
        messages:
            OpenAI messages list (preferred for token estimation).
        **kwargs:
            Additional call parameters forwarded unchanged.

        Returns
        -------
        dict[str, object]
            Updated call parameters; check ``"_budget_blocked"`` key.
        """
        estimated_input = self._estimate_input_tokens(prompt, messages)
        effective_output = max_tokens if max_tokens is not None else self._default_output_tokens

        can_afford, recommendation = self._tracker.check(
            model, estimated_input, effective_output
        )

        if not can_afford:
            logger.warning(
                "OpenAI call to '%s' blocked by budget: %s",
                model,
                recommendation.message,
            )
            return {
                "model": model,
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
        cost: float | None = None,
    ) -> None:
        """Record actual cost after an OpenAI API call completes.

        Parameters
        ----------
        model:
            Model identifier that was used.
        input_tokens:
            Actual prompt tokens from the API response.
        output_tokens:
            Actual completion tokens from the API response.
        cost:
            Explicit cost in USD; derived from pricing tables when None.
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
        messages: list[dict[str, object]] | None,
    ) -> int:
        """Estimate input tokens from prompt text or messages list."""
        try:
            from agent_energy_budget.pricing.token_counter import TokenCounter

            counter = TokenCounter()
            if messages:
                text_msgs: list[dict[str, str]] = [
                    {str(k): str(v) for k, v in msg.items()}
                    for msg in messages
                    if isinstance(msg, dict)
                ]
                return counter.count_messages(text_msgs)
            if prompt:
                return counter.count(prompt)
        except Exception:
            pass
        if prompt:
            return max(1, len(prompt) // 4)
        return self._default_input_tokens
