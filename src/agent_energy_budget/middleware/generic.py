"""Generic budget guard decorator.

``budget_guard`` wraps any LLM-calling function with pre-call budget
checking and post-call cost recording. It works with any provider or
framework by extracting token usage from a standard return value protocol.
"""
from __future__ import annotations

import functools
import inspect
import logging
from collections.abc import Callable
from typing import TypeVar

from agent_energy_budget.budget.tracker import BudgetExceededError, BudgetTracker

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., object])


class BudgetGuardError(RuntimeError):
    """Raised by ``budget_guard`` when the budget check blocks a call."""

    def __init__(self, message: str, remaining_usd: float) -> None:
        self.remaining_usd = remaining_usd
        super().__init__(message)


def budget_guard(
    tracker: BudgetTracker,
    model: str,
    input_tokens: int = 1000,
    output_tokens: int = 512,
    *,
    raise_on_block: bool = True,
) -> Callable[[F], F]:
    """Decorator factory: wrap an LLM-calling function with budget enforcement.

    The wrapped function is called only when the budget check passes. After
    a successful call, the tracker records the estimated cost unless the
    wrapped function returns an object with ``usage`` metadata (see below).

    Usage metadata protocol
    -----------------------
    If the return value has a ``.usage`` attribute with ``input_tokens``
    and ``output_tokens`` int attributes, the actual token counts are used
    for cost recording. Otherwise the decorator's *input_tokens* and
    *output_tokens* parameters are used.

    Parameters
    ----------
    tracker:
        The BudgetTracker to check against and record to.
    model:
        Model identifier for this call.
    input_tokens:
        Estimated input token count for the pre-call budget check.
    output_tokens:
        Estimated output token count for the pre-call budget check.
    raise_on_block:
        When True (default), raises :class:`BudgetGuardError` if the budget
        check blocks the call. When False, returns None instead.

    Returns
    -------
    Callable[[F], F]
        Decorator that enforces budget on the wrapped function.

    Examples
    --------
    >>> @budget_guard(tracker, model="gpt-4o-mini", input_tokens=500)
    ... def call_llm(prompt: str) -> str:
    ...     ...  # actual API call
    """

    def decorator(func: F) -> F:
        is_async = inspect.iscoroutinefunction(func)

        @functools.wraps(func)
        def sync_wrapper(*args: object, **kwargs: object) -> object:
            can_afford, recommendation = tracker.check(model, input_tokens, output_tokens)

            if not can_afford:
                msg = (
                    f"Budget check blocked call to '{model}': "
                    f"{recommendation.message}"
                )
                if raise_on_block:
                    raise BudgetGuardError(msg, recommendation.remaining_usd)
                logger.warning(msg)
                return None

            result = func(*args, **kwargs)
            _record_from_result(tracker, model, input_tokens, output_tokens, result)
            return result

        @functools.wraps(func)
        async def async_wrapper(*args: object, **kwargs: object) -> object:
            can_afford, recommendation = tracker.check(model, input_tokens, output_tokens)

            if not can_afford:
                msg = (
                    f"Budget check blocked call to '{model}': "
                    f"{recommendation.message}"
                )
                if raise_on_block:
                    raise BudgetGuardError(msg, recommendation.remaining_usd)
                logger.warning(msg)
                return None

            result = await func(*args, **kwargs)  # type: ignore[misc]
            _record_from_result(tracker, model, input_tokens, output_tokens, result)
            return result

        wrapper = async_wrapper if is_async else sync_wrapper
        return wrapper  # type: ignore[return-value]

    return decorator


def _record_from_result(
    tracker: BudgetTracker,
    model: str,
    default_input_tokens: int,
    default_output_tokens: int,
    result: object,
) -> None:
    """Extract token usage from *result* and record cost in *tracker*."""
    actual_input = default_input_tokens
    actual_output = default_output_tokens

    # Try usage attribute (Anthropic / OpenAI SDK style)
    usage = getattr(result, "usage", None)
    if usage is not None:
        actual_input = getattr(usage, "input_tokens", None) or getattr(
            usage, "prompt_tokens", default_input_tokens
        )
        actual_output = getattr(usage, "output_tokens", None) or getattr(
            usage, "completion_tokens", default_output_tokens
        )

    tracker.record(
        model=model,
        input_tokens=int(actual_input),
        output_tokens=int(actual_output),
    )
