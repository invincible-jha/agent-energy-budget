"""Abstract base class for all degradation strategies."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class DegradationResult:
    """Result returned by any degradation strategy.

    Parameters
    ----------
    can_proceed:
        Whether the LLM call should proceed (possibly with modified params).
    recommended_model:
        Model to use (may differ from requested if a downgrade occurred).
    max_tokens:
        Maximum output tokens allowed. 0 means the call should be blocked.
    action:
        Short machine-readable action label (e.g. "reduce_tokens").
    message:
        Human-readable explanation for the decision.
    """

    can_proceed: bool
    recommended_model: str
    max_tokens: int
    action: str
    message: str


class DegradationStrategyBase(ABC):
    """Abstract base class that all degradation strategies must implement.

    Subclasses implement :meth:`apply` which receives the requested call
    parameters and the remaining budget, and returns a :class:`DegradationResult`
    describing how (or whether) to proceed.
    """

    @abstractmethod
    def apply(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        remaining_budget: float,
    ) -> DegradationResult:
        """Evaluate whether / how to proceed given the current budget.

        Parameters
        ----------
        model:
            The model requested for the upcoming LLM call.
        input_tokens:
            Expected number of input tokens.
        output_tokens:
            Expected number of output tokens.
        remaining_budget:
            Remaining budget in USD for the current period.

        Returns
        -------
        DegradationResult
            Instructions for the budget tracker / middleware layer.
        """
        ...

    def name(self) -> str:
        """Return the canonical strategy name (used in StrategyRegistry)."""
        return type(self).__name__.lower().replace("strategy", "")
