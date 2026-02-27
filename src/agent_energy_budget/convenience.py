"""Convenience API for agent-energy-budget — 3-line quickstart.

Example
-------
::

    from agent_energy_budget import Budget
    budget = Budget(limit=5.00)
    can_afford, rec = budget.check("claude-haiku-4", input_tokens=1000)
    print(can_afford, rec.action)

"""
from __future__ import annotations

from typing import Any, Tuple


class Budget:
    """Zero-config agent budget manager for the 80% use case.

    Wraps BudgetTracker with in-memory storage and sensible defaults.
    No filesystem configuration is required.

    Parameters
    ----------
    limit:
        Daily USD budget limit. None means unlimited.
    agent_id:
        Agent identifier for tracking purposes.

    Example
    -------
    ::

        from agent_energy_budget import Budget
        budget = Budget(limit=5.00)
        can_afford, rec = budget.check("claude-haiku-4", 1000)
        if can_afford:
            budget.record("claude-haiku-4", 1000, 512)
    """

    def __init__(
        self,
        limit: float | None = None,
        agent_id: str = "quickstart-agent",
    ) -> None:
        import tempfile
        from agent_energy_budget.budget.config import BudgetConfig
        from agent_energy_budget.budget.tracker import BudgetTracker

        self._temp_dir = tempfile.mkdtemp(prefix="agent_energy_budget_")
        self._config = BudgetConfig(
            agent_id=agent_id,
            daily_limit=limit if limit is not None else 0.0,
        )
        self._tracker = BudgetTracker(
            config=self._config,
            storage_dir=self._temp_dir,
        )
        self.limit = limit
        self.agent_id = agent_id

    def check(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int = 512,
    ) -> Tuple[bool, Any]:
        """Check whether a call to the given model fits the budget.

        Parameters
        ----------
        model:
            Model identifier (e.g. ``"claude-haiku-4"``).
        input_tokens:
            Expected input token count.
        output_tokens:
            Expected output token count (default 512).

        Returns
        -------
        tuple[bool, BudgetRecommendation]
            ``(can_afford, recommendation)`` — recommendation includes
            ``.action``, ``.message``, and ``.estimated_cost_usd``.

        Example
        -------
        ::

            budget = Budget(limit=1.00)
            ok, rec = budget.check("claude-haiku-4", 500)
            print(ok, rec.estimated_cost_usd)
        """
        return self._tracker.check(model, input_tokens, output_tokens)

    def record(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        cost: float | None = None,
    ) -> float:
        """Record actual spend after an LLM call.

        Parameters
        ----------
        model:
            Model that was used.
        input_tokens:
            Actual input tokens consumed.
        output_tokens:
            Actual output tokens generated.
        cost:
            Actual cost in USD. Calculated from pricing tables if None.

        Returns
        -------
        float
            The cost that was recorded.
        """
        return self._tracker.record(model, input_tokens, output_tokens, cost)

    def status(self, period: str = "daily") -> Any:
        """Return current budget utilisation snapshot.

        Parameters
        ----------
        period:
            One of ``"daily"``, ``"weekly"``, or ``"monthly"``.

        Returns
        -------
        BudgetStatus
            Snapshot with ``.spent_usd``, ``.remaining_usd``, ``.utilisation_pct``.
        """
        return self._tracker.status(period)

    @property
    def tracker(self) -> Any:
        """The underlying BudgetTracker instance."""
        return self._tracker

    def __repr__(self) -> str:
        return f"Budget(limit={self.limit!r}, agent_id={self.agent_id!r})"
