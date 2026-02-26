"""Hierarchical sub-budget allocation on top of a parent BudgetTracker.

SubBudget wraps a parent BudgetTracker and maintains an independent
spending ledger for a named sub-agent. Every cost is recorded both
locally and forwarded to the parent, so the parent's totals always
reflect the full team spend.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone

from agent_energy_budget.budget.tracker import BudgetStatus, BudgetTracker
from agent_energy_budget.pricing.tables import get_pricing


@dataclass
class _SubCostEntry:
    """Internal record of a cost charged to this sub-budget."""

    amount_usd: float
    model: str
    operation: str
    recorded_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


class SubBudget:
    """Child budget that tracks spend against an allocated slice of a parent.

    All costs recorded here are forwarded to the parent BudgetTracker
    so that the parent always holds an accurate aggregate view.

    Parameters
    ----------
    parent_tracker:
        The parent BudgetTracker that governs the top-level budget.
    sub_id:
        Unique identifier for this sub-agent or sub-task.
    allocated_usd:
        USD amount carved out of the parent budget for this sub-budget.

    Examples
    --------
    >>> config = BudgetConfig(agent_id="team", daily_limit=10.0)
    >>> parent = BudgetTracker(config)
    >>> sub = SubBudget(parent_tracker=parent, sub_id="researcher", allocated_usd=3.0)
    >>> sub.record_cost(0.05, model="claude-haiku-4", operation="summarise")
    """

    def __init__(
        self,
        parent_tracker: BudgetTracker,
        sub_id: str,
        allocated_usd: float,
    ) -> None:
        if allocated_usd < 0:
            raise ValueError(
                f"allocated_usd must be >= 0; got {allocated_usd}"
            )
        self._parent = parent_tracker
        self._sub_id = sub_id
        self._allocated_usd = allocated_usd
        self._spent_usd: float = 0.0
        self._entries: list[_SubCostEntry] = []
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def sub_id(self) -> str:
        """Unique identifier for this sub-budget."""
        return self._sub_id

    @property
    def allocated_usd(self) -> float:
        """Total USD allocated to this sub-budget."""
        return self._allocated_usd

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record_cost(
        self,
        amount_usd: float,
        model: str = "",
        operation: str = "",
    ) -> None:
        """Record a cost against both this sub-budget and the parent tracker.

        The cost is added to the sub-budget's local ledger first, then
        forwarded to the parent tracker via :meth:`BudgetTracker.record`.

        Parameters
        ----------
        amount_usd:
            Cost in USD to record.
        model:
            Optional model identifier for the operation.
        operation:
            Optional free-text operation label (e.g. "summarise", "embed").

        Raises
        ------
        ValueError
            If *amount_usd* is negative.
        """
        if amount_usd < 0:
            raise ValueError(f"amount_usd must be >= 0; got {amount_usd}")

        entry = _SubCostEntry(
            amount_usd=amount_usd,
            model=model,
            operation=operation,
        )

        with self._lock:
            self._spent_usd = round(self._spent_usd + amount_usd, 8)
            self._entries.append(entry)

        # Resolve token counts for the parent record if a model is given
        if model:
            try:
                pricing = get_pricing(model)
                # Back-calculate approximate tokens from cost and pricing
                # (1 input token equivalent for bookkeeping; cost drives record)
                _ = pricing  # pricing resolved — use cost directly
            except KeyError:
                pass

        # Forward to parent — record as a generic cost entry
        # parent.record() expects model/input_tokens/output_tokens/cost;
        # we pass cost directly and use 0 tokens to avoid double-counting.
        self._parent.record(
            model=model or "sub_budget",
            input_tokens=0,
            output_tokens=0,
            cost=amount_usd,
        )

    def status(self) -> BudgetStatus:
        """Return a snapshot of this sub-budget's current utilisation.

        Returns
        -------
        BudgetStatus
            Snapshot using the sub-budget's own allocation as the limit.
        """
        with self._lock:
            spent = self._spent_usd
            entries = list(self._entries)

        remaining = round(self._allocated_usd - spent, 8)
        utilisation_pct = (
            round((spent / self._allocated_usd) * 100.0, 2)
            if self._allocated_usd > 0
            else 0.0
        )
        call_count = len(entries)
        avg_cost = round(spent / call_count, 8) if call_count > 0 else 0.0

        return BudgetStatus(
            agent_id=self._sub_id,
            period="sub_budget",
            limit_usd=self._allocated_usd,
            spent_usd=round(spent, 8),
            remaining_usd=remaining,
            utilisation_pct=utilisation_pct,
            call_count=call_count,
            avg_cost_per_call=avg_cost,
        )

    def is_within_budget(self, estimated_cost: float) -> bool:
        """Return True if *estimated_cost* fits within the remaining allocation.

        Parameters
        ----------
        estimated_cost:
            Prospective cost in USD for an upcoming operation.

        Returns
        -------
        bool
            True when ``spent + estimated_cost <= allocated_usd``.
        """
        with self._lock:
            return round(self._spent_usd + estimated_cost, 8) <= self._allocated_usd

    def remaining_usd(self) -> float:
        """Return remaining allocated budget in USD."""
        with self._lock:
            return round(self._allocated_usd - self._spent_usd, 8)

    def entries(self) -> list[_SubCostEntry]:
        """Return a copy of all recorded cost entries for this sub-budget."""
        with self._lock:
            return list(self._entries)
