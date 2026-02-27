"""Real-time budget enforcement with atomic threading.Lock updates.

:class:`BudgetEnforcer` provides a synchronous pre-call check that
atomically deducts a tentative reservation from the remaining budget.
If the call completes, the reservation is confirmed.  If the call
fails or is cancelled, the reservation is released.

This two-phase commit pattern prevents concurrent callers from
simultaneously draining a budget that should block one of them.

Usage
-----
::

    from agent_energy_budget.enforcement import BudgetEnforcer, EnforcerConfig

    config = EnforcerConfig(limit_usd=1.00, period_label="daily")
    enforcer = BudgetEnforcer(config)

    result = enforcer.check_and_reserve(estimated_cost_usd=0.05)
    if result.allowed:
        # call LLM
        enforcer.confirm(result.reservation_id, actual_cost_usd=0.04)
    else:
        print("Budget exceeded:", result.rejection_reason)
"""
from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class EnforcerConfig:
    """Configuration for :class:`BudgetEnforcer`.

    Parameters
    ----------
    limit_usd:
        Hard budget cap in USD.  Must be positive.  Use ``float("inf")``
        for an effectively unlimited budget.
    period_label:
        Human-readable label for the budget period (e.g. ``"daily"``).
    allow_overrun_fraction:
        Fraction of the limit by which a single call may exceed the
        remaining budget before being blocked.  Default ``0.0`` means
        no overrun is permitted.  Set to e.g. ``0.05`` to allow 5% grace.
    agent_id:
        Identifier of the agent this enforcer is managing.
    """

    limit_usd: float
    period_label: str = "daily"
    allow_overrun_fraction: float = 0.0
    agent_id: str = "default"

    def __post_init__(self) -> None:
        if self.limit_usd <= 0:
            raise ValueError(
                f"limit_usd must be positive, got {self.limit_usd!r}."
            )
        if not (0.0 <= self.allow_overrun_fraction <= 1.0):
            raise ValueError(
                f"allow_overrun_fraction must be in [0, 1], "
                f"got {self.allow_overrun_fraction!r}."
            )


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EnforcementResult:
    """Result of a :meth:`BudgetEnforcer.check_and_reserve` call.

    Parameters
    ----------
    allowed:
        ``True`` when the call may proceed.
    reservation_id:
        Opaque token to pass to :meth:`~BudgetEnforcer.confirm` or
        :meth:`~BudgetEnforcer.release`.  Empty string when *allowed*
        is ``False``.
    reserved_usd:
        Amount reserved from the budget (equal to the estimated cost).
    remaining_after_reservation_usd:
        Budget remaining after the reservation was deducted.
    rejection_reason:
        Human-readable message explaining why the call was blocked.
        Empty string when *allowed* is True.
    """

    allowed: bool
    reservation_id: str
    reserved_usd: float
    remaining_after_reservation_usd: float
    rejection_reason: str = ""


@dataclass
class EnforcerStatus:
    """Point-in-time snapshot of the enforcer's budget state.

    Parameters
    ----------
    agent_id:
        The agent identifier.
    period_label:
        Budget period label.
    limit_usd:
        Hard cap in USD.
    spent_usd:
        Confirmed spend so far.
    reserved_usd:
        Amount currently held in pending reservations.
    remaining_usd:
        Effective remaining budget (limit - spent - reserved).
    active_reservations:
        Number of open (unconfirmed/unreleased) reservations.
    total_calls_allowed:
        Running count of calls that were allowed.
    total_calls_rejected:
        Running count of calls that were rejected.
    """

    agent_id: str
    period_label: str
    limit_usd: float
    spent_usd: float
    reserved_usd: float
    remaining_usd: float
    active_reservations: int
    total_calls_allowed: int
    total_calls_rejected: int


# ---------------------------------------------------------------------------
# BudgetEnforcer
# ---------------------------------------------------------------------------


class BudgetEnforcer:
    """Synchronous real-time budget enforcer with atomic reservation logic.

    All public methods are thread-safe.  Internal state is protected by a
    single :class:`threading.Lock`; no I/O is performed inside the lock.

    Parameters
    ----------
    config:
        :class:`EnforcerConfig` specifying the limit and policy.

    Example
    -------
    ::

        config = EnforcerConfig(limit_usd=5.00, agent_id="worker-1")
        enforcer = BudgetEnforcer(config)

        result = enforcer.check_and_reserve(0.10)
        if result.allowed:
            enforcer.confirm(result.reservation_id, actual_cost_usd=0.09)
    """

    def __init__(self, config: EnforcerConfig) -> None:
        self._config = config
        self._lock = threading.Lock()

        # Confirmed spend (immutable once confirmed)
        self._spent_usd: float = 0.0

        # Pending reservations: reservation_id -> reserved_amount
        self._reservations: dict[str, float] = {}

        # Statistics
        self._total_allowed: int = 0
        self._total_rejected: int = 0

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def config(self) -> EnforcerConfig:
        """The configuration for this enforcer."""
        return self._config

    # ------------------------------------------------------------------
    # Core public API
    # ------------------------------------------------------------------

    def check_and_reserve(self, estimated_cost_usd: float) -> EnforcementResult:
        """Atomically check budget and reserve *estimated_cost_usd* if allowed.

        The reservation is deducted from the effective remaining budget so
        that concurrent callers cannot both succeed when only one can afford
        the call.

        Parameters
        ----------
        estimated_cost_usd:
            Estimated cost of the upcoming LLM call in USD.  Must be >= 0.

        Returns
        -------
        EnforcementResult
            ``allowed=True`` with a reservation_id when the call may
            proceed; ``allowed=False`` with a reason when blocked.

        Raises
        ------
        ValueError
            If *estimated_cost_usd* is negative.
        """
        if estimated_cost_usd < 0:
            raise ValueError(
                f"estimated_cost_usd must be non-negative, "
                f"got {estimated_cost_usd!r}."
            )

        with self._lock:
            effective_remaining = self._effective_remaining()
            grace = self._config.limit_usd * self._config.allow_overrun_fraction
            allowable_limit = effective_remaining + grace

            if estimated_cost_usd > allowable_limit:
                self._total_rejected += 1
                return EnforcementResult(
                    allowed=False,
                    reservation_id="",
                    reserved_usd=0.0,
                    remaining_after_reservation_usd=effective_remaining,
                    rejection_reason=(
                        f"Estimated cost ${estimated_cost_usd:.6f} exceeds "
                        f"{self._config.period_label} remaining budget "
                        f"${effective_remaining:.6f} for agent "
                        f"'{self._config.agent_id}'."
                    ),
                )

            reservation_id = str(uuid.uuid4())
            self._reservations[reservation_id] = estimated_cost_usd
            remaining_after = effective_remaining - estimated_cost_usd
            self._total_allowed += 1

        return EnforcementResult(
            allowed=True,
            reservation_id=reservation_id,
            reserved_usd=estimated_cost_usd,
            remaining_after_reservation_usd=remaining_after,
        )

    def confirm(
        self, reservation_id: str, actual_cost_usd: float | None = None
    ) -> float:
        """Confirm a reservation after a successful LLM call.

        The reservation is removed and the actual cost is added to the
        confirmed spend.  If *actual_cost_usd* is ``None``, the originally
        reserved amount is used.

        Parameters
        ----------
        reservation_id:
            Token returned by :meth:`check_and_reserve`.
        actual_cost_usd:
            Actual cost incurred.  Pass ``None`` to use the reserved amount.

        Returns
        -------
        float
            The actual cost that was recorded.

        Raises
        ------
        KeyError
            If *reservation_id* is not a pending reservation.
        ValueError
            If *actual_cost_usd* is negative.
        """
        with self._lock:
            if reservation_id not in self._reservations:
                raise KeyError(
                    f"Reservation '{reservation_id}' not found.  "
                    "It may have already been confirmed or released."
                )
            reserved_amount = self._reservations.pop(reservation_id)
            cost_to_record = reserved_amount if actual_cost_usd is None else actual_cost_usd
            if cost_to_record < 0:
                raise ValueError(
                    f"actual_cost_usd must be non-negative, got {cost_to_record!r}."
                )
            self._spent_usd += cost_to_record

        return cost_to_record

    def release(self, reservation_id: str) -> float:
        """Release a reservation without recording any spend.

        Call this when a reserved call is cancelled or fails before
        incurring cost.  The reserved amount is returned to the budget.

        Parameters
        ----------
        reservation_id:
            Token returned by :meth:`check_and_reserve`.

        Returns
        -------
        float
            The released reservation amount in USD.

        Raises
        ------
        KeyError
            If *reservation_id* is not a pending reservation.
        """
        with self._lock:
            if reservation_id not in self._reservations:
                raise KeyError(
                    f"Reservation '{reservation_id}' not found."
                )
            return self._reservations.pop(reservation_id)

    def record_direct(self, cost_usd: float) -> None:
        """Record cost directly without a prior reservation.

        Use this for background or non-reserved spend (e.g., streaming
        tokens received after budget was checked).

        Parameters
        ----------
        cost_usd:
            Cost to add to confirmed spend.  Must be non-negative.

        Raises
        ------
        ValueError
            If *cost_usd* is negative.
        """
        if cost_usd < 0:
            raise ValueError(f"cost_usd must be non-negative, got {cost_usd!r}.")
        with self._lock:
            self._spent_usd += cost_usd

    def reset(self) -> None:
        """Reset all spend and reservations to zero (new budget period).

        Active reservations are discarded; confirmed spend is zeroed.
        Statistics counters are also reset.
        """
        with self._lock:
            self._spent_usd = 0.0
            self._reservations.clear()
            self._total_allowed = 0
            self._total_rejected = 0

    def status(self) -> EnforcerStatus:
        """Return a snapshot of the current budget state.

        Returns
        -------
        EnforcerStatus
            Thread-safe point-in-time snapshot.
        """
        with self._lock:
            reserved = sum(self._reservations.values())
            remaining = self._effective_remaining()
            return EnforcerStatus(
                agent_id=self._config.agent_id,
                period_label=self._config.period_label,
                limit_usd=self._config.limit_usd,
                spent_usd=self._spent_usd,
                reserved_usd=reserved,
                remaining_usd=remaining,
                active_reservations=len(self._reservations),
                total_calls_allowed=self._total_allowed,
                total_calls_rejected=self._total_rejected,
            )

    # ------------------------------------------------------------------
    # Internal helpers (must be called with _lock held)
    # ------------------------------------------------------------------

    def _effective_remaining(self) -> float:
        """Return remaining budget accounting for confirmed spend and reservations."""
        reserved = sum(self._reservations.values())
        return self._config.limit_usd - self._spent_usd - reserved
