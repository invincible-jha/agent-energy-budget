"""Core budget tracking with pre-call checking and JSONL persistence.

BudgetTracker is the main entry point for agent cost control. It:

- Checks whether an upcoming call can be afforded before it is made.
- Records actual costs after successful calls.
- Applies a configurable degradation strategy when budgets are tight.
- Persists all records to a JSONL file for reporting and auditing.
- Fires alerts at configurable utilisation thresholds.
- Supports hierarchical sub-budget allocation for multi-agent teams.
"""
from __future__ import annotations

import json
import logging
import pathlib
import threading
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Union

from agent_energy_budget.budget.alerts import BudgetAlertManager
from agent_energy_budget.budget.config import BudgetConfig, DegradationStrategy
from agent_energy_budget.pricing.tables import get_pricing

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Public data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BudgetRecommendation:
    """Outcome of a pre-call budget check.

    Parameters
    ----------
    can_afford:
        True when the call should proceed (possibly with constraints).
    model:
        The model that should be used (may differ from requested if downgraded).
    estimated_cost_usd:
        Estimated cost of the recommended call.
    remaining_usd:
        Budget remaining after this call would be charged.
    action:
        Short label describing what the tracker recommends.
    message:
        Human-readable explanation of the recommendation.
    max_output_tokens:
        Maximum output tokens to request (reduced to fit budget when needed).
    """

    can_afford: bool
    model: str
    estimated_cost_usd: float
    remaining_usd: float
    action: str
    message: str
    max_output_tokens: int


@dataclass
class BudgetStatus:
    """Snapshot of current budget usage for a single period.

    Parameters
    ----------
    agent_id:
        The agent whose budget this describes.
    period:
        Period label ("daily", "weekly", or "monthly").
    limit_usd:
        The budget cap in USD for this period.
    spent_usd:
        Amount spent so far this period.
    remaining_usd:
        Amount remaining (limit - spent); may be negative if overspent.
    utilisation_pct:
        Percentage of budget consumed (0–100+).
    call_count:
        Number of LLM calls recorded this period.
    avg_cost_per_call:
        Average cost per call in USD.
    """

    agent_id: str
    period: str
    limit_usd: float
    spent_usd: float
    remaining_usd: float
    utilisation_pct: float
    call_count: int
    avg_cost_per_call: float


@dataclass(frozen=True)
class _CallRecord:
    """Internal JSONL record for a single LLM call."""

    agent_id: str
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    recorded_at: str  # ISO-8601


# ---------------------------------------------------------------------------
# BudgetTracker
# ---------------------------------------------------------------------------


class BudgetExceededError(RuntimeError):
    """Raised by BLOCK_WITH_ERROR strategy when budget is exhausted."""

    def __init__(self, agent_id: str, remaining_usd: float, estimated_cost_usd: float) -> None:
        self.agent_id = agent_id
        self.remaining_usd = remaining_usd
        self.estimated_cost_usd = estimated_cost_usd
        super().__init__(
            f"Budget exceeded for agent '{agent_id}': "
            f"estimated cost ${estimated_cost_usd:.6f} exceeds "
            f"remaining budget ${remaining_usd:.6f}."
        )


class BudgetTracker:
    """Thread-safe agent budget tracker with JSONL persistence.

    Parameters
    ----------
    config:
        BudgetConfig that defines limits and strategy for this agent.
    storage_dir:
        Directory for JSONL spend logs. Defaults to ``~/.agent_energy_budget``.
    alert_manager:
        Optional custom alert manager. A default one is created if not provided.

    Examples
    --------
    >>> config = BudgetConfig(agent_id="my-agent", daily_limit=1.0)
    >>> tracker = BudgetTracker(config)
    >>> can_afford, rec = tracker.check("claude-haiku-4", 2000)
    >>> if can_afford:
    ...     tracker.record("claude-haiku-4", input_tokens=2000, output_tokens=512, cost=rec.estimated_cost_usd)
    """

    def __init__(
        self,
        config: BudgetConfig,
        storage_dir: Union[str, pathlib.Path, None] = None,
        alert_manager: BudgetAlertManager | None = None,
    ) -> None:
        self._config = config
        self._lock = threading.Lock()

        # Storage
        if storage_dir is None:
            storage_dir = pathlib.Path.home() / ".agent_energy_budget"
        self._storage_dir = pathlib.Path(storage_dir)
        self._storage_dir.mkdir(parents=True, exist_ok=True)
        self._log_path = self._storage_dir / f"{config.agent_id}.jsonl"

        # Alert manager
        thresholds = config.alert_thresholds
        self._alert_manager = alert_manager or BudgetAlertManager(
            warning_threshold=thresholds.warning,
            critical_threshold=thresholds.critical,
            exhausted_threshold=thresholds.exhausted,
        )

        # In-memory spend cache (rebuilt from JSONL when needed)
        self._records: list[_CallRecord] = []
        self._load_records()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def config(self) -> BudgetConfig:
        """The BudgetConfig associated with this tracker."""
        return self._config

    @property
    def agent_id(self) -> str:
        """The agent identifier."""
        return self._config.agent_id

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_records(self) -> None:
        """Load existing JSONL records into memory."""
        if not self._log_path.exists():
            return
        with self._lock:
            try:
                lines = self._log_path.read_text(encoding="utf-8").splitlines()
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                    data: object = json.loads(line)
                    if not isinstance(data, dict):
                        continue
                    try:
                        self._records.append(
                            _CallRecord(
                                agent_id=str(data.get("agent_id", self._config.agent_id)),
                                model=str(data["model"]),
                                input_tokens=int(data.get("input_tokens", 0)),
                                output_tokens=int(data.get("output_tokens", 0)),
                                cost_usd=float(data["cost_usd"]),
                                recorded_at=str(data["recorded_at"]),
                            )
                        )
                    except (KeyError, ValueError, TypeError):
                        pass  # Skip malformed records
            except Exception as exc:
                logger.warning("Failed to load budget records from %s: %s", self._log_path, exc)

    def _append_record(self, record: _CallRecord) -> None:
        """Append a call record to JSONL storage."""
        line = json.dumps(
            {
                "agent_id": record.agent_id,
                "model": record.model,
                "input_tokens": record.input_tokens,
                "output_tokens": record.output_tokens,
                "cost_usd": record.cost_usd,
                "recorded_at": record.recorded_at,
            }
        )
        try:
            with self._log_path.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")
        except OSError as exc:
            logger.error("Failed to persist budget record: %s", exc)

    def _spent_since(self, since: date) -> tuple[float, int]:
        """Return (total_cost_usd, call_count) for records since *since*."""
        cutoff = datetime.combine(since, datetime.min.time()).replace(tzinfo=timezone.utc)
        total = 0.0
        count = 0
        for record in self._records:
            try:
                ts = datetime.fromisoformat(record.recorded_at)
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                if ts >= cutoff:
                    total += record.cost_usd
                    count += 1
            except ValueError:
                continue
        return round(total, 8), count

    def _today_start(self) -> date:
        return datetime.now(timezone.utc).date()

    def _week_start(self) -> date:
        today = self._today_start()
        return today - timedelta(days=today.weekday())  # Monday

    def _month_start(self) -> date:
        today = self._today_start()
        return today.replace(day=1)

    def _best_active_limit(self) -> tuple[float, str, date]:
        """Return the tightest active (non-zero) limit and its period label.

        Returns
        -------
        tuple[float, str, date]
            (limit_usd, period_label, period_start_date)
        """
        candidates: list[tuple[float, str, date]] = []
        if self._config.daily_limit > 0:
            candidates.append((self._config.daily_limit, "daily", self._today_start()))
        if self._config.weekly_limit > 0:
            candidates.append((self._config.weekly_limit, "weekly", self._week_start()))
        if self._config.monthly_limit > 0:
            candidates.append((self._config.monthly_limit, "monthly", self._month_start()))

        if not candidates:
            return (float("inf"), "none", self._today_start())

        # Tightest = smallest limit
        return min(candidates, key=lambda t: t[0])

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int = 512,
    ) -> tuple[bool, BudgetRecommendation]:
        """Check whether an upcoming LLM call fits within the budget.

        Applies the configured DegradationStrategy when the call would
        exceed the remaining budget. The BLOCK_WITH_ERROR strategy raises
        :class:`BudgetExceededError` immediately.

        Parameters
        ----------
        model:
            Requested model identifier.
        input_tokens:
            Expected input token count for this call.
        output_tokens:
            Expected output token count (default 512).

        Returns
        -------
        tuple[bool, BudgetRecommendation]
            ``(can_afford, recommendation)`` where ``can_afford`` mirrors
            ``recommendation.can_afford``.

        Raises
        ------
        BudgetExceededError
            When degradation_strategy is BLOCK_WITH_ERROR and the budget
            is insufficient.
        """
        with self._lock:
            limit_usd, period_label, period_start = self._best_active_limit()
            spent_usd, call_count = self._spent_since(period_start)
            remaining_usd = limit_usd - spent_usd

        try:
            pricing = get_pricing(model)
        except KeyError:
            # Unknown model — assume very cheap to avoid false blocks
            from agent_energy_budget.pricing.tables import ModelPricing, ModelTier, ProviderName

            pricing = ModelPricing(
                model=model,
                provider=ProviderName.CUSTOM,
                tier=ModelTier.EFFICIENT,
                input_per_million=0.50,
                output_per_million=1.50,
            )

        estimated_cost = pricing.cost_for_tokens(input_tokens, output_tokens)

        # --- Unlimited budget fast path ---
        if limit_usd == float("inf"):
            rec = BudgetRecommendation(
                can_afford=True,
                model=model,
                estimated_cost_usd=estimated_cost,
                remaining_usd=float("inf"),
                action="proceed",
                message="No budget limit configured; proceeding.",
                max_output_tokens=output_tokens,
            )
            return True, rec

        # Fire any due alerts
        self._alert_manager.check_and_fire(
            agent_id=self._config.agent_id,
            period=period_label,
            spent_usd=spent_usd,
            limit_usd=limit_usd,
        )

        if estimated_cost <= remaining_usd:
            rec = BudgetRecommendation(
                can_afford=True,
                model=model,
                estimated_cost_usd=estimated_cost,
                remaining_usd=round(remaining_usd - estimated_cost, 8),
                action="proceed",
                message=(
                    f"Estimated cost ${estimated_cost:.6f} fits within "
                    f"{period_label} remaining budget ${remaining_usd:.6f}."
                ),
                max_output_tokens=output_tokens,
            )
            return True, rec

        # --- Budget would be exceeded — apply degradation strategy ---
        strategy = self._config.degradation_strategy

        if strategy == DegradationStrategy.BLOCK_WITH_ERROR:
            raise BudgetExceededError(
                agent_id=self._config.agent_id,
                remaining_usd=remaining_usd,
                estimated_cost_usd=estimated_cost,
            )

        if strategy == DegradationStrategy.TOKEN_REDUCTION:
            max_output = pricing.max_output_for_budget(remaining_usd, input_tokens)
            if max_output <= 0:
                rec = BudgetRecommendation(
                    can_afford=False,
                    model=model,
                    estimated_cost_usd=estimated_cost,
                    remaining_usd=remaining_usd,
                    action="block",
                    message=(
                        f"Insufficient budget for even a minimal response. "
                        f"Remaining: ${remaining_usd:.6f}."
                    ),
                    max_output_tokens=0,
                )
                return False, rec

            reduced_cost = pricing.cost_for_tokens(input_tokens, max_output)
            rec = BudgetRecommendation(
                can_afford=True,
                model=model,
                estimated_cost_usd=reduced_cost,
                remaining_usd=round(remaining_usd - reduced_cost, 8),
                action="reduce_tokens",
                message=(
                    f"Output reduced from {output_tokens} to {max_output} tokens "
                    f"to fit ${remaining_usd:.6f} remaining budget."
                ),
                max_output_tokens=max_output,
            )
            return True, rec

        if strategy == DegradationStrategy.MODEL_DOWNGRADE:
            from agent_energy_budget.pricing.tables import cheapest_model_within_budget

            best = cheapest_model_within_budget(
                budget_usd=remaining_usd,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )
            if best is None:
                rec = BudgetRecommendation(
                    can_afford=False,
                    model=model,
                    estimated_cost_usd=estimated_cost,
                    remaining_usd=remaining_usd,
                    action="block",
                    message=(
                        f"No affordable model found for budget ${remaining_usd:.6f}."
                    ),
                    max_output_tokens=0,
                )
                return False, rec

            downgrade_cost = best.cost_for_tokens(input_tokens, output_tokens)
            rec = BudgetRecommendation(
                can_afford=True,
                model=best.model,
                estimated_cost_usd=downgrade_cost,
                remaining_usd=round(remaining_usd - downgrade_cost, 8),
                action="model_downgrade",
                message=(
                    f"Downgraded from '{model}' to '{best.model}' "
                    f"(${downgrade_cost:.6f}) to fit ${remaining_usd:.6f} budget."
                ),
                max_output_tokens=output_tokens,
            )
            return True, rec

        if strategy == DegradationStrategy.CACHED_FALLBACK:
            rec = BudgetRecommendation(
                can_afford=False,
                model=model,
                estimated_cost_usd=estimated_cost,
                remaining_usd=remaining_usd,
                action="use_cache",
                message=(
                    f"Budget exceeded — use cached response if available. "
                    f"Remaining: ${remaining_usd:.6f}."
                ),
                max_output_tokens=0,
            )
            return False, rec

        # Fallback — unknown strategy, block
        rec = BudgetRecommendation(
            can_afford=False,
            model=model,
            estimated_cost_usd=estimated_cost,
            remaining_usd=remaining_usd,
            action="block",
            message=f"Budget exceeded; remaining ${remaining_usd:.6f}.",
            max_output_tokens=0,
        )
        return False, rec

    def record(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        cost: float | None = None,
    ) -> float:
        """Record actual spend after a successful LLM call.

        Parameters
        ----------
        model:
            Model that was used.
        input_tokens:
            Actual input tokens consumed.
        output_tokens:
            Actual output tokens generated.
        cost:
            Actual cost in USD. If None, it is calculated from pricing tables.

        Returns
        -------
        float
            The cost that was recorded.
        """
        if cost is None:
            try:
                pricing = get_pricing(model)
                cost = pricing.cost_for_tokens(input_tokens, output_tokens)
            except KeyError:
                cost = 0.0

        record = _CallRecord(
            agent_id=self._config.agent_id,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost,
            recorded_at=datetime.now(timezone.utc).isoformat(),
        )
        with self._lock:
            self._records.append(record)
        self._append_record(record)

        # Check alerts after recording
        limit_usd, period_label, period_start = self._best_active_limit()
        if limit_usd < float("inf"):
            spent_usd, _ = self._spent_since(period_start)
            self._alert_manager.check_and_fire(
                agent_id=self._config.agent_id,
                period=period_label,
                spent_usd=spent_usd,
                limit_usd=limit_usd,
            )

        return cost

    def status(self, period: str = "daily") -> BudgetStatus:
        """Return a snapshot of current budget utilisation.

        Parameters
        ----------
        period:
            One of "daily", "weekly", or "monthly".

        Returns
        -------
        BudgetStatus
            Current usage snapshot.
        """
        period_map: dict[str, tuple[float, date]] = {
            "daily": (self._config.daily_limit, self._today_start()),
            "weekly": (self._config.weekly_limit, self._week_start()),
            "monthly": (self._config.monthly_limit, self._month_start()),
        }
        if period not in period_map:
            raise ValueError(f"Unknown period {period!r}; use 'daily', 'weekly', or 'monthly'.")

        limit_usd, period_start = period_map[period]

        with self._lock:
            spent_usd, call_count = self._spent_since(period_start)

        remaining_usd = (limit_usd - spent_usd) if limit_usd > 0 else float("inf")
        utilisation_pct = (spent_usd / limit_usd * 100.0) if limit_usd > 0 else 0.0
        avg_cost = (spent_usd / call_count) if call_count > 0 else 0.0

        return BudgetStatus(
            agent_id=self._config.agent_id,
            period=period,
            limit_usd=limit_usd,
            spent_usd=round(spent_usd, 8),
            remaining_usd=round(remaining_usd, 8) if remaining_usd != float("inf") else float("inf"),
            utilisation_pct=round(utilisation_pct, 2),
            call_count=call_count,
            avg_cost_per_call=round(avg_cost, 8),
        )

    def allocate_sub_budget(
        self,
        sub_agent_id: str,
        fraction: float,
        storage_dir: Union[str, pathlib.Path, None] = None,
    ) -> "BudgetTracker":
        """Create a child BudgetTracker with a fractional sub-budget.

        The sub-tracker has the same degradation strategy and alert
        thresholds as the parent, with limits scaled by *fraction*.

        Parameters
        ----------
        sub_agent_id:
            Agent ID for the child tracker.
        fraction:
            Fraction of parent limits to allocate (0.0 < fraction <= 1.0).
        storage_dir:
            Override storage directory for the child tracker.

        Returns
        -------
        BudgetTracker
            New independent child tracker.

        Raises
        ------
        ValueError
            If *fraction* is not in (0.0, 1.0].
        """
        if not 0.0 < fraction <= 1.0:
            raise ValueError(f"fraction must be in (0, 1]; got {fraction}")

        sub_config = BudgetConfig(
            agent_id=sub_agent_id,
            daily_limit=round(self._config.daily_limit * fraction, 8),
            weekly_limit=round(self._config.weekly_limit * fraction, 8),
            monthly_limit=round(self._config.monthly_limit * fraction, 8),
            degradation_strategy=self._config.degradation_strategy,
            alert_thresholds=self._config.alert_thresholds,
            model_preferences=self._config.model_preferences,
            currency=self._config.currency,
            tags={**self._config.tags, "parent_agent_id": self._config.agent_id},
        )
        return BudgetTracker(
            config=sub_config,
            storage_dir=storage_dir or self._storage_dir,
        )

    def reset_period_alerts(self, period: str = "daily") -> None:
        """Reset fired alert state for a period (call at period rollover).

        Parameters
        ----------
        period:
            Period label to reset ("daily", "weekly", "monthly").
        """
        self._alert_manager.reset_period(self._config.agent_id, period)

    def total_lifetime_spend(self) -> float:
        """Return total USD spent across all recorded history."""
        with self._lock:
            return round(sum(r.cost_usd for r in self._records), 8)
