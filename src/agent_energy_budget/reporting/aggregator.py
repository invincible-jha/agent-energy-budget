"""Cost aggregation from JSONL spend logs.

CostAggregator reads one or more JSONL log files produced by BudgetTracker
and aggregates spend data by agent, model, provider, task, and time period.
"""
from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Union

from agent_energy_budget.pricing.tables import PROVIDER_PRICING


@dataclass(frozen=True)
class SpendRecord:
    """A single parsed spend entry from a JSONL log.

    Parameters
    ----------
    agent_id:
        Agent that incurred this cost.
    model:
        Model used.
    provider:
        Provider extracted from pricing table (or "unknown").
    input_tokens:
        Input tokens consumed.
    output_tokens:
        Output tokens generated.
    cost_usd:
        Cost in USD.
    recorded_at:
        UTC datetime of the record.
    task:
        Optional task label (if stored in record).
    """

    agent_id: str
    model: str
    provider: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    recorded_at: datetime
    task: str = ""


class CostAggregator:
    """Aggregate LLM spend records from JSONL log files.

    Parameters
    ----------
    log_paths:
        One or more paths to JSONL log files. Can also be a directory,
        in which case all ``*.jsonl`` files in that directory are loaded.
    """

    def __init__(
        self,
        log_paths: Union[str, pathlib.Path, list[Union[str, pathlib.Path]]],
    ) -> None:
        if isinstance(log_paths, (str, pathlib.Path)):
            resolved = pathlib.Path(log_paths)
            if resolved.is_dir():
                self._paths: list[pathlib.Path] = sorted(resolved.glob("*.jsonl"))
            else:
                self._paths = [resolved]
        else:
            self._paths = [pathlib.Path(p) for p in log_paths]

        self._records: list[SpendRecord] = []
        self._loaded = False

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load(self) -> "CostAggregator":
        """Parse all JSONL files and populate the in-memory record list.

        Returns
        -------
        CostAggregator
            Self, for method chaining.
        """
        self._records = []
        for path in self._paths:
            if not path.exists():
                continue
            try:
                lines = path.read_text(encoding="utf-8").splitlines()
            except OSError:
                continue
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                try:
                    raw: object = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(raw, dict):
                    continue
                record = _parse_record(raw)
                if record is not None:
                    self._records.append(record)
        self._loaded = True
        return self

    def _ensure_loaded(self) -> None:
        if not self._loaded:
            self.load()

    # ------------------------------------------------------------------
    # Filtering helpers
    # ------------------------------------------------------------------

    def _filter_by_period(
        self, records: list[SpendRecord], period: str
    ) -> list[SpendRecord]:
        now = datetime.now(timezone.utc)
        today = now.date()

        if period == "today":
            cutoff = datetime.combine(today, datetime.min.time()).replace(tzinfo=timezone.utc)
        elif period == "week":
            week_start = today - timedelta(days=today.weekday())
            cutoff = datetime.combine(week_start, datetime.min.time()).replace(tzinfo=timezone.utc)
        elif period == "month":
            month_start = today.replace(day=1)
            cutoff = datetime.combine(month_start, datetime.min.time()).replace(tzinfo=timezone.utc)
        elif period == "all":
            return records
        else:
            raise ValueError(f"Unknown period {period!r}; use 'today', 'week', 'month', or 'all'.")

        return [r for r in records if r.recorded_at >= cutoff]

    # ------------------------------------------------------------------
    # Aggregation queries
    # ------------------------------------------------------------------

    def total_cost(self, period: str = "all") -> float:
        """Return total cost in USD for the specified period.

        Parameters
        ----------
        period:
            "today", "week", "month", or "all".

        Returns
        -------
        float
            Total spend in USD.
        """
        self._ensure_loaded()
        filtered = self._filter_by_period(self._records, period)
        return round(sum(r.cost_usd for r in filtered), 8)

    def by_agent(self, period: str = "all") -> dict[str, float]:
        """Aggregate total cost per agent.

        Parameters
        ----------
        period:
            Time period filter.

        Returns
        -------
        dict[str, float]
            agent_id -> total_cost_usd mapping, sorted by cost descending.
        """
        self._ensure_loaded()
        filtered = self._filter_by_period(self._records, period)
        totals: dict[str, float] = {}
        for record in filtered:
            totals[record.agent_id] = round(
                totals.get(record.agent_id, 0.0) + record.cost_usd, 8
            )
        return dict(sorted(totals.items(), key=lambda kv: kv[1], reverse=True))

    def by_model(self, period: str = "all") -> dict[str, float]:
        """Aggregate total cost per model.

        Parameters
        ----------
        period:
            Time period filter.

        Returns
        -------
        dict[str, float]
            model -> total_cost_usd mapping, sorted by cost descending.
        """
        self._ensure_loaded()
        filtered = self._filter_by_period(self._records, period)
        totals: dict[str, float] = {}
        for record in filtered:
            totals[record.model] = round(
                totals.get(record.model, 0.0) + record.cost_usd, 8
            )
        return dict(sorted(totals.items(), key=lambda kv: kv[1], reverse=True))

    def by_provider(self, period: str = "all") -> dict[str, float]:
        """Aggregate total cost per provider.

        Parameters
        ----------
        period:
            Time period filter.

        Returns
        -------
        dict[str, float]
            provider -> total_cost_usd mapping, sorted by cost descending.
        """
        self._ensure_loaded()
        filtered = self._filter_by_period(self._records, period)
        totals: dict[str, float] = {}
        for record in filtered:
            totals[record.provider] = round(
                totals.get(record.provider, 0.0) + record.cost_usd, 8
            )
        return dict(sorted(totals.items(), key=lambda kv: kv[1], reverse=True))

    def by_task(self, period: str = "all") -> dict[str, float]:
        """Aggregate total cost per task label.

        Parameters
        ----------
        period:
            Time period filter.

        Returns
        -------
        dict[str, float]
            task -> total_cost_usd mapping, sorted by cost descending.
        """
        self._ensure_loaded()
        filtered = self._filter_by_period(self._records, period)
        totals: dict[str, float] = {}
        for record in filtered:
            task = record.task or "untagged"
            totals[task] = round(totals.get(task, 0.0) + record.cost_usd, 8)
        return dict(sorted(totals.items(), key=lambda kv: kv[1], reverse=True))

    def call_count(self, period: str = "all") -> int:
        """Return total number of LLM calls in the period."""
        self._ensure_loaded()
        return len(self._filter_by_period(self._records, period))

    def daily_breakdown(self) -> dict[str, float]:
        """Return total cost per calendar day for all loaded history.

        Returns
        -------
        dict[str, float]
            ISO date string -> total_cost_usd, sorted ascending by date.
        """
        self._ensure_loaded()
        daily: dict[str, float] = {}
        for record in self._records:
            day = record.recorded_at.date().isoformat()
            daily[day] = round(daily.get(day, 0.0) + record.cost_usd, 8)
        return dict(sorted(daily.items()))

    def records(self, period: str = "all") -> list[SpendRecord]:
        """Return filtered raw records.

        Parameters
        ----------
        period:
            Time period filter.

        Returns
        -------
        list[SpendRecord]
            Matching records.
        """
        self._ensure_loaded()
        return self._filter_by_period(self._records, period)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _parse_record(raw: dict[str, object]) -> SpendRecord | None:
    """Parse a raw JSONL dict into a SpendRecord, or None if invalid."""
    try:
        model = str(raw["model"])
        cost_usd = float(raw["cost_usd"])  # type: ignore[arg-type]
        recorded_at_str = str(raw["recorded_at"])
        recorded_at = datetime.fromisoformat(recorded_at_str)
        if recorded_at.tzinfo is None:
            recorded_at = recorded_at.replace(tzinfo=timezone.utc)
    except (KeyError, ValueError, TypeError):
        return None

    # Resolve provider from pricing table
    pricing = PROVIDER_PRICING.get(model)
    provider = pricing.provider.value if pricing else "unknown"

    return SpendRecord(
        agent_id=str(raw.get("agent_id", "unknown")),
        model=model,
        provider=provider,
        input_tokens=int(raw.get("input_tokens", 0)),  # type: ignore[arg-type]
        output_tokens=int(raw.get("output_tokens", 0)),  # type: ignore[arg-type]
        cost_usd=cost_usd,
        recorded_at=recorded_at,
        task=str(raw.get("task", "")),
    )
