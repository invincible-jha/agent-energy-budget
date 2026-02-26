"""Budget report generation in CSV, JSON, and Markdown formats.

BudgetReporter produces human-readable and machine-readable summaries
from aggregated cost data.
"""
from __future__ import annotations

import csv
import io
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Union

from agent_energy_budget.reporting.aggregator import CostAggregator


class ReportPeriod(str, Enum):
    """Standard report periods."""

    TODAY = "today"
    WEEK = "week"
    MONTH = "month"
    ALL = "all"


@dataclass(frozen=True)
class ReportSummary:
    """High-level summary statistics for a report.

    Parameters
    ----------
    period:
        The period this summary covers.
    generated_at:
        UTC timestamp when the report was generated.
    total_cost_usd:
        Total spend in USD.
    call_count:
        Number of LLM calls.
    avg_cost_per_call:
        Average cost per call in USD.
    top_agent:
        Agent with the highest spend (or empty string if none).
    top_model:
        Model with the highest spend (or empty string if none).
    top_provider:
        Provider with the highest spend (or empty string if none).
    """

    period: str
    generated_at: str
    total_cost_usd: float
    call_count: int
    avg_cost_per_call: float
    top_agent: str
    top_model: str
    top_provider: str


class BudgetReporter:
    """Generate budget reports from a CostAggregator.

    Parameters
    ----------
    aggregator:
        Pre-loaded (or auto-loading) CostAggregator to draw data from.
    """

    def __init__(self, aggregator: CostAggregator) -> None:
        self._aggregator = aggregator

    # ------------------------------------------------------------------
    # Summary helpers
    # ------------------------------------------------------------------

    def summary(self, period: Union[str, ReportPeriod] = ReportPeriod.TODAY) -> ReportSummary:
        """Compute a high-level summary for the specified period.

        Parameters
        ----------
        period:
            Report period ("today", "week", "month", or "all").

        Returns
        -------
        ReportSummary
            Aggregated statistics.
        """
        period_str = period.value if isinstance(period, ReportPeriod) else period

        total_cost = self._aggregator.total_cost(period_str)
        call_count = self._aggregator.call_count(period_str)
        avg_cost = round(total_cost / call_count, 8) if call_count > 0 else 0.0

        by_agent = self._aggregator.by_agent(period_str)
        by_model = self._aggregator.by_model(period_str)
        by_provider = self._aggregator.by_provider(period_str)

        top_agent = next(iter(by_agent), "")
        top_model = next(iter(by_model), "")
        top_provider = next(iter(by_provider), "")

        return ReportSummary(
            period=period_str,
            generated_at=datetime.now(timezone.utc).isoformat(),
            total_cost_usd=total_cost,
            call_count=call_count,
            avg_cost_per_call=avg_cost,
            top_agent=top_agent,
            top_model=top_model,
            top_provider=top_provider,
        )

    # ------------------------------------------------------------------
    # Format: JSON
    # ------------------------------------------------------------------

    def to_json(
        self,
        period: Union[str, ReportPeriod] = ReportPeriod.TODAY,
        *,
        indent: int = 2,
    ) -> str:
        """Render a full report as a JSON string.

        Parameters
        ----------
        period:
            Report period.
        indent:
            JSON indentation level.

        Returns
        -------
        str
            JSON-formatted report string.
        """
        period_str = period.value if isinstance(period, ReportPeriod) else period
        summ = self.summary(period_str)

        report: dict[str, object] = {
            "summary": {
                "period": summ.period,
                "generated_at": summ.generated_at,
                "total_cost_usd": summ.total_cost_usd,
                "call_count": summ.call_count,
                "avg_cost_per_call": summ.avg_cost_per_call,
                "top_agent": summ.top_agent,
                "top_model": summ.top_model,
                "top_provider": summ.top_provider,
            },
            "by_agent": self._aggregator.by_agent(period_str),
            "by_model": self._aggregator.by_model(period_str),
            "by_provider": self._aggregator.by_provider(period_str),
            "by_task": self._aggregator.by_task(period_str),
            "daily_breakdown": self._aggregator.daily_breakdown(),
        }
        return json.dumps(report, indent=indent, default=str)

    # ------------------------------------------------------------------
    # Format: CSV
    # ------------------------------------------------------------------

    def to_csv(
        self,
        period: Union[str, ReportPeriod] = ReportPeriod.TODAY,
    ) -> str:
        """Render individual spend records as CSV.

        Parameters
        ----------
        period:
            Report period.

        Returns
        -------
        str
            CSV string with header row.
        """
        period_str = period.value if isinstance(period, ReportPeriod) else period
        records = self._aggregator.records(period_str)

        buffer = io.StringIO()
        writer = csv.writer(buffer)
        writer.writerow(
            [
                "recorded_at",
                "agent_id",
                "model",
                "provider",
                "task",
                "input_tokens",
                "output_tokens",
                "cost_usd",
            ]
        )
        for record in sorted(records, key=lambda r: r.recorded_at):
            writer.writerow(
                [
                    record.recorded_at.isoformat(),
                    record.agent_id,
                    record.model,
                    record.provider,
                    record.task,
                    record.input_tokens,
                    record.output_tokens,
                    f"{record.cost_usd:.8f}",
                ]
            )
        return buffer.getvalue()

    # ------------------------------------------------------------------
    # Format: Markdown
    # ------------------------------------------------------------------

    def to_markdown(
        self,
        period: Union[str, ReportPeriod] = ReportPeriod.TODAY,
    ) -> str:
        """Render a human-readable Markdown report.

        Parameters
        ----------
        period:
            Report period.

        Returns
        -------
        str
            Markdown-formatted report string.
        """
        period_str = period.value if isinstance(period, ReportPeriod) else period
        summ = self.summary(period_str)
        by_agent = self._aggregator.by_agent(period_str)
        by_model = self._aggregator.by_model(period_str)
        by_provider = self._aggregator.by_provider(period_str)
        by_task = self._aggregator.by_task(period_str)
        daily = self._aggregator.daily_breakdown()

        lines: list[str] = [
            f"# Budget Report — {period_str.capitalize()}",
            "",
            f"**Generated:** {summ.generated_at}",
            "",
            "## Summary",
            "",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Total Cost | ${summ.total_cost_usd:.6f} |",
            f"| Call Count | {summ.call_count} |",
            f"| Avg Cost/Call | ${summ.avg_cost_per_call:.6f} |",
            f"| Top Agent | {summ.top_agent or '—'} |",
            f"| Top Model | {summ.top_model or '—'} |",
            f"| Top Provider | {summ.top_provider or '—'} |",
            "",
        ]

        if by_agent:
            lines += [
                "## Cost by Agent",
                "",
                "| Agent | Cost (USD) |",
                "|-------|-----------|",
            ]
            for agent, cost in by_agent.items():
                lines.append(f"| {agent} | ${cost:.6f} |")
            lines.append("")

        if by_model:
            lines += [
                "## Cost by Model",
                "",
                "| Model | Cost (USD) |",
                "|-------|-----------|",
            ]
            for model, cost in by_model.items():
                lines.append(f"| {model} | ${cost:.6f} |")
            lines.append("")

        if by_provider:
            lines += [
                "## Cost by Provider",
                "",
                "| Provider | Cost (USD) |",
                "|----------|-----------|",
            ]
            for provider, cost in by_provider.items():
                lines.append(f"| {provider} | ${cost:.6f} |")
            lines.append("")

        if by_task:
            lines += [
                "## Cost by Task",
                "",
                "| Task | Cost (USD) |",
                "|------|-----------|",
            ]
            for task, cost in by_task.items():
                lines.append(f"| {task} | ${cost:.6f} |")
            lines.append("")

        if daily:
            lines += [
                "## Daily Breakdown",
                "",
                "| Date | Cost (USD) |",
                "|------|-----------|",
            ]
            for day, cost in daily.items():
                lines.append(f"| {day} | ${cost:.6f} |")
            lines.append("")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # File output
    # ------------------------------------------------------------------

    def save(
        self,
        output_path: str,
        period: Union[str, ReportPeriod] = ReportPeriod.TODAY,
        format: str = "json",
    ) -> None:
        """Write a report to disk.

        Parameters
        ----------
        output_path:
            Destination file path.
        period:
            Report period.
        format:
            One of "json", "csv", or "md".

        Raises
        ------
        ValueError
            If an unsupported format is specified.
        """
        import pathlib

        format_map: dict[str, str] = {
            "json": self.to_json(period),
            "csv": self.to_csv(period),
            "md": self.to_markdown(period),
            "markdown": self.to_markdown(period),
        }
        if format not in format_map:
            raise ValueError(
                f"Unsupported format {format!r}. Use one of: {list(format_map)}"
            )
        pathlib.Path(output_path).write_text(format_map[format], encoding="utf-8")
