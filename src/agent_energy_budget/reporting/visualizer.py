"""ASCII terminal visualisations for budget data.

AsciiVisualizer renders simple, terminal-friendly bar charts and tables
using only the Python standard library. Works in any terminal that supports
basic text output (no ANSI requirements, though Rich can enhance it).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from agent_energy_budget.reporting.top_n import CostHotspot


@dataclass
class BarChartConfig:
    """Configuration for ASCII bar chart rendering.

    Parameters
    ----------
    width:
        Total width of the chart in characters (default 60).
    bar_char:
        Character used to fill bars (default '#').
    empty_char:
        Character used for the unfilled portion (default '.').
    show_values:
        Whether to append the numeric value after each bar.
    value_format:
        Python format string for numeric values (default '.4f').
    title_width:
        Maximum label width before truncation (default 20).
    """

    width: int = 60
    bar_char: str = "#"
    empty_char: str = "."
    show_values: bool = True
    value_format: str = ".4f"
    title_width: int = 20


class AsciiVisualizer:
    """Render ASCII charts for budget utilisation and cost distribution.

    Parameters
    ----------
    chart_config:
        Optional custom bar chart configuration.
    """

    def __init__(self, chart_config: BarChartConfig | None = None) -> None:
        self._config = chart_config or BarChartConfig()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _truncate(self, text: str, max_len: int) -> str:
        if len(text) <= max_len:
            return text.ljust(max_len)
        return text[: max_len - 3] + "..."

    def _bar(self, fraction: float, bar_width: int) -> str:
        filled = round(fraction * bar_width)
        filled = max(0, min(bar_width, filled))
        return (
            self._config.bar_char * filled
            + self._config.empty_char * (bar_width - filled)
        )

    # ------------------------------------------------------------------
    # Budget utilisation bar
    # ------------------------------------------------------------------

    def budget_utilisation_bar(
        self,
        agent_id: str,
        spent_usd: float,
        limit_usd: float,
        period: str = "daily",
    ) -> str:
        """Render a single budget utilisation bar.

        Parameters
        ----------
        agent_id:
            Agent identifier for the label.
        spent_usd:
            Amount spent in USD.
        limit_usd:
            Budget limit in USD.
        period:
            Period label for the header.

        Returns
        -------
        str
            Multi-line ASCII chart string.
        """
        fraction = min(1.0, spent_usd / limit_usd) if limit_usd > 0 else 0.0
        pct = fraction * 100.0
        bar_width = self._config.width - self._config.title_width - 3
        bar = self._bar(fraction, bar_width)
        label = self._truncate(f"{agent_id} ({period})", self._config.title_width)
        value_str = f" ${spent_usd:.4f}/${limit_usd:.4f} ({pct:.1f}%)" if self._config.show_values else ""
        return f"{label} [{bar}]{value_str}"

    # ------------------------------------------------------------------
    # Cost distribution chart
    # ------------------------------------------------------------------

    def cost_distribution_chart(
        self,
        cost_map: dict[str, float],
        title: str = "Cost Distribution",
        max_rows: int = 20,
    ) -> str:
        """Render a horizontal bar chart of costs.

        Parameters
        ----------
        cost_map:
            name -> cost_usd mapping (will be sorted descending).
        title:
            Chart title.
        max_rows:
            Maximum number of rows to render.

        Returns
        -------
        str
            Multi-line ASCII chart.
        """
        if not cost_map:
            return f"{title}\n(no data)"

        sorted_items = sorted(cost_map.items(), key=lambda kv: kv[1], reverse=True)[:max_rows]
        max_cost = max(v for _, v in sorted_items)
        bar_width = self._config.width - self._config.title_width - 3

        lines: list[str] = [
            title,
            "=" * (self._config.width + self._config.title_width + 5),
        ]
        for name, cost in sorted_items:
            fraction = cost / max_cost if max_cost > 0 else 0.0
            bar = self._bar(fraction, bar_width)
            label = self._truncate(name, self._config.title_width)
            value_str = f" ${cost:{self._config.value_format}}" if self._config.show_values else ""
            lines.append(f"{label} [{bar}]{value_str}")

        lines.append("=" * (self._config.width + self._config.title_width + 5))
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Top-N hotspot table
    # ------------------------------------------------------------------

    def hotspot_table(
        self,
        hotspots: list[CostHotspot],
        title: str = "Top Cost Consumers",
    ) -> str:
        """Render a formatted table of CostHotspot entries.

        Parameters
        ----------
        hotspots:
            List of hotspot entries to display.
        title:
            Table title.

        Returns
        -------
        str
            Multi-line ASCII table.
        """
        if not hotspots:
            return f"{title}\n(no data)"

        col_widths = {"rank": 4, "name": 30, "cost": 14, "pct": 8}
        separator = (
            "+" + "-" * (col_widths["rank"] + 2)
            + "+" + "-" * (col_widths["name"] + 2)
            + "+" + "-" * (col_widths["cost"] + 2)
            + "+" + "-" * (col_widths["pct"] + 2)
            + "+"
        )
        header = (
            f"| {'#':>{col_widths['rank']}} "
            f"| {'Name':<{col_widths['name']}} "
            f"| {'Cost (USD)':>{col_widths['cost']}} "
            f"| {'% of Total':>{col_widths['pct']}} |"
        )

        lines: list[str] = [title, separator, header, separator]
        for h in hotspots:
            name_str = self._truncate(h.name, col_widths["name"]).rstrip()
            cost_str = f"${h.total_cost_usd:.6f}"
            pct_str = f"{h.pct_of_total:.1f}%"
            lines.append(
                f"| {h.rank:>{col_widths['rank']}} "
                f"| {name_str:<{col_widths['name']}} "
                f"| {cost_str:>{col_widths['cost']}} "
                f"| {pct_str:>{col_widths['pct']}} |"
            )
        lines.append(separator)
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Multi-agent budget status grid
    # ------------------------------------------------------------------

    def status_grid(
        self,
        statuses: list[dict[str, object]],
    ) -> str:
        """Render a grid of agent budget statuses.

        Parameters
        ----------
        statuses:
            List of dicts with keys: agent_id, period, spent_usd,
            limit_usd, utilisation_pct, call_count.

        Returns
        -------
        str
            Multi-line ASCII status grid.
        """
        if not statuses:
            return "No budget status data available."

        header = f"{'Agent':<25} {'Period':<8} {'Utilisation':<14} {'Spent/Limit':<22} {'Calls':<6}"
        separator = "-" * len(header)
        lines: list[str] = ["Budget Status Grid", separator, header, separator]

        for s in statuses:
            agent_id = str(s.get("agent_id", ""))
            period = str(s.get("period", ""))
            spent = float(s.get("spent_usd", 0))  # type: ignore[arg-type]
            limit = float(s.get("limit_usd", 0))  # type: ignore[arg-type]
            util_pct = float(s.get("utilisation_pct", 0))  # type: ignore[arg-type]
            calls = int(s.get("call_count", 0))  # type: ignore[arg-type]

            # Mini utilisation bar (10 chars)
            mini_bar_width = 10
            fraction = min(1.0, util_pct / 100.0)
            filled = round(fraction * mini_bar_width)
            mini_bar = "#" * filled + "." * (mini_bar_width - filled)

            util_str = f"[{mini_bar}] {util_pct:.1f}%"
            spend_str = f"${spent:.4f}/${limit:.4f}" if limit > 0 else f"${spent:.4f}/unlimited"

            agent_display = agent_id[:24]
            lines.append(
                f"{agent_display:<25} {period:<8} {util_str:<14} {spend_str:<22} {calls:<6}"
            )

        lines.append(separator)
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Daily spend sparkline
    # ------------------------------------------------------------------

    def daily_sparkline(
        self,
        daily_costs: dict[str, float],
        title: str = "Daily Spend Trend",
    ) -> str:
        """Render a compact sparkline of daily costs.

        Uses Unicode block characters (▁▂▃▄▅▆▇█) for compact representation.

        Parameters
        ----------
        daily_costs:
            ISO date -> cost_usd mapping (will be sorted by date).
        title:
            Chart title.

        Returns
        -------
        str
            Multi-line sparkline string.
        """
        if not daily_costs:
            return f"{title}\n(no data)"

        blocks = " ▁▂▃▄▅▆▇█"
        sorted_days = sorted(daily_costs.items())
        costs = [v for _, v in sorted_days]
        max_cost = max(costs) if costs else 1.0
        if max_cost == 0:
            max_cost = 1.0

        spark = ""
        for cost in costs:
            idx = math.floor((cost / max_cost) * (len(blocks) - 1))
            spark += blocks[min(idx, len(blocks) - 1)]

        first_day = sorted_days[0][0] if sorted_days else ""
        last_day = sorted_days[-1][0] if sorted_days else ""
        total = sum(costs)

        lines = [
            title,
            f"{first_day} {'':─<{len(spark) - 2}} {last_day}",
            spark,
            f"Total: ${total:.4f}  |  Days: {len(costs)}  |  Avg/day: ${total / max(len(costs), 1):.4f}",
        ]
        return "\n".join(lines)
