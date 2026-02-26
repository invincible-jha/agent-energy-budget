"""Top-N cost consumer analysis.

TopNAnalyzer identifies the most expensive agents, models, tasks, and
providers in the spend history — helping teams pinpoint cost hotspots
quickly.
"""
from __future__ import annotations

from dataclasses import dataclass

from agent_energy_budget.reporting.aggregator import CostAggregator


@dataclass(frozen=True)
class CostHotspot:
    """A single entry in a top-N cost analysis result.

    Parameters
    ----------
    rank:
        Position in the ranking (1 = most expensive).
    dimension:
        What this entry represents ("agent", "model", "task", "provider").
    name:
        Identifier value for the dimension.
    total_cost_usd:
        Total cost in USD.
    pct_of_total:
        Percentage of the overall spend this entry accounts for.
    """

    rank: int
    dimension: str
    name: str
    total_cost_usd: float
    pct_of_total: float


class TopNAnalyzer:
    """Identify top-N most expensive consumers from a CostAggregator.

    Parameters
    ----------
    aggregator:
        Data source. Will be auto-loaded on first query if not already.
    """

    def __init__(self, aggregator: CostAggregator) -> None:
        self._aggregator = aggregator

    # ------------------------------------------------------------------
    # Core analysis
    # ------------------------------------------------------------------

    def _build_hotspots(
        self,
        dimension: str,
        cost_map: dict[str, float],
        overall_total: float,
        top_n: int,
    ) -> list[CostHotspot]:
        """Convert a cost_map to a sorted, ranked list of CostHotspot objects.

        Parameters
        ----------
        dimension:
            Label for what the map represents.
        cost_map:
            name -> cost_usd mapping.
        overall_total:
            Total spend used for percentage calculation.
        top_n:
            Maximum entries to return.

        Returns
        -------
        list[CostHotspot]
            Top-N entries, ranked 1 = most expensive.
        """
        sorted_items = sorted(cost_map.items(), key=lambda kv: kv[1], reverse=True)
        results: list[CostHotspot] = []
        for rank, (name, cost) in enumerate(sorted_items[:top_n], start=1):
            pct = (cost / overall_total * 100.0) if overall_total > 0 else 0.0
            results.append(
                CostHotspot(
                    rank=rank,
                    dimension=dimension,
                    name=name,
                    total_cost_usd=cost,
                    pct_of_total=round(pct, 2),
                )
            )
        return results

    # ------------------------------------------------------------------
    # Public queries
    # ------------------------------------------------------------------

    def top_agents(
        self, n: int = 10, period: str = "all"
    ) -> list[CostHotspot]:
        """Return the N most expensive agents.

        Parameters
        ----------
        n:
            Number of results to return.
        period:
            Time period filter.

        Returns
        -------
        list[CostHotspot]
            Ranked hotspots for agents.
        """
        cost_map = self._aggregator.by_agent(period)
        total = self._aggregator.total_cost(period)
        return self._build_hotspots("agent", cost_map, total, n)

    def top_models(
        self, n: int = 10, period: str = "all"
    ) -> list[CostHotspot]:
        """Return the N most expensive models.

        Parameters
        ----------
        n:
            Number of results to return.
        period:
            Time period filter.

        Returns
        -------
        list[CostHotspot]
            Ranked hotspots for models.
        """
        cost_map = self._aggregator.by_model(period)
        total = self._aggregator.total_cost(period)
        return self._build_hotspots("model", cost_map, total, n)

    def top_tasks(
        self, n: int = 10, period: str = "all"
    ) -> list[CostHotspot]:
        """Return the N most expensive tasks.

        Parameters
        ----------
        n:
            Number of results to return.
        period:
            Time period filter.

        Returns
        -------
        list[CostHotspot]
            Ranked hotspots for tasks.
        """
        cost_map = self._aggregator.by_task(period)
        total = self._aggregator.total_cost(period)
        return self._build_hotspots("task", cost_map, total, n)

    def top_providers(
        self, n: int = 10, period: str = "all"
    ) -> list[CostHotspot]:
        """Return the N most expensive providers.

        Parameters
        ----------
        n:
            Number of results to return.
        period:
            Time period filter.

        Returns
        -------
        list[CostHotspot]
            Ranked hotspots for providers.
        """
        cost_map = self._aggregator.by_provider(period)
        total = self._aggregator.total_cost(period)
        return self._build_hotspots("provider", cost_map, total, n)

    def top_by(
        self,
        dimension: str,
        n: int = 10,
        period: str = "all",
    ) -> list[CostHotspot]:
        """Generic top-N query by dimension name.

        Parameters
        ----------
        dimension:
            One of "agent", "model", "task", "provider".
        n:
            Number of results to return.
        period:
            Time period filter.

        Returns
        -------
        list[CostHotspot]
            Ranked hotspots.

        Raises
        ------
        ValueError
            If dimension is not one of the supported values.
        """
        dispatch: dict[str, list[CostHotspot]] = {
            "agent": self.top_agents(n, period),
            "model": self.top_models(n, period),
            "task": self.top_tasks(n, period),
            "provider": self.top_providers(n, period),
        }
        if dimension not in dispatch:
            raise ValueError(
                f"Unknown dimension {dimension!r}. "
                f"Valid dimensions: {sorted(dispatch.keys())}"
            )
        return dispatch[dimension]

    def hotspot_report(
        self, n: int = 5, period: str = "all"
    ) -> dict[str, list[CostHotspot]]:
        """Return top-N hotspots for all dimensions in one call.

        Parameters
        ----------
        n:
            Number of results per dimension.
        period:
            Time period filter.

        Returns
        -------
        dict[str, list[CostHotspot]]
            Maps "agent", "model", "task", "provider" to their top-N lists.
        """
        return {
            "agent": self.top_agents(n, period),
            "model": self.top_models(n, period),
            "task": self.top_tasks(n, period),
            "provider": self.top_providers(n, period),
        }
