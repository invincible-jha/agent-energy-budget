"""CrewAI adapter for agent_energy_budget."""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

_DEFAULT_PRICING: dict[str, float] = {
    "gpt-4o": 0.000005,
    "gpt-4o-mini": 0.0000006,
    "gpt-4-turbo": 0.00001,
}


class CrewAICostTracker:
    """Cost tracking adapter for CrewAI.

    Tracks token usage and accumulated USD cost per task and per agent within
    a CrewAI crew execution.

    Usage::

        from agent_energy_budget.adapters.crewai import CrewAICostTracker
        tracker = CrewAICostTracker()
    """

    def __init__(self, model_pricing: dict[str, float] | None = None) -> None:
        self.model_pricing: dict[str, float] = model_pricing if model_pricing is not None else _DEFAULT_PRICING
        self.total_tokens: int = 0
        self.total_cost_usd: float = 0.0
        self._task_costs: dict[str, float] = {}
        self._agent_costs: dict[str, float] = {}
        logger.info("CrewAICostTracker initialized.")

    def on_task_start(self, task_name: str) -> None:
        """Record the start of a CrewAI task.

        Initialises per-task cost tracking if not already present.
        """
        if task_name not in self._task_costs:
            self._task_costs[task_name] = 0.0

    def on_task_end(self, task_name: str, tokens: int) -> dict[str, Any]:
        """Record the end of a CrewAI task and accumulate costs.

        Returns a per-task cost summary.
        """
        price_per_token = 0.000002
        task_cost = tokens * price_per_token
        self.total_tokens += tokens
        self.total_cost_usd += task_cost
        self._task_costs[task_name] = self._task_costs.get(task_name, 0.0) + task_cost
        return {
            "task_name": task_name,
            "tokens": tokens,
            "cost_usd": task_cost,
            "total_cost_usd": self.total_cost_usd,
        }

    def get_crew_cost(self) -> float:
        """Return the cumulative USD cost for the entire crew run."""
        return self.total_cost_usd

    def get_agent_costs(self) -> dict[str, Any]:
        """Return cost breakdown by agent and task."""
        return {
            "agent_costs": dict(self._agent_costs),
            "task_costs": dict(self._task_costs),
            "total_cost_usd": self.total_cost_usd,
        }

    def reset(self) -> None:
        """Reset all cost and token counters to zero."""
        self.total_tokens = 0
        self.total_cost_usd = 0.0
        self._task_costs = {}
        self._agent_costs = {}
        logger.info("CrewAICostTracker reset.")
