"""Microsoft Agents adapter for agent_energy_budget."""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

_DEFAULT_PRICING: dict[str, float] = {
    "gpt-4o": 0.000005,
    "gpt-4o-mini": 0.0000006,
    "gpt-4-turbo": 0.00001,
}


class MicrosoftCostTracker:
    """Cost tracking adapter for Microsoft Agents.

    Tracks per-turn token usage and activity counts across a Bot Framework /
    Microsoft Agents conversation.

    Usage::

        from agent_energy_budget.adapters.microsoft_agents import MicrosoftCostTracker
        tracker = MicrosoftCostTracker()
    """

    def __init__(self, model_pricing: dict[str, float] | None = None) -> None:
        self.model_pricing: dict[str, float] = model_pricing if model_pricing is not None else _DEFAULT_PRICING
        self.total_tokens: int = 0
        self.total_cost_usd: float = 0.0
        self._activity_count: int = 0
        logger.info("MicrosoftCostTracker initialized.")

    def on_turn(self, model: str, tokens: int) -> dict[str, Any]:
        """Record a conversation turn and accumulate cost.

        Returns a per-turn cost summary including cumulative totals.
        """
        price_per_token = self.model_pricing.get(model, 0.000002)
        turn_cost = tokens * price_per_token
        self.total_tokens += tokens
        self.total_cost_usd += turn_cost
        return {
            "model": model,
            "tokens": tokens,
            "cost_usd": turn_cost,
            "total_tokens": self.total_tokens,
            "total_cost_usd": self.total_cost_usd,
        }

    def on_activity(self, activity_type: str) -> None:
        """Record a Bot Framework activity.

        Increments the activity counter for reporting.
        """
        self._activity_count += 1
        logger.debug("Activity recorded: %s (total: %d)", activity_type, self._activity_count)

    def get_conversation_cost(self) -> float:
        """Return the cumulative USD cost for the current conversation."""
        return self.total_cost_usd

    def reset(self) -> None:
        """Reset all cost, token, and activity counters to zero."""
        self.total_tokens = 0
        self.total_cost_usd = 0.0
        self._activity_count = 0
        logger.info("MicrosoftCostTracker reset.")
