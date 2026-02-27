"""OpenAI Agents SDK adapter for agent_energy_budget."""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

_DEFAULT_PRICING: dict[str, float] = {
    "gpt-4o": 0.000005,
    "gpt-4o-mini": 0.0000006,
    "gpt-4-turbo": 0.00001,
    "o1": 0.000015,
    "o1-mini": 0.000003,
}


class OpenAICostTracker:
    """Cost tracking adapter for the OpenAI Agents SDK.

    Tracks completion token usage and accumulated USD cost for sessions that
    use the OpenAI Agents SDK runner.

    Usage::

        from agent_energy_budget.adapters.openai_agents import OpenAICostTracker
        tracker = OpenAICostTracker()
    """

    def __init__(self, model_pricing: dict[str, float] | None = None) -> None:
        self.model_pricing: dict[str, float] = model_pricing if model_pricing is not None else _DEFAULT_PRICING
        self.total_tokens: int = 0
        self.total_cost_usd: float = 0.0
        self._tool_calls: int = 0
        logger.info("OpenAICostTracker initialized.")

    def on_completion(self, model: str, tokens: int) -> dict[str, Any]:
        """Record a completion event and accumulate cost.

        Returns a summary of tokens used and cost incurred for this completion.
        """
        price_per_token = self.model_pricing.get(model, 0.000002)
        call_cost = tokens * price_per_token
        self.total_tokens += tokens
        self.total_cost_usd += call_cost
        return {
            "model": model,
            "tokens": tokens,
            "cost_usd": call_cost,
            "total_tokens": self.total_tokens,
            "total_cost_usd": self.total_cost_usd,
        }

    def on_tool_call(self, tool_name: str) -> None:
        """Record that a tool call occurred in this session.

        Increments the tool call counter for reporting purposes.
        """
        self._tool_calls += 1
        logger.debug("Tool call recorded: %s (total: %d)", tool_name, self._tool_calls)

    def get_session_cost(self) -> float:
        """Return the cumulative USD cost for the current session."""
        return self.total_cost_usd

    def reset(self) -> None:
        """Reset all cost, token, and tool call counters to zero."""
        self.total_tokens = 0
        self.total_cost_usd = 0.0
        self._tool_calls = 0
        logger.info("OpenAICostTracker reset.")
