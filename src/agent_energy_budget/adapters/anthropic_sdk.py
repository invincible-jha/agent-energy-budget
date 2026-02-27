"""Anthropic SDK adapter for agent_energy_budget."""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

_DEFAULT_PRICING: dict[str, float] = {
    "claude-opus-4-5": 0.000015,
    "claude-sonnet-4-5": 0.000003,
    "claude-haiku-3-5": 0.00000025,
    "claude-3-opus-20240229": 0.000015,
    "claude-3-sonnet-20240229": 0.000003,
}


class AnthropicCostTracker:
    """Cost tracking adapter for the Anthropic SDK.

    Tracks input and output token usage separately and accumulates USD cost
    across Anthropic messages API calls in a session.

    Usage::

        from agent_energy_budget.adapters.anthropic_sdk import AnthropicCostTracker
        tracker = AnthropicCostTracker()
    """

    def __init__(self, model_pricing: dict[str, float] | None = None) -> None:
        self.model_pricing: dict[str, float] = model_pricing if model_pricing is not None else _DEFAULT_PRICING
        self.total_tokens: int = 0
        self.total_cost_usd: float = 0.0
        self._input_tokens: int = 0
        self._output_tokens: int = 0
        self._tool_calls: int = 0
        logger.info("AnthropicCostTracker initialized.")

    def on_message(self, model: str, input_tokens: int, output_tokens: int) -> dict[str, Any]:
        """Record an Anthropic messages API call and accumulate costs.

        Uses a 3:1 output-to-input price ratio if only one price is stored.
        Returns a per-call cost summary.
        """
        price_per_token = self.model_pricing.get(model, 0.000003)
        # Output tokens are typically priced higher; apply a simple 3x multiplier.
        input_cost = input_tokens * price_per_token
        output_cost = output_tokens * price_per_token * 3
        call_cost = input_cost + output_cost
        tokens_total = input_tokens + output_tokens
        self._input_tokens += input_tokens
        self._output_tokens += output_tokens
        self.total_tokens += tokens_total
        self.total_cost_usd += call_cost
        return {
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost_usd": call_cost,
            "total_cost_usd": self.total_cost_usd,
        }

    def on_tool_use(self, tool_name: str) -> None:
        """Record a tool_use event from an Anthropic response.

        Increments the tool call counter for session reporting.
        """
        self._tool_calls += 1
        logger.debug("Anthropic tool use recorded: %s (total: %d)", tool_name, self._tool_calls)

    def get_session_cost(self) -> float:
        """Return the cumulative USD cost for the current session."""
        return self.total_cost_usd

    def reset(self) -> None:
        """Reset all cost, token, and tool call counters to zero."""
        self.total_tokens = 0
        self.total_cost_usd = 0.0
        self._input_tokens = 0
        self._output_tokens = 0
        self._tool_calls = 0
        logger.info("AnthropicCostTracker reset.")
