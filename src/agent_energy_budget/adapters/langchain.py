"""LangChain adapter for agent_energy_budget."""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

_DEFAULT_PRICING: dict[str, float] = {
    "gpt-4o": 0.000005,
    "gpt-4o-mini": 0.0000006,
    "gpt-4-turbo": 0.00001,
    "gpt-3.5-turbo": 0.0000005,
}


class LangChainCostTracker:
    """Cost tracking adapter for LangChain.

    Tracks token usage and accumulated USD cost across LangChain LLM calls.

    Usage::

        from agent_energy_budget.adapters.langchain import LangChainCostTracker
        tracker = LangChainCostTracker()
    """

    def __init__(self, model_pricing: dict[str, float] | None = None) -> None:
        self.model_pricing: dict[str, float] = model_pricing if model_pricing is not None else _DEFAULT_PRICING
        self.total_tokens: int = 0
        self.total_cost_usd: float = 0.0
        self._token_usage_by_model: dict[str, int] = {}
        logger.info("LangChainCostTracker initialized.")

    def on_llm_start(self, model: str) -> None:
        """Record the start of an LLM call.

        Ensures the model is tracked in token usage map.
        """
        if model not in self._token_usage_by_model:
            self._token_usage_by_model[model] = 0

    def on_llm_end(self, response: Any, tokens: int) -> dict[str, Any]:
        """Record the end of an LLM call and accumulate costs.

        Returns a summary of tokens used and cost incurred for this call.
        """
        model_str = str(response) if response is not None else "unknown"
        price_per_token = self.model_pricing.get(model_str, 0.000002)
        call_cost = tokens * price_per_token
        self.total_tokens += tokens
        self.total_cost_usd += call_cost
        self._token_usage_by_model[model_str] = (
            self._token_usage_by_model.get(model_str, 0) + tokens
        )
        return {
            "tokens": tokens,
            "cost_usd": call_cost,
            "total_tokens": self.total_tokens,
            "total_cost_usd": self.total_cost_usd,
        }

    def get_total_cost(self) -> float:
        """Return cumulative USD cost across all LLM calls."""
        return self.total_cost_usd

    def get_token_usage(self) -> dict[str, Any]:
        """Return token usage breakdown by model and the overall total."""
        return {
            "total_tokens": self.total_tokens,
            "by_model": dict(self._token_usage_by_model),
        }

    def reset(self) -> None:
        """Reset all cost and token counters to zero."""
        self.total_tokens = 0
        self.total_cost_usd = 0.0
        self._token_usage_by_model = {}
        logger.info("LangChainCostTracker reset.")
