"""PricingTable — model pricing data for pre-execution cost prediction.

This module provides a lightweight PricingTable wrapping per-1K-token
pricing for 20+ models. It is intentionally separate from the existing
``pricing.tables`` module (which uses per-million-token rates) so that
the prediction subsystem has a self-contained, independently testable
unit with a simple cost model.

All prices are approximate USD values as of 2026-02.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelPricing:
    """Per-model pricing in USD per 1,000 tokens.

    Parameters
    ----------
    input_cost_per_1k:
        Cost in USD for 1,000 input (prompt) tokens.
    output_cost_per_1k:
        Cost in USD for 1,000 output (completion) tokens.
    cached_input_cost_per_1k:
        Cost in USD for 1,000 cached input tokens, or None if caching
        is not supported / not applicable.
    """

    input_cost_per_1k: float
    output_cost_per_1k: float
    cached_input_cost_per_1k: Optional[float] = None

    def cost_for_tokens(
        self,
        input_tokens: int,
        output_tokens: int,
        cached_tokens: int = 0,
    ) -> float:
        """Calculate the USD cost for a given token combination.

        Parameters
        ----------
        input_tokens:
            Non-cached input token count.
        output_tokens:
            Output token count.
        cached_tokens:
            Cached input token count (0 when caching is not used).

        Returns
        -------
        float
            Total cost in USD.
        """
        input_cost = (input_tokens / 1_000) * self.input_cost_per_1k
        output_cost = (output_tokens / 1_000) * self.output_cost_per_1k
        cached_cost = 0.0
        if cached_tokens > 0 and self.cached_input_cost_per_1k is not None:
            cached_cost = (cached_tokens / 1_000) * self.cached_input_cost_per_1k
        return round(input_cost + output_cost + cached_cost, 8)


# ---------------------------------------------------------------------------
# Built-in pricing data — approximate USD / 1K tokens as of 2026-02
# ---------------------------------------------------------------------------

_BUILTIN_PRICING: dict[str, ModelPricing] = {
    # ── Anthropic ──────────────────────────────────────────────────────────
    "claude-opus-4": ModelPricing(
        input_cost_per_1k=0.015,
        output_cost_per_1k=0.075,
        cached_input_cost_per_1k=0.0015,
    ),
    "claude-sonnet-4": ModelPricing(
        input_cost_per_1k=0.003,
        output_cost_per_1k=0.015,
        cached_input_cost_per_1k=0.0003,
    ),
    "claude-haiku-4": ModelPricing(
        input_cost_per_1k=0.0008,
        output_cost_per_1k=0.004,
        cached_input_cost_per_1k=0.00008,
    ),
    "claude-opus-3-5": ModelPricing(
        input_cost_per_1k=0.015,
        output_cost_per_1k=0.075,
        cached_input_cost_per_1k=0.0015,
    ),
    "claude-sonnet-3-5": ModelPricing(
        input_cost_per_1k=0.003,
        output_cost_per_1k=0.015,
        cached_input_cost_per_1k=0.0003,
    ),
    # ── OpenAI ─────────────────────────────────────────────────────────────
    "gpt-4o": ModelPricing(
        input_cost_per_1k=0.0025,
        output_cost_per_1k=0.010,
        cached_input_cost_per_1k=0.00125,
    ),
    "gpt-4o-mini": ModelPricing(
        input_cost_per_1k=0.00015,
        output_cost_per_1k=0.0006,
        cached_input_cost_per_1k=0.000075,
    ),
    "gpt-4-turbo": ModelPricing(
        input_cost_per_1k=0.010,
        output_cost_per_1k=0.030,
    ),
    "gpt-3.5-turbo": ModelPricing(
        input_cost_per_1k=0.0005,
        output_cost_per_1k=0.0015,
    ),
    "o3": ModelPricing(
        input_cost_per_1k=0.010,
        output_cost_per_1k=0.040,
    ),
    "o3-mini": ModelPricing(
        input_cost_per_1k=0.0011,
        output_cost_per_1k=0.0044,
    ),
    "o1": ModelPricing(
        input_cost_per_1k=0.015,
        output_cost_per_1k=0.060,
    ),
    # ── Google ─────────────────────────────────────────────────────────────
    "gemini-2.0-flash": ModelPricing(
        input_cost_per_1k=0.0001,
        output_cost_per_1k=0.0004,
    ),
    "gemini-2.5-pro": ModelPricing(
        input_cost_per_1k=0.0035,
        output_cost_per_1k=0.0105,
    ),
    "gemini-1.5-flash": ModelPricing(
        input_cost_per_1k=0.000075,
        output_cost_per_1k=0.0003,
    ),
    "gemini-1.5-pro": ModelPricing(
        input_cost_per_1k=0.00125,
        output_cost_per_1k=0.005,
    ),
    # ── Mistral ────────────────────────────────────────────────────────────
    "mistral-large": ModelPricing(
        input_cost_per_1k=0.002,
        output_cost_per_1k=0.006,
    ),
    "mistral-small": ModelPricing(
        input_cost_per_1k=0.0002,
        output_cost_per_1k=0.0006,
    ),
    "mixtral-8x7b": ModelPricing(
        input_cost_per_1k=0.0006,
        output_cost_per_1k=0.0006,
    ),
    # ── Open-source / hosted ───────────────────────────────────────────────
    "llama-3.3-70b": ModelPricing(
        input_cost_per_1k=0.00059,
        output_cost_per_1k=0.00079,
    ),
    "llama-3.1-8b": ModelPricing(
        input_cost_per_1k=0.00018,
        output_cost_per_1k=0.00018,
    ),
    "deepseek-v3": ModelPricing(
        input_cost_per_1k=0.00027,
        output_cost_per_1k=0.0011,
    ),
    "deepseek-r1": ModelPricing(
        input_cost_per_1k=0.00055,
        output_cost_per_1k=0.00219,
    ),
}

# Aliases for common short names
_PRICING_ALIASES: dict[str, str] = {
    "claude-opus": "claude-opus-4",
    "opus": "claude-opus-4",
    "claude-sonnet": "claude-sonnet-4",
    "sonnet": "claude-sonnet-4",
    "claude-haiku": "claude-haiku-4",
    "haiku": "claude-haiku-4",
    "gpt4o": "gpt-4o",
    "gpt-4o-2024": "gpt-4o",
    "gpt4o-mini": "gpt-4o-mini",
    "flash": "gemini-2.0-flash",
    "gemini-flash": "gemini-2.0-flash",
    "gemini-pro": "gemini-2.5-pro",
    "mistral-large-latest": "mistral-large",
    "mistral-small-latest": "mistral-small",
    "llama3": "llama-3.3-70b",
    "llama-3": "llama-3.3-70b",
    "deepseek": "deepseek-v3",
}


class PricingTable:
    """Registry of model pricing data supporting runtime updates.

    Wraps a mutable copy of the built-in pricing table so callers can
    add or override pricing without affecting the module-level constants.

    Parameters
    ----------
    initial_pricing:
        Optional mapping of model name → ModelPricing to use as the
        starting dataset. Defaults to the built-in table.
    """

    def __init__(
        self,
        initial_pricing: dict[str, ModelPricing] | None = None,
    ) -> None:
        self._pricing: dict[str, ModelPricing] = dict(
            initial_pricing if initial_pricing is not None else _BUILTIN_PRICING
        )

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def get_pricing(self, model: str) -> ModelPricing:
        """Retrieve pricing for *model* with alias resolution.

        Lookup order:
        1. Exact match.
        2. Alias table match.
        3. Prefix/substring scan.

        Parameters
        ----------
        model:
            Model identifier string (exact or short-form alias).

        Returns
        -------
        ModelPricing

        Raises
        ------
        KeyError
            If no pricing record can be found for *model*.
        """
        normalised = model.strip().lower()

        if normalised in self._pricing:
            return self._pricing[normalised]

        resolved = _PRICING_ALIASES.get(normalised)
        if resolved and resolved in self._pricing:
            return self._pricing[resolved]

        # Prefix / substring scan
        for key in self._pricing:
            if key.startswith(normalised) or normalised in key:
                return self._pricing[key]

        raise KeyError(
            f"No pricing found for model {model!r}. "
            f"Known models: {sorted(self._pricing.keys())}"
        )

    def list_models(self) -> list[str]:
        """Return sorted list of all model identifiers in this table.

        Returns
        -------
        list[str]
        """
        return sorted(self._pricing.keys())

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def update_pricing(self, model: str, pricing: ModelPricing) -> None:
        """Add or replace pricing for *model*.

        Parameters
        ----------
        model:
            Model identifier string (used as the canonical key).
        pricing:
            New pricing record.
        """
        self._pricing[model.strip().lower()] = pricing

    def remove_pricing(self, model: str) -> None:
        """Remove pricing for *model* from the table.

        Parameters
        ----------
        model:
            Model identifier string.

        Raises
        ------
        KeyError
            If the model is not in the table.
        """
        normalised = model.strip().lower()
        if normalised not in self._pricing:
            raise KeyError(f"Model {model!r} not found in pricing table")
        del self._pricing[normalised]

    def __len__(self) -> int:
        return len(self._pricing)

    def __repr__(self) -> str:
        return f"PricingTable(models={len(self._pricing)})"
