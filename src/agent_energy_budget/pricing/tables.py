"""Provider pricing tables with per-million-token rates.

All prices are in USD per million tokens as of 2026-02.
Update via PricingUpdater or CustomPricingManager as rates change.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Literal


class ProviderName(str, Enum):
    """Known LLM providers."""

    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    GOOGLE = "google"
    MISTRAL = "mistral"
    META = "meta"
    DEEPSEEK = "deepseek"
    CUSTOM = "custom"


class ModelTier(str, Enum):
    """Quality/cost tier classification for models.

    nano      — ultra-cheap, small context, fast
    efficient — good value, solid quality
    standard  — mainstream flagship-class models
    premium   — top capability, highest cost
    """

    NANO = "nano"
    EFFICIENT = "efficient"
    STANDARD = "standard"
    PREMIUM = "premium"


@dataclass(frozen=True)
class ModelPricing:
    """Per-model pricing in USD per million tokens.

    Parameters
    ----------
    model:
        Canonical model identifier string.
    provider:
        The provider that operates this model.
    tier:
        Quality/cost tier classification.
    input_per_million:
        Cost in USD for one million input (prompt) tokens.
    output_per_million:
        Cost in USD for one million output (completion) tokens.
    context_window:
        Maximum context window in tokens (0 = unknown).
    supports_vision:
        Whether the model accepts image inputs.
    """

    model: str
    provider: ProviderName
    tier: ModelTier
    input_per_million: float
    output_per_million: float
    context_window: int = 0
    supports_vision: bool = False

    def cost_for_tokens(self, input_tokens: int, output_tokens: int) -> float:
        """Calculate the USD cost for a given token count.

        Parameters
        ----------
        input_tokens:
            Number of prompt/input tokens consumed.
        output_tokens:
            Number of completion/output tokens generated.

        Returns
        -------
        float
            Total cost in USD.
        """
        input_cost = (input_tokens / 1_000_000) * self.input_per_million
        output_cost = (output_tokens / 1_000_000) * self.output_per_million
        return round(input_cost + output_cost, 8)

    def max_output_for_budget(self, budget_usd: float, input_tokens: int) -> int:
        """Return maximum output tokens achievable within a USD budget.

        Parameters
        ----------
        budget_usd:
            Available budget in USD.
        input_tokens:
            Input tokens already committed (their cost is subtracted first).

        Returns
        -------
        int
            Maximum output token count, capped at context_window when known.
            Returns 0 if the input cost alone exceeds the budget.
        """
        input_cost = (input_tokens / 1_000_000) * self.input_per_million
        remaining = budget_usd - input_cost
        if remaining <= 0:
            return 0
        max_output = int((remaining / self.output_per_million) * 1_000_000)
        if self.context_window > 0:
            max_output = min(max_output, self.context_window)
        return max_output


# ---------------------------------------------------------------------------
# Provider pricing table — edit rows or add rows here for new models.
# ---------------------------------------------------------------------------

PROVIDER_PRICING: dict[str, ModelPricing] = {
    # ── Anthropic ──────────────────────────────────────────────────────────
    "claude-opus-4": ModelPricing(
        model="claude-opus-4",
        provider=ProviderName.ANTHROPIC,
        tier=ModelTier.PREMIUM,
        input_per_million=15.00,
        output_per_million=75.00,
        context_window=200_000,
        supports_vision=True,
    ),
    "claude-sonnet-4": ModelPricing(
        model="claude-sonnet-4",
        provider=ProviderName.ANTHROPIC,
        tier=ModelTier.STANDARD,
        input_per_million=3.00,
        output_per_million=15.00,
        context_window=200_000,
        supports_vision=True,
    ),
    "claude-haiku-4": ModelPricing(
        model="claude-haiku-4",
        provider=ProviderName.ANTHROPIC,
        tier=ModelTier.EFFICIENT,
        input_per_million=0.80,
        output_per_million=4.00,
        context_window=200_000,
        supports_vision=True,
    ),
    # ── OpenAI ─────────────────────────────────────────────────────────────
    "gpt-4o": ModelPricing(
        model="gpt-4o",
        provider=ProviderName.OPENAI,
        tier=ModelTier.STANDARD,
        input_per_million=2.50,
        output_per_million=10.00,
        context_window=128_000,
        supports_vision=True,
    ),
    "gpt-4o-mini": ModelPricing(
        model="gpt-4o-mini",
        provider=ProviderName.OPENAI,
        tier=ModelTier.EFFICIENT,
        input_per_million=0.15,
        output_per_million=0.60,
        context_window=128_000,
        supports_vision=True,
    ),
    "o3-mini": ModelPricing(
        model="o3-mini",
        provider=ProviderName.OPENAI,
        tier=ModelTier.STANDARD,
        input_per_million=1.10,
        output_per_million=4.40,
        context_window=128_000,
        supports_vision=False,
    ),
    # ── Google ─────────────────────────────────────────────────────────────
    "gemini-2.0-flash": ModelPricing(
        model="gemini-2.0-flash",
        provider=ProviderName.GOOGLE,
        tier=ModelTier.EFFICIENT,
        input_per_million=0.10,
        output_per_million=0.40,
        context_window=1_000_000,
        supports_vision=True,
    ),
    "gemini-2.5-pro": ModelPricing(
        model="gemini-2.5-pro",
        provider=ProviderName.GOOGLE,
        tier=ModelTier.PREMIUM,
        input_per_million=3.50,
        output_per_million=10.50,
        context_window=1_000_000,
        supports_vision=True,
    ),
    # ── Mistral ────────────────────────────────────────────────────────────
    "mistral-large": ModelPricing(
        model="mistral-large",
        provider=ProviderName.MISTRAL,
        tier=ModelTier.STANDARD,
        input_per_million=2.00,
        output_per_million=6.00,
        context_window=128_000,
        supports_vision=False,
    ),
    "mistral-small": ModelPricing(
        model="mistral-small",
        provider=ProviderName.MISTRAL,
        tier=ModelTier.EFFICIENT,
        input_per_million=0.20,
        output_per_million=0.60,
        context_window=32_000,
        supports_vision=False,
    ),
    # ── Open-source / hosted ───────────────────────────────────────────────
    "llama-3.3-70b": ModelPricing(
        model="llama-3.3-70b",
        provider=ProviderName.META,
        tier=ModelTier.EFFICIENT,
        input_per_million=0.59,
        output_per_million=0.79,
        context_window=128_000,
        supports_vision=False,
    ),
    "deepseek-v3": ModelPricing(
        model="deepseek-v3",
        provider=ProviderName.DEEPSEEK,
        tier=ModelTier.EFFICIENT,
        input_per_million=0.27,
        output_per_million=1.10,
        context_window=64_000,
        supports_vision=False,
    ),
}

# ---------------------------------------------------------------------------
# Tier groupings — ordered best-to-cheapest within each tier.
# ---------------------------------------------------------------------------

MODEL_TIERS: dict[ModelTier, list[str]] = {
    ModelTier.PREMIUM: ["claude-opus-4", "gemini-2.5-pro"],
    ModelTier.STANDARD: ["claude-sonnet-4", "gpt-4o", "o3-mini", "mistral-large"],
    ModelTier.EFFICIENT: [
        "claude-haiku-4",
        "gpt-4o-mini",
        "gemini-2.0-flash",
        "mistral-small",
        "llama-3.3-70b",
        "deepseek-v3",
    ],
    ModelTier.NANO: [],  # Reserved for sub-1B parameter models
}

# Ordered tier list from cheapest to most expensive (used in downgrade logic)
_TIER_ORDER: list[ModelTier] = [
    ModelTier.NANO,
    ModelTier.EFFICIENT,
    ModelTier.STANDARD,
    ModelTier.PREMIUM,
]

# Alias list for fuzzy model name resolution
_MODEL_ALIASES: dict[str, str] = {
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


def get_pricing(model: str) -> ModelPricing:
    """Resolve a model name to its pricing record with fuzzy matching.

    Lookup order:
    1. Exact key match in PROVIDER_PRICING.
    2. Alias table match.
    3. Prefix/substring scan across all known model keys.

    Parameters
    ----------
    model:
        Model identifier string, potentially using short or alternate names.

    Returns
    -------
    ModelPricing
        The resolved pricing record.

    Raises
    ------
    KeyError
        If no model can be resolved from the provided string.
    """
    normalised = model.strip().lower()

    # Exact match
    if normalised in PROVIDER_PRICING:
        return PROVIDER_PRICING[normalised]

    # Alias table
    if normalised in _MODEL_ALIASES:
        return PROVIDER_PRICING[_MODEL_ALIASES[normalised]]

    # Prefix/substring scan (first match wins)
    for key in PROVIDER_PRICING:
        if key.startswith(normalised) or normalised in key:
            return PROVIDER_PRICING[key]

    raise KeyError(
        f"No pricing found for model {model!r}. "
        f"Known models: {sorted(PROVIDER_PRICING.keys())}"
    )


def models_by_tier(
    tier: ModelTier,
    *,
    custom_pricing: dict[str, ModelPricing] | None = None,
) -> list[ModelPricing]:
    """Return all ModelPricing records for a given tier, cheapest first.

    Parameters
    ----------
    tier:
        The tier to filter by.
    custom_pricing:
        Optional extra pricing entries to include in the search.

    Returns
    -------
    list[ModelPricing]
        Pricing records sorted ascending by combined input+output cost.
    """
    combined: dict[str, ModelPricing] = {**PROVIDER_PRICING}
    if custom_pricing:
        combined.update(custom_pricing)

    matches = [p for p in combined.values() if p.tier == tier]
    return sorted(matches, key=lambda p: p.input_per_million + p.output_per_million)


def cheapest_model_within_budget(
    budget_usd: float,
    input_tokens: int,
    output_tokens: int,
    *,
    preferred_tier: ModelTier | None = None,
    custom_pricing: dict[str, ModelPricing] | None = None,
) -> ModelPricing | None:
    """Find the highest-quality model that fits within the given budget.

    Searches tiers from cheapest to most expensive. Within each tier the
    model with the best (lowest) combined rate that still fits is returned.

    Parameters
    ----------
    budget_usd:
        Available budget in USD.
    input_tokens:
        Number of input tokens for the proposed call.
    output_tokens:
        Number of output tokens for the proposed call.
    preferred_tier:
        Start search from this tier (skip cheaper tiers).
    custom_pricing:
        Optional custom entries to include.

    Returns
    -------
    ModelPricing | None
        The best affordable model, or None if nothing fits.
    """
    combined: dict[str, ModelPricing] = {**PROVIDER_PRICING}
    if custom_pricing:
        combined.update(custom_pricing)

    start_index = 0
    if preferred_tier is not None and preferred_tier in _TIER_ORDER:
        start_index = _TIER_ORDER.index(preferred_tier)

    best: ModelPricing | None = None
    for tier in _TIER_ORDER[start_index:]:
        candidates = [p for p in combined.values() if p.tier == tier]
        candidates.sort(key=lambda p: p.input_per_million + p.output_per_million)
        for candidate in candidates:
            cost = candidate.cost_for_tokens(input_tokens, output_tokens)
            if cost <= budget_usd:
                best = candidate  # keep upgrading within affordable tiers
    return best


# Re-export tier order for other modules that need it
TIER_ORDER: list[ModelTier] = _TIER_ORDER
