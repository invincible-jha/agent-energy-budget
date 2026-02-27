"""Data models for the cost-aware model router.

All types are immutable frozen dataclasses validated at system boundaries.
ModelProfile describes a single LLM's cost, quality, and capability
characteristics. BudgetConfig (router-specific) holds per-route budget
constraints. RoutingDecision captures the output of a routing call.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

RoutingStrategy = Literal["cheapest_first", "quality_first", "balanced", "budget_aware"]
TaskComplexity = Literal["low", "medium", "high"]


# ---------------------------------------------------------------------------
# ModelProfile
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelProfile:
    """Immutable profile describing a single LLM for routing purposes.

    Parameters
    ----------
    name:
        Canonical model identifier (e.g. ``"gpt-4o-mini"``).
    provider:
        Provider name (e.g. ``"openai"``, ``"anthropic"``).
    cost_per_1k_input:
        USD cost per 1 000 input (prompt) tokens.
    cost_per_1k_output:
        USD cost per 1 000 output (completion) tokens.
    quality_score:
        Normalised quality score in [0.0, 1.0]. Higher is better.
    max_context:
        Maximum context window in tokens.
    latency_p50_ms:
        Median end-to-end latency in milliseconds (informational only).

    Examples
    --------
    >>> profile = ModelProfile(
    ...     name="gpt-4o-mini",
    ...     provider="openai",
    ...     cost_per_1k_input=0.00015,
    ...     cost_per_1k_output=0.00060,
    ...     quality_score=0.72,
    ...     max_context=128_000,
    ...     latency_p50_ms=800,
    ... )
    >>> profile.cost_for_tokens(1000, 512)
    0.000458
    """

    name: str
    provider: str
    cost_per_1k_input: float
    cost_per_1k_output: float
    quality_score: float
    max_context: int
    latency_p50_ms: int

    def cost_for_tokens(self, input_tokens: int, output_tokens: int) -> float:
        """Calculate the USD cost for a given number of tokens.

        Parameters
        ----------
        input_tokens:
            Number of prompt/input tokens.
        output_tokens:
            Number of completion/output tokens.

        Returns
        -------
        float
            Total cost in USD rounded to 8 decimal places.
        """
        input_cost = (input_tokens / 1_000) * self.cost_per_1k_input
        output_cost = (output_tokens / 1_000) * self.cost_per_1k_output
        return round(input_cost + output_cost, 8)

    def cost_efficiency_ratio(self) -> float:
        """Return quality divided by combined cost rate (quality per dollar).

        Higher values indicate better value (more quality per unit cost).
        Uses a nominal 1 000 input + 500 output token call for the cost
        denominator so the ratio is comparable across models.

        Returns
        -------
        float
            Quality-per-cost ratio. Returns inf if both costs are zero.
        """
        nominal_cost = self.cost_for_tokens(1_000, 500)
        if nominal_cost == 0.0:
            return float("inf")
        return self.quality_score / nominal_cost

    def __post_init__(self) -> None:
        """Validate field constraints after construction."""
        if not self.name:
            raise ValueError("ModelProfile.name must not be empty.")
        if not 0.0 <= self.quality_score <= 1.0:
            raise ValueError(
                f"ModelProfile.quality_score must be in [0.0, 1.0]; "
                f"got {self.quality_score!r} for model {self.name!r}."
            )
        if self.cost_per_1k_input < 0.0:
            raise ValueError(
                f"ModelProfile.cost_per_1k_input must be >= 0; "
                f"got {self.cost_per_1k_input!r} for model {self.name!r}."
            )
        if self.cost_per_1k_output < 0.0:
            raise ValueError(
                f"ModelProfile.cost_per_1k_output must be >= 0; "
                f"got {self.cost_per_1k_output!r} for model {self.name!r}."
            )
        if self.max_context < 0:
            raise ValueError(
                f"ModelProfile.max_context must be >= 0; "
                f"got {self.max_context!r} for model {self.name!r}."
            )
        if self.latency_p50_ms < 0:
            raise ValueError(
                f"ModelProfile.latency_p50_ms must be >= 0; "
                f"got {self.latency_p50_ms!r} for model {self.name!r}."
            )


# ---------------------------------------------------------------------------
# RouterBudgetConfig
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RouterBudgetConfig:
    """Budget configuration scoped to the router (not the global BudgetTracker).

    Parameters
    ----------
    total_budget_usd:
        Maximum USD available for this routing session or period.
    alert_threshold_pct:
        Percentage of budget consumed at which to switch to degraded routing.
        Must be in [0.0, 100.0].
    min_quality_score:
        Minimum acceptable quality score for any selected model.
        Must be in [0.0, 1.0].

    Examples
    --------
    >>> cfg = RouterBudgetConfig(
    ...     total_budget_usd=10.0,
    ...     alert_threshold_pct=80.0,
    ...     min_quality_score=0.5,
    ... )
    >>> cfg.alert_threshold_fraction
    0.8
    """

    total_budget_usd: float
    alert_threshold_pct: float = 80.0
    min_quality_score: float = 0.0

    def __post_init__(self) -> None:
        """Validate field constraints after construction."""
        if self.total_budget_usd < 0.0:
            raise ValueError(
                f"RouterBudgetConfig.total_budget_usd must be >= 0; "
                f"got {self.total_budget_usd!r}."
            )
        if not 0.0 <= self.alert_threshold_pct <= 100.0:
            raise ValueError(
                f"RouterBudgetConfig.alert_threshold_pct must be in [0.0, 100.0]; "
                f"got {self.alert_threshold_pct!r}."
            )
        if not 0.0 <= self.min_quality_score <= 1.0:
            raise ValueError(
                f"RouterBudgetConfig.min_quality_score must be in [0.0, 1.0]; "
                f"got {self.min_quality_score!r}."
            )

    @property
    def alert_threshold_fraction(self) -> float:
        """Return alert_threshold_pct expressed as a fraction in [0.0, 1.0]."""
        return self.alert_threshold_pct / 100.0


# Keep the public name consistent with the specification (BudgetConfig alias).
BudgetConfig = RouterBudgetConfig


# ---------------------------------------------------------------------------
# RoutingDecision
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RoutingDecision:
    """Immutable result of a single routing call.

    Parameters
    ----------
    selected_model:
        The ModelProfile chosen by the routing strategy.
    reason:
        Human-readable explanation of why this model was selected.
    estimated_cost:
        Estimated USD cost for the routed call (based on a nominal token count
        used during selection; actual cost depends on real token usage).
    remaining_budget:
        Budget remaining after deducting ``estimated_cost``.

    Examples
    --------
    >>> decision = RoutingDecision(
    ...     selected_model=some_profile,
    ...     reason="Lowest cost model within budget.",
    ...     estimated_cost=0.0005,
    ...     remaining_budget=9.9995,
    ... )
    """

    selected_model: ModelProfile
    reason: str
    estimated_cost: float
    remaining_budget: float


# ---------------------------------------------------------------------------
# Catalogue of realistic model profiles (used as defaults)
# ---------------------------------------------------------------------------

#: Pre-built catalogue of realistic model profiles for common providers.
#: Cost rates are expressed in USD per 1 000 tokens (not per million) so that
#: the values match how ModelProfile.cost_for_tokens is defined.
DEFAULT_MODEL_PROFILES: list[ModelProfile] = [
    ModelProfile(
        name="gpt-4o",
        provider="openai",
        cost_per_1k_input=0.0025,   # $2.50 / 1M = $0.0025 / 1K
        cost_per_1k_output=0.01000,  # $10.00 / 1M
        quality_score=0.90,
        max_context=128_000,
        latency_p50_ms=900,
    ),
    ModelProfile(
        name="gpt-4o-mini",
        provider="openai",
        cost_per_1k_input=0.000150,  # $0.15 / 1M
        cost_per_1k_output=0.000600,  # $0.60 / 1M
        quality_score=0.72,
        max_context=128_000,
        latency_p50_ms=500,
    ),
    ModelProfile(
        name="claude-sonnet-4-6",
        provider="anthropic",
        cost_per_1k_input=0.003000,  # $3.00 / 1M
        cost_per_1k_output=0.015000,  # $15.00 / 1M
        quality_score=0.92,
        max_context=200_000,
        latency_p50_ms=1_100,
    ),
    ModelProfile(
        name="claude-haiku-4-5",
        provider="anthropic",
        cost_per_1k_input=0.000800,  # $0.80 / 1M
        cost_per_1k_output=0.004000,  # $4.00 / 1M
        quality_score=0.75,
        max_context=200_000,
        latency_p50_ms=400,
    ),
    ModelProfile(
        name="llama-3-70b",
        provider="meta",
        cost_per_1k_input=0.000590,  # $0.59 / 1M
        cost_per_1k_output=0.000790,  # $0.79 / 1M
        quality_score=0.78,
        max_context=128_000,
        latency_p50_ms=700,
    ),
    ModelProfile(
        name="mistral-7b",
        provider="mistral",
        cost_per_1k_input=0.000200,  # $0.20 / 1M
        cost_per_1k_output=0.000600,  # $0.60 / 1M
        quality_score=0.60,
        max_context=32_000,
        latency_p50_ms=300,
    ),
]
