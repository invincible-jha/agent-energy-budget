"""Unit tests for agent_energy_budget.pricing.tables."""
from __future__ import annotations

import pytest

from agent_energy_budget.pricing.tables import (
    MODEL_TIERS,
    PROVIDER_PRICING,
    TIER_ORDER,
    ModelPricing,
    ModelTier,
    ProviderName,
    cheapest_model_within_budget,
    get_pricing,
    models_by_tier,
)


# ---------------------------------------------------------------------------
# ModelPricing dataclass
# ---------------------------------------------------------------------------


class TestModelPricingCostForTokens:
    def test_zero_tokens_returns_zero(self) -> None:
        pricing = PROVIDER_PRICING["claude-haiku-4"]
        assert pricing.cost_for_tokens(0, 0) == 0.0

    def test_one_million_input_tokens_costs_input_rate(self) -> None:
        pricing = PROVIDER_PRICING["claude-haiku-4"]
        cost = pricing.cost_for_tokens(1_000_000, 0)
        assert cost == pytest.approx(pricing.input_per_million, rel=1e-6)

    def test_one_million_output_tokens_costs_output_rate(self) -> None:
        pricing = PROVIDER_PRICING["claude-haiku-4"]
        cost = pricing.cost_for_tokens(0, 1_000_000)
        assert cost == pytest.approx(pricing.output_per_million, rel=1e-6)

    def test_combined_input_and_output_cost(self) -> None:
        pricing = ModelPricing(
            model="test-model",
            provider=ProviderName.CUSTOM,
            tier=ModelTier.EFFICIENT,
            input_per_million=2.0,
            output_per_million=6.0,
        )
        # 500_000 input = $1.00, 1_000_000 output = $6.00 => $7.00
        cost = pricing.cost_for_tokens(500_000, 1_000_000)
        assert cost == pytest.approx(7.0, rel=1e-6)

    def test_cost_rounded_to_eight_decimals(self) -> None:
        pricing = PROVIDER_PRICING["gpt-4o-mini"]
        cost = pricing.cost_for_tokens(100, 100)
        # Verify it is representable as float with 8 decimal precision
        assert round(cost, 8) == cost

    def test_claude_opus_premium_pricing(self) -> None:
        pricing = PROVIDER_PRICING["claude-opus-4"]
        assert pricing.input_per_million == 15.0
        assert pricing.output_per_million == 75.0

    def test_small_token_counts_produce_positive_cost(self) -> None:
        pricing = PROVIDER_PRICING["gpt-4o"]
        cost = pricing.cost_for_tokens(10, 10)
        assert cost > 0.0


class TestModelPricingMaxOutputForBudget:
    def test_zero_budget_returns_zero(self) -> None:
        pricing = PROVIDER_PRICING["claude-haiku-4"]
        result = pricing.max_output_for_budget(0.0, 0)
        assert result == 0

    def test_budget_less_than_input_cost_returns_zero(self) -> None:
        pricing = PROVIDER_PRICING["claude-opus-4"]
        # 1M input tokens at $15/M = $15 cost; budget is $1
        result = pricing.max_output_for_budget(1.0, 1_000_000)
        assert result == 0

    def test_large_budget_allows_many_output_tokens(self) -> None:
        pricing = PROVIDER_PRICING["gpt-4o-mini"]
        result = pricing.max_output_for_budget(100.0, 0)
        assert result > 100_000

    def test_context_window_caps_output(self) -> None:
        pricing = PROVIDER_PRICING["mistral-small"]  # context_window=32_000
        result = pricing.max_output_for_budget(1000.0, 0)
        assert result == pricing.context_window

    def test_model_without_context_window_is_uncapped(self) -> None:
        pricing = ModelPricing(
            model="no-cap-model",
            provider=ProviderName.CUSTOM,
            tier=ModelTier.NANO,
            input_per_million=0.01,
            output_per_million=0.01,
            context_window=0,  # unknown
        )
        result = pricing.max_output_for_budget(100.0, 0)
        # With context_window=0, no cap is applied
        assert result > 1_000_000


# ---------------------------------------------------------------------------
# PROVIDER_PRICING table structure
# ---------------------------------------------------------------------------


class TestProviderPricingTable:
    def test_table_is_non_empty(self) -> None:
        assert len(PROVIDER_PRICING) > 0

    def test_all_entries_are_model_pricing_instances(self) -> None:
        for key, value in PROVIDER_PRICING.items():
            assert isinstance(value, ModelPricing), f"Entry {key!r} is not ModelPricing"

    def test_all_model_keys_match_model_field(self) -> None:
        for key, pricing in PROVIDER_PRICING.items():
            assert pricing.model == key

    def test_anthropic_models_present(self) -> None:
        assert "claude-opus-4" in PROVIDER_PRICING
        assert "claude-sonnet-4" in PROVIDER_PRICING
        assert "claude-haiku-4" in PROVIDER_PRICING

    def test_openai_models_present(self) -> None:
        assert "gpt-4o" in PROVIDER_PRICING
        assert "gpt-4o-mini" in PROVIDER_PRICING

    def test_google_models_present(self) -> None:
        assert "gemini-2.0-flash" in PROVIDER_PRICING
        assert "gemini-2.5-pro" in PROVIDER_PRICING

    def test_premium_models_have_highest_prices(self) -> None:
        premium = [p for p in PROVIDER_PRICING.values() if p.tier == ModelTier.PREMIUM]
        efficient = [p for p in PROVIDER_PRICING.values() if p.tier == ModelTier.EFFICIENT]
        # All premium input rates should exceed all efficient input rates
        if premium and efficient:
            min_premium = min(p.input_per_million for p in premium)
            max_efficient = max(p.input_per_million for p in efficient)
            assert min_premium > max_efficient

    def test_all_models_have_positive_rates(self) -> None:
        for pricing in PROVIDER_PRICING.values():
            assert pricing.input_per_million > 0
            assert pricing.output_per_million > 0

    def test_vision_models_flag(self) -> None:
        assert PROVIDER_PRICING["claude-opus-4"].supports_vision is True
        assert PROVIDER_PRICING["o3-mini"].supports_vision is False


# ---------------------------------------------------------------------------
# get_pricing
# ---------------------------------------------------------------------------


class TestGetPricing:
    def test_exact_key_match(self) -> None:
        pricing = get_pricing("claude-haiku-4")
        assert pricing.model == "claude-haiku-4"

    def test_alias_resolution_sonnet(self) -> None:
        pricing = get_pricing("sonnet")
        assert pricing.model == "claude-sonnet-4"

    def test_alias_resolution_haiku(self) -> None:
        pricing = get_pricing("haiku")
        assert pricing.model == "claude-haiku-4"

    def test_alias_resolution_opus(self) -> None:
        pricing = get_pricing("opus")
        assert pricing.model == "claude-opus-4"

    def test_alias_resolution_flash(self) -> None:
        pricing = get_pricing("flash")
        assert pricing.model == "gemini-2.0-flash"

    def test_alias_gpt4o(self) -> None:
        pricing = get_pricing("gpt4o")
        assert pricing.model == "gpt-4o"

    def test_whitespace_stripped(self) -> None:
        pricing = get_pricing("  claude-haiku-4  ")
        assert pricing.model == "claude-haiku-4"

    def test_case_insensitive(self) -> None:
        pricing = get_pricing("CLAUDE-HAIKU-4")
        assert pricing.model == "claude-haiku-4"

    def test_unknown_model_raises_key_error(self) -> None:
        with pytest.raises(KeyError):
            get_pricing("totally-unknown-model-xyz")

    def test_substring_scan_finds_partial_match(self) -> None:
        pricing = get_pricing("deepseek")
        assert pricing.model == "deepseek-v3"


# ---------------------------------------------------------------------------
# models_by_tier
# ---------------------------------------------------------------------------


class TestModelsByTier:
    def test_premium_tier_returns_correct_models(self) -> None:
        results = models_by_tier(ModelTier.PREMIUM)
        tiers = {p.tier for p in results}
        assert tiers == {ModelTier.PREMIUM}

    def test_efficient_tier_has_multiple_models(self) -> None:
        results = models_by_tier(ModelTier.EFFICIENT)
        assert len(results) >= 3

    def test_results_sorted_cheapest_first(self) -> None:
        results = models_by_tier(ModelTier.EFFICIENT)
        costs = [p.input_per_million + p.output_per_million for p in results]
        assert costs == sorted(costs)

    def test_custom_pricing_included_when_matching_tier(self) -> None:
        custom_model = ModelPricing(
            model="my-custom-efficient",
            provider=ProviderName.CUSTOM,
            tier=ModelTier.EFFICIENT,
            input_per_million=0.05,
            output_per_million=0.10,
        )
        results = models_by_tier(ModelTier.EFFICIENT, custom_pricing={"my-custom-efficient": custom_model})
        names = [p.model for p in results]
        assert "my-custom-efficient" in names

    def test_nano_tier_is_empty_by_default(self) -> None:
        results = models_by_tier(ModelTier.NANO)
        assert results == []


# ---------------------------------------------------------------------------
# cheapest_model_within_budget
# ---------------------------------------------------------------------------


class TestCheapestModelWithinBudget:
    def test_returns_none_when_no_model_fits(self) -> None:
        result = cheapest_model_within_budget(0.0, 1_000_000, 1_000_000)
        assert result is None

    def test_returns_a_model_with_generous_budget(self) -> None:
        result = cheapest_model_within_budget(100.0, 100, 100)
        assert result is not None

    def test_returned_model_fits_within_budget(self) -> None:
        budget = 0.001
        result = cheapest_model_within_budget(budget, 100, 100)
        if result is not None:
            actual_cost = result.cost_for_tokens(100, 100)
            assert actual_cost <= budget

    def test_preferred_tier_skips_cheaper_tiers(self) -> None:
        result = cheapest_model_within_budget(
            100.0, 100, 100, preferred_tier=ModelTier.STANDARD
        )
        # Should not return any NANO or EFFICIENT tier model
        if result is not None:
            assert result.tier in (ModelTier.STANDARD, ModelTier.PREMIUM)

    def test_custom_pricing_is_considered(self) -> None:
        ultra_cheap = ModelPricing(
            model="ultra-cheap",
            provider=ProviderName.CUSTOM,
            tier=ModelTier.NANO,
            input_per_million=0.001,
            output_per_million=0.001,
        )
        result = cheapest_model_within_budget(
            0.00001, 1, 1, custom_pricing={"ultra-cheap": ultra_cheap}
        )
        assert result is not None

    def test_tier_order_is_correct_ordering(self) -> None:
        assert TIER_ORDER[0] == ModelTier.NANO
        assert TIER_ORDER[-1] == ModelTier.PREMIUM
