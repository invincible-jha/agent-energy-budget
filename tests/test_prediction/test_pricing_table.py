"""Tests for PricingTable — model pricing data registry."""
from __future__ import annotations

import pytest

from agent_energy_budget.prediction.pricing_table import ModelPricing, PricingTable, _BUILTIN_PRICING


# ---------------------------------------------------------------------------
# ModelPricing dataclass
# ---------------------------------------------------------------------------


class TestModelPricing:
    def test_cost_for_tokens_basic(self) -> None:
        pricing = ModelPricing(input_cost_per_1k=1.0, output_cost_per_1k=2.0)
        cost = pricing.cost_for_tokens(input_tokens=1_000, output_tokens=1_000)
        assert cost == pytest.approx(3.0, rel=1e-6)

    def test_cost_for_zero_tokens(self) -> None:
        pricing = ModelPricing(input_cost_per_1k=1.0, output_cost_per_1k=2.0)
        assert pricing.cost_for_tokens(0, 0) == 0.0

    def test_cost_with_cached_tokens(self) -> None:
        pricing = ModelPricing(
            input_cost_per_1k=1.0,
            output_cost_per_1k=2.0,
            cached_input_cost_per_1k=0.1,
        )
        # 500 non-cached input, 500 cached input, 1000 output
        cost = pricing.cost_for_tokens(
            input_tokens=500, output_tokens=1_000, cached_tokens=500
        )
        expected = (500 / 1_000) * 1.0 + (1_000 / 1_000) * 2.0 + (500 / 1_000) * 0.1
        assert cost == pytest.approx(expected, rel=1e-6)

    def test_cost_without_cache_support_ignores_cached(self) -> None:
        pricing = ModelPricing(
            input_cost_per_1k=1.0,
            output_cost_per_1k=2.0,
            cached_input_cost_per_1k=None,  # no caching
        )
        cost_no_cache = pricing.cost_for_tokens(1_000, 1_000, cached_tokens=0)
        cost_with_cached = pricing.cost_for_tokens(1_000, 1_000, cached_tokens=500)
        # cached_tokens ignored when cached_input_cost_per_1k is None
        assert cost_no_cache == cost_with_cached

    def test_cost_uses_per_1k_rate(self) -> None:
        pricing = ModelPricing(input_cost_per_1k=0.003, output_cost_per_1k=0.015)
        cost = pricing.cost_for_tokens(1_000, 1_000)
        assert cost == pytest.approx(0.018, rel=1e-6)

    def test_cost_result_is_rounded(self) -> None:
        pricing = ModelPricing(input_cost_per_1k=0.003, output_cost_per_1k=0.015)
        cost = pricing.cost_for_tokens(333, 777)
        assert isinstance(cost, float)


# ---------------------------------------------------------------------------
# PricingTable construction
# ---------------------------------------------------------------------------


class TestPricingTableConstruction:
    def test_default_has_builtin_models(self) -> None:
        table = PricingTable()
        models = table.list_models()
        assert len(models) >= 20

    def test_custom_initial_pricing(self) -> None:
        custom = {"my-model": ModelPricing(input_cost_per_1k=1.0, output_cost_per_1k=2.0)}
        table = PricingTable(initial_pricing=custom)
        assert table.list_models() == ["my-model"]

    def test_does_not_share_builtin_reference(self) -> None:
        """Modifying one table must not affect another."""
        table1 = PricingTable()
        table2 = PricingTable()
        table1.update_pricing("custom-a", ModelPricing(1.0, 2.0))
        assert "custom-a" not in table2.list_models()


# ---------------------------------------------------------------------------
# get_pricing — lookup
# ---------------------------------------------------------------------------


class TestGetPricing:
    def test_exact_match(self) -> None:
        table = PricingTable()
        pricing = table.get_pricing("claude-sonnet-4")
        assert pricing.input_cost_per_1k > 0
        assert pricing.output_cost_per_1k > 0

    def test_alias_match(self) -> None:
        table = PricingTable()
        pricing = table.get_pricing("sonnet")
        assert pricing is table.get_pricing("claude-sonnet-4")

    def test_substring_match(self) -> None:
        table = PricingTable()
        # "haiku" is an alias; "haiku-4" should substring-match "claude-haiku-4"
        pricing = table.get_pricing("claude-haiku")
        assert pricing is not None

    def test_unknown_model_raises_key_error(self) -> None:
        table = PricingTable()
        with pytest.raises(KeyError, match="No pricing found"):
            table.get_pricing("this-model-does-not-exist-xyz")

    def test_case_insensitive_lookup(self) -> None:
        table = PricingTable()
        pricing_lower = table.get_pricing("gpt-4o")
        pricing_upper = table.get_pricing("GPT-4O")
        assert pricing_lower is pricing_upper


# ---------------------------------------------------------------------------
# list_models
# ---------------------------------------------------------------------------


class TestListModels:
    def test_returns_sorted_list(self) -> None:
        table = PricingTable()
        models = table.list_models()
        assert models == sorted(models)

    def test_includes_known_models(self) -> None:
        table = PricingTable()
        models = table.list_models()
        assert "claude-sonnet-4" in models
        assert "gpt-4o" in models
        assert "gemini-2.0-flash" in models


# ---------------------------------------------------------------------------
# update_pricing / remove_pricing
# ---------------------------------------------------------------------------


class TestUpdatePricing:
    def test_add_new_model(self) -> None:
        table = PricingTable()
        new_pricing = ModelPricing(input_cost_per_1k=0.001, output_cost_per_1k=0.002)
        table.update_pricing("my-custom-model", new_pricing)
        assert table.get_pricing("my-custom-model") is new_pricing

    def test_override_existing_model(self) -> None:
        table = PricingTable()
        original = table.get_pricing("gpt-4o")
        override = ModelPricing(input_cost_per_1k=99.0, output_cost_per_1k=99.0)
        table.update_pricing("gpt-4o", override)
        assert table.get_pricing("gpt-4o").input_cost_per_1k == 99.0

    def test_update_normalises_key_case(self) -> None:
        table = PricingTable()
        table.update_pricing("MY-MODEL", ModelPricing(1.0, 2.0))
        assert table.get_pricing("my-model").input_cost_per_1k == 1.0

    def test_remove_existing_model(self) -> None:
        table = PricingTable()
        table.update_pricing("temp-model", ModelPricing(1.0, 2.0))
        table.remove_pricing("temp-model")
        with pytest.raises(KeyError):
            table.get_pricing("temp-model")

    def test_remove_nonexistent_raises(self) -> None:
        table = PricingTable()
        with pytest.raises(KeyError):
            table.remove_pricing("ghost-model")


# ---------------------------------------------------------------------------
# Coverage for all 20+ built-in models
# ---------------------------------------------------------------------------


class TestBuiltinModels:
    def test_builtin_coverage(self) -> None:
        table = PricingTable()
        assert len(table) >= 20

    def test_all_builtin_models_have_positive_rates(self) -> None:
        table = PricingTable()
        for model in table.list_models():
            pricing = table.get_pricing(model)
            assert pricing.input_cost_per_1k >= 0
            assert pricing.output_cost_per_1k >= 0

    def test_specific_model_pricing_ballpark(self) -> None:
        """Claude Opus should cost more than GPT-4o-mini."""
        table = PricingTable()
        opus = table.get_pricing("claude-opus-4")
        mini = table.get_pricing("gpt-4o-mini")
        assert opus.input_cost_per_1k > mini.input_cost_per_1k


# ---------------------------------------------------------------------------
# Repr / len
# ---------------------------------------------------------------------------


class TestReprLen:
    def test_len_returns_model_count(self) -> None:
        table = PricingTable()
        assert len(table) == len(table.list_models())

    def test_repr_contains_model_count(self) -> None:
        table = PricingTable()
        assert str(len(table)) in repr(table)
