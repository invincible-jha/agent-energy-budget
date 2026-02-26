"""Unit tests for agent_energy_budget.pricing.custom."""
from __future__ import annotations

import json
import pathlib

import pytest

from agent_energy_budget.pricing.custom import CustomPricingManager, VolumeDiscount
from agent_energy_budget.pricing.tables import PROVIDER_PRICING, ModelPricing, ModelTier, ProviderName


# ---------------------------------------------------------------------------
# VolumeDiscount
# ---------------------------------------------------------------------------


class TestVolumeDiscount:
    def _base_pricing(self) -> ModelPricing:
        return ModelPricing(
            model="base-model",
            provider=ProviderName.CUSTOM,
            tier=ModelTier.EFFICIENT,
            input_per_million=2.0,
            output_per_million=6.0,
        )

    def test_discount_not_applied_below_threshold(self) -> None:
        discount = VolumeDiscount(
            model="base-model",
            monthly_token_threshold=1_000_000,
            input_discount_pct=20.0,
            output_discount_pct=20.0,
        )
        pricing = self._base_pricing()
        result = discount.apply_to(pricing, monthly_tokens_used=500_000)
        assert result.input_per_million == pytest.approx(2.0)
        assert result.output_per_million == pytest.approx(6.0)

    def test_discount_applied_at_threshold(self) -> None:
        discount = VolumeDiscount(
            model="base-model",
            monthly_token_threshold=1_000_000,
            input_discount_pct=20.0,
            output_discount_pct=10.0,
        )
        pricing = self._base_pricing()
        result = discount.apply_to(pricing, monthly_tokens_used=1_000_000)
        assert result.input_per_million == pytest.approx(2.0 * 0.80)
        assert result.output_per_million == pytest.approx(6.0 * 0.90)

    def test_discount_applied_above_threshold(self) -> None:
        discount = VolumeDiscount(
            model="base-model",
            monthly_token_threshold=1_000_000,
            input_discount_pct=50.0,
            output_discount_pct=50.0,
        )
        pricing = self._base_pricing()
        result = discount.apply_to(pricing, monthly_tokens_used=2_000_000)
        assert result.input_per_million == pytest.approx(1.0)
        assert result.output_per_million == pytest.approx(3.0)

    def test_hundred_percent_discount_clamps_to_zero(self) -> None:
        discount = VolumeDiscount(
            model="base-model",
            monthly_token_threshold=1,
            input_discount_pct=100.0,
            output_discount_pct=100.0,
        )
        pricing = self._base_pricing()
        result = discount.apply_to(pricing, monthly_tokens_used=10)
        assert result.input_per_million == 0.0
        assert result.output_per_million == 0.0

    def test_discounted_pricing_preserves_other_fields(self) -> None:
        discount = VolumeDiscount(
            model="base-model",
            monthly_token_threshold=1,
            input_discount_pct=10.0,
            output_discount_pct=10.0,
        )
        pricing = self._base_pricing()
        result = discount.apply_to(pricing, monthly_tokens_used=100)
        assert result.model == pricing.model
        assert result.provider == pricing.provider
        assert result.tier == pricing.tier


# ---------------------------------------------------------------------------
# CustomPricingManager
# ---------------------------------------------------------------------------


@pytest.fixture()
def manager() -> CustomPricingManager:
    return CustomPricingManager()


class TestCustomPricingManagerRegistration:
    def test_register_creates_model_pricing(self, manager: CustomPricingManager) -> None:
        pricing = manager.register(
            model="my-fine-tune",
            provider="anthropic",
            tier="efficient",
            input_per_million=0.5,
            output_per_million=1.5,
        )
        assert pricing.model == "my-fine-tune"
        assert pricing.input_per_million == 0.5

    def test_registered_model_retrievable_via_get(self, manager: CustomPricingManager) -> None:
        manager.register(
            model="retrieval-test",
            provider="openai",
            tier="standard",
            input_per_million=2.0,
            output_per_million=6.0,
        )
        result = manager.get("retrieval-test")
        assert result is not None
        assert result.model == "retrieval-test"

    def test_get_unknown_model_returns_none(self, manager: CustomPricingManager) -> None:
        assert manager.get("nonexistent-model") is None

    def test_unknown_provider_defaults_to_custom(self, manager: CustomPricingManager) -> None:
        pricing = manager.register(
            model="weird-provider-model",
            provider="my-unknown-provider",
            tier="nano",
            input_per_million=0.01,
            output_per_million=0.02,
        )
        assert pricing.provider == ProviderName.CUSTOM

    def test_unknown_tier_defaults_to_efficient(self, manager: CustomPricingManager) -> None:
        pricing = manager.register(
            model="weird-tier-model",
            provider="custom",
            tier="ultra-nano",
            input_per_million=0.01,
            output_per_million=0.02,
        )
        assert pricing.tier == ModelTier.EFFICIENT

    def test_list_custom_returns_all_registered(self, manager: CustomPricingManager) -> None:
        manager.register("model-a", "custom", "nano", 0.01, 0.02)
        manager.register("model-b", "custom", "nano", 0.03, 0.04)
        listing = manager.list_custom()
        names = [p.model for p in listing]
        assert "model-a" in names
        assert "model-b" in names

    def test_remove_existing_returns_true(self, manager: CustomPricingManager) -> None:
        manager.register("removable", "custom", "nano", 0.01, 0.02)
        result = manager.remove("removable")
        assert result is True

    def test_remove_nonexistent_returns_false(self, manager: CustomPricingManager) -> None:
        result = manager.remove("ghost-model")
        assert result is False

    def test_remove_makes_get_return_none(self, manager: CustomPricingManager) -> None:
        manager.register("gone", "custom", "nano", 0.01, 0.02)
        manager.remove("gone")
        assert manager.get("gone") is None


class TestCustomPricingManagerVolumeDiscounts:
    def test_discount_applied_when_threshold_met(self, manager: CustomPricingManager) -> None:
        manager.register("discounted-model", "custom", "efficient", 2.0, 6.0)
        discount = VolumeDiscount(
            model="discounted-model",
            monthly_token_threshold=1_000,
            input_discount_pct=25.0,
            output_discount_pct=25.0,
        )
        manager.add_volume_discount(discount)
        effective = manager.get_effective_pricing("discounted-model", monthly_tokens_used=10_000)
        assert effective is not None
        assert effective.input_per_million == pytest.approx(1.5)

    def test_discount_not_applied_below_threshold(self, manager: CustomPricingManager) -> None:
        manager.register("undiscounted-model", "custom", "efficient", 2.0, 6.0)
        discount = VolumeDiscount(
            model="undiscounted-model",
            monthly_token_threshold=1_000_000,
            input_discount_pct=25.0,
            output_discount_pct=25.0,
        )
        manager.add_volume_discount(discount)
        effective = manager.get_effective_pricing("undiscounted-model", monthly_tokens_used=500)
        assert effective is not None
        assert effective.input_per_million == pytest.approx(2.0)

    def test_get_effective_pricing_falls_back_to_provider_table(self, manager: CustomPricingManager) -> None:
        effective = manager.get_effective_pricing("claude-haiku-4")
        assert effective is not None
        assert effective.model == "claude-haiku-4"

    def test_get_effective_pricing_returns_none_for_unknown(self, manager: CustomPricingManager) -> None:
        result = manager.get_effective_pricing("totally-unknown-model-xyz")
        assert result is None


class TestCustomPricingManagerCommitToGlobal:
    def test_commit_adds_to_provider_pricing(self, manager: CustomPricingManager) -> None:
        unique_name = "commit-test-model-xyz"
        manager.register(unique_name, "custom", "nano", 0.001, 0.002)
        committed = manager.commit_to_global()
        assert unique_name in committed
        assert unique_name in PROVIDER_PRICING
        del PROVIDER_PRICING[unique_name]

    def test_commit_overwrite_false_skips_existing(self, manager: CustomPricingManager) -> None:
        original_rate = PROVIDER_PRICING["gpt-4o-mini"].input_per_million
        manager.register("gpt-4o-mini", "openai", "efficient", 9999.0, 9999.0)
        manager.commit_to_global(overwrite_existing=False)
        assert PROVIDER_PRICING["gpt-4o-mini"].input_per_million == original_rate


class TestCustomPricingManagerPersistence:
    def test_save_and_reload_from_disk(self, tmp_path: pathlib.Path) -> None:
        path = tmp_path / "custom_pricing.json"
        mgr1 = CustomPricingManager(persist_path=path)
        mgr1.register("persisted-model", "custom", "nano", 0.01, 0.02)

        mgr2 = CustomPricingManager(persist_path=path)
        result = mgr2.get("persisted-model")
        assert result is not None
        assert result.input_per_million == pytest.approx(0.01)

    def test_export_to_file_creates_json(self, tmp_path: pathlib.Path) -> None:
        mgr = CustomPricingManager()
        mgr.register("export-model", "custom", "nano", 0.05, 0.10)
        export_path = tmp_path / "export.json"
        mgr.export_to_file(export_path)
        assert export_path.exists()
        data = json.loads(export_path.read_text(encoding="utf-8"))
        assert "export-model" in data
