"""Unit tests for agent_energy_budget.pricing.updater."""
from __future__ import annotations

import json
import pathlib
from unittest.mock import MagicMock, patch

import pytest

from agent_energy_budget.pricing.tables import PROVIDER_PRICING, ModelPricing, ModelTier, ProviderName
from agent_energy_budget.pricing.updater import (
    PricingUpdateError,
    PricingUpdater,
    _parse_pricing_dict,
)


# ---------------------------------------------------------------------------
# _parse_pricing_dict helper
# ---------------------------------------------------------------------------


class TestParsePricingDict:
    def test_valid_entry_is_parsed(self) -> None:
        data = {
            "test-model": {
                "provider": "anthropic",
                "tier": "efficient",
                "input_per_million": 1.0,
                "output_per_million": 3.0,
                "context_window": 128000,
                "supports_vision": False,
            }
        }
        result = _parse_pricing_dict(data)
        assert "test-model" in result
        assert result["test-model"].input_per_million == 1.0

    def test_missing_required_field_is_skipped(self) -> None:
        data = {"bad-model": {"provider": "openai"}}  # missing input_per_million
        result = _parse_pricing_dict(data)
        assert "bad-model" not in result

    def test_non_dict_entry_is_skipped(self) -> None:
        data = {"not-a-dict": "just a string"}  # type: ignore[dict-item]
        result = _parse_pricing_dict(data)
        assert "not-a-dict" not in result

    def test_unknown_provider_defaults_to_custom(self) -> None:
        # _parse_pricing_dict does not coerce invalid enum values; entries with
        # an unrecognised provider are logged as warnings and skipped entirely.
        data = {
            "custom-llm": {
                "provider": "unknown-provider-xyz",
                "tier": "efficient",
                "input_per_million": 0.5,
                "output_per_million": 1.5,
            }
        }
        result = _parse_pricing_dict(data)
        assert "custom-llm" not in result

    def test_unknown_tier_defaults_to_efficient(self) -> None:
        # _parse_pricing_dict does not coerce invalid enum values; entries with
        # an unrecognised tier are logged as warnings and skipped entirely.
        data = {
            "weird-tier-model": {
                "provider": "openai",
                "tier": "hyper-premium",
                "input_per_million": 0.5,
                "output_per_million": 1.5,
            }
        }
        result = _parse_pricing_dict(data)
        assert "weird-tier-model" not in result

    def test_empty_dict_returns_empty(self) -> None:
        result = _parse_pricing_dict({})
        assert result == {}

    def test_multiple_valid_entries(self) -> None:
        data = {
            "model-a": {
                "provider": "openai",
                "tier": "standard",
                "input_per_million": 2.0,
                "output_per_million": 6.0,
            },
            "model-b": {
                "provider": "anthropic",
                "tier": "premium",
                "input_per_million": 15.0,
                "output_per_million": 75.0,
            },
        }
        result = _parse_pricing_dict(data)
        assert len(result) == 2


# ---------------------------------------------------------------------------
# PricingUpdater.apply_updates
# ---------------------------------------------------------------------------


class TestPricingUpdaterApplyUpdates:
    def test_new_model_is_added(self) -> None:
        updater = PricingUpdater()
        unique_model_name = "test-apply-updates-new-model-xyz"
        updates = {
            unique_model_name: ModelPricing(
                model=unique_model_name,
                provider=ProviderName.CUSTOM,
                tier=ModelTier.NANO,
                input_per_million=0.01,
                output_per_million=0.02,
            )
        }
        applied = updater.apply_updates(updates)
        assert unique_model_name in applied
        assert unique_model_name in PROVIDER_PRICING
        # Cleanup
        del PROVIDER_PRICING[unique_model_name]

    def test_existing_model_overwritten_by_default(self) -> None:
        updater = PricingUpdater()
        original_rate = PROVIDER_PRICING["gpt-4o-mini"].input_per_million
        updates = {
            "gpt-4o-mini": ModelPricing(
                model="gpt-4o-mini",
                provider=ProviderName.OPENAI,
                tier=ModelTier.EFFICIENT,
                input_per_million=999.0,
                output_per_million=999.0,
            )
        }
        updater.apply_updates(updates, overwrite_existing=True)
        assert PROVIDER_PRICING["gpt-4o-mini"].input_per_million == 999.0
        # Restore original
        PROVIDER_PRICING["gpt-4o-mini"] = ModelPricing(
            model="gpt-4o-mini",
            provider=ProviderName.OPENAI,
            tier=ModelTier.EFFICIENT,
            input_per_million=original_rate,
            output_per_million=0.60,
            context_window=128_000,
            supports_vision=True,
        )

    def test_overwrite_false_preserves_existing(self) -> None:
        updater = PricingUpdater()
        original_rate = PROVIDER_PRICING["gpt-4o-mini"].input_per_million
        updates = {
            "gpt-4o-mini": ModelPricing(
                model="gpt-4o-mini",
                provider=ProviderName.OPENAI,
                tier=ModelTier.EFFICIENT,
                input_per_million=999.0,
                output_per_million=999.0,
            )
        }
        applied = updater.apply_updates(updates, overwrite_existing=False)
        assert "gpt-4o-mini" not in applied
        assert PROVIDER_PRICING["gpt-4o-mini"].input_per_million == original_rate

    def test_returns_list_of_applied_model_names(self) -> None:
        updater = PricingUpdater()
        model_name = "test-return-list-model"
        updates = {
            model_name: ModelPricing(
                model=model_name,
                provider=ProviderName.CUSTOM,
                tier=ModelTier.NANO,
                input_per_million=0.01,
                output_per_million=0.02,
            )
        }
        applied = updater.apply_updates(updates)
        assert model_name in applied
        del PROVIDER_PRICING[model_name]


# ---------------------------------------------------------------------------
# PricingUpdater.load_custom_pricing
# ---------------------------------------------------------------------------


class TestPricingUpdaterLoadCustomPricing:
    def test_loads_valid_json_file(self, tmp_path: pathlib.Path) -> None:
        pricing_file = tmp_path / "pricing.json"
        model_name = "file-loaded-model"
        pricing_file.write_text(
            json.dumps({
                model_name: {
                    "provider": "custom",
                    "tier": "nano",
                    "input_per_million": 0.01,
                    "output_per_million": 0.02,
                }
            }),
            encoding="utf-8",
        )
        updater = PricingUpdater()
        applied = updater.load_custom_pricing(pricing_file)
        assert model_name in applied
        if model_name in PROVIDER_PRICING:
            del PROVIDER_PRICING[model_name]

    def test_missing_file_raises_pricing_update_error(self) -> None:
        updater = PricingUpdater()
        with pytest.raises(PricingUpdateError):
            updater.load_custom_pricing("/nonexistent/path/pricing.json")

    def test_invalid_json_raises_pricing_update_error(self, tmp_path: pathlib.Path) -> None:
        bad_file = tmp_path / "bad.json"
        bad_file.write_text("not json!!!", encoding="utf-8")
        updater = PricingUpdater()
        with pytest.raises(PricingUpdateError):
            updater.load_custom_pricing(bad_file)

    def test_non_object_root_raises_pricing_update_error(self, tmp_path: pathlib.Path) -> None:
        bad_file = tmp_path / "bad.json"
        bad_file.write_text(json.dumps([1, 2, 3]), encoding="utf-8")
        updater = PricingUpdater()
        with pytest.raises(PricingUpdateError, match="Expected a JSON object"):
            updater.load_custom_pricing(bad_file)


# ---------------------------------------------------------------------------
# PricingUpdater.fetch_remote_pricing (mocked)
# ---------------------------------------------------------------------------


class TestPricingUpdaterFetchRemote:
    def test_network_error_raises_pricing_update_error(self) -> None:
        updater = PricingUpdater(pricing_url="http://invalid-host-xyz.test/pricing.json")
        with patch("urllib.request.urlopen", side_effect=OSError("connection refused")):
            with pytest.raises(PricingUpdateError, match="Failed to fetch"):
                updater.fetch_remote_pricing()

    def test_invalid_json_response_raises_pricing_update_error(self) -> None:
        mock_response = MagicMock()
        mock_response.read.return_value = b"not json"
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_response):
            updater = PricingUpdater()
            with pytest.raises(PricingUpdateError, match="Invalid JSON"):
                updater.fetch_remote_pricing()

    def test_non_object_json_root_raises_pricing_update_error(self) -> None:
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps([1, 2, 3]).encode("utf-8")
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_response):
            updater = PricingUpdater()
            with pytest.raises(PricingUpdateError, match="Expected a JSON object"):
                updater.fetch_remote_pricing()

    def test_valid_response_returns_pricing_dict(self) -> None:
        payload = {
            "remote-model": {
                "provider": "custom",
                "tier": "nano",
                "input_per_million": 0.05,
                "output_per_million": 0.10,
            }
        }
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps(payload).encode("utf-8")
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_response):
            updater = PricingUpdater()
            result = updater.fetch_remote_pricing()
        assert "remote-model" in result
