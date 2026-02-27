"""Tests for agent_energy_budget.caching.cache_tracker — CacheTokenTracker."""
from __future__ import annotations

import pytest

from agent_energy_budget.caching.cache_tracker import (
    CacheTokenTracker,
    CachePricingConfig,
    CacheUsageRecord,
    CacheTrackerStats,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def default_tracker() -> CacheTokenTracker:
    return CacheTokenTracker()


@pytest.fixture()
def custom_config() -> CachePricingConfig:
    return CachePricingConfig(
        base_input_price_per_million=3.00,
        cache_read_discount=0.10,   # 90% off
        cache_write_premium=1.25,
        output_price_override_per_million=15.00,
    )


@pytest.fixture()
def custom_tracker(custom_config: CachePricingConfig) -> CacheTokenTracker:
    return CacheTokenTracker(custom_config)


# ===========================================================================
# CachePricingConfig validation
# ===========================================================================


class TestCachePricingConfig:
    def test_defaults(self) -> None:
        config = CachePricingConfig()
        assert config.base_input_price_per_million == 3.00
        assert config.cache_read_discount == 0.10
        assert config.cache_write_premium == 1.25

    def test_negative_input_price_raises(self) -> None:
        with pytest.raises(ValueError, match="base_input_price_per_million"):
            CachePricingConfig(base_input_price_per_million=-1.0)

    def test_invalid_discount_raises(self) -> None:
        with pytest.raises(ValueError, match="cache_read_discount"):
            CachePricingConfig(cache_read_discount=1.5)

    def test_negative_write_premium_raises(self) -> None:
        with pytest.raises(ValueError, match="cache_write_premium"):
            CachePricingConfig(cache_write_premium=-0.1)


# ===========================================================================
# record_response — Anthropic-style keys
# ===========================================================================


class TestRecordResponseAnthropic:
    def test_anthropic_style_no_cache(self, custom_tracker: CacheTokenTracker) -> None:
        usage = {
            "input_tokens": 1000,
            "output_tokens": 200,
        }
        record = custom_tracker.record_response(usage, base_output_price_per_million=15.0)
        assert record.total_input_tokens == 1000
        assert record.cache_read_tokens == 0
        assert record.cache_write_tokens == 0
        assert record.was_cached is False
        assert record.savings_usd == 0.0

    def test_anthropic_style_with_cache_read(self, custom_tracker: CacheTokenTracker) -> None:
        usage = {
            "input_tokens": 1000,
            "cache_read_input_tokens": 800,
            "output_tokens": 200,
        }
        record = custom_tracker.record_response(usage, base_output_price_per_million=15.0)
        assert record.was_cached is True
        assert record.cache_read_tokens == 800
        assert record.non_cached_input_tokens == 200
        # Savings should be positive (cache is cheaper)
        assert record.savings_usd > 0.0

    def test_anthropic_style_with_cache_write(self, custom_tracker: CacheTokenTracker) -> None:
        usage = {
            "input_tokens": 1000,
            "cache_creation_input_tokens": 500,
            "output_tokens": 200,
        }
        record = custom_tracker.record_response(usage, base_output_price_per_million=15.0)
        assert record.cache_write_tokens == 500
        # Cache writes are more expensive, so full cost < actual cost when writes happen
        assert record.cost_usd > 0


# ===========================================================================
# record_response — OpenAI-style keys
# ===========================================================================


class TestRecordResponseOpenAI:
    def test_openai_style_no_cache(self, default_tracker: CacheTokenTracker) -> None:
        usage = {
            "prompt_tokens": 500,
            "completion_tokens": 100,
        }
        record = default_tracker.record_response(usage)
        assert record.total_input_tokens == 500
        assert record.output_tokens == 100
        assert not record.was_cached

    def test_openai_style_with_cached_tokens(self, default_tracker: CacheTokenTracker) -> None:
        usage = {
            "prompt_tokens": 1000,
            "cached_tokens": 600,
            "completion_tokens": 200,
        }
        record = default_tracker.record_response(usage)
        assert record.was_cached is True
        assert record.cache_read_tokens == 600


# ===========================================================================
# Cost calculations
# ===========================================================================


class TestCostCalculations:
    def test_no_cache_full_price(self) -> None:
        config = CachePricingConfig(
            base_input_price_per_million=3.00,
            cache_read_discount=0.10,
            cache_write_premium=1.0,
        )
        tracker = CacheTokenTracker(config)
        # 1M input tokens = $3.00; 1M output tokens = $15.00
        usage = {"input_tokens": 1_000_000, "output_tokens": 1_000_000}
        record = tracker.record_response(usage, base_output_price_per_million=15.0)
        assert abs(record.cost_usd - 18.00) < 0.001
        assert record.savings_usd == 0.0

    def test_cache_read_discount_applied(self) -> None:
        config = CachePricingConfig(
            base_input_price_per_million=3.00,
            cache_read_discount=0.10,  # 10% of full price
        )
        tracker = CacheTokenTracker(config)
        # All 1M input tokens served from cache
        usage = {
            "input_tokens": 1_000_000,
            "cache_read_input_tokens": 1_000_000,
            "output_tokens": 0,
        }
        record = tracker.record_response(usage)
        # 1M cache read tokens at $0.30/M (10% of $3.00)
        expected_cost = 0.30
        assert abs(record.cost_usd - expected_cost) < 0.001
        assert record.savings_usd > 0

    def test_savings_equals_full_minus_actual(self) -> None:
        tracker = CacheTokenTracker()
        usage = {
            "input_tokens": 1000,
            "cache_read_input_tokens": 500,
            "output_tokens": 100,
        }
        record = tracker.record_response(usage, base_output_price_per_million=10.0)
        assert abs(record.savings_usd - (record.full_price_cost_usd - record.cost_usd)) < 1e-9


# ===========================================================================
# stats()
# ===========================================================================


class TestStats:
    def test_empty_stats(self, default_tracker: CacheTokenTracker) -> None:
        stats = default_tracker.stats()
        assert stats.total_responses == 0
        assert stats.cache_hit_rate == 0.0
        assert stats.total_cost_usd == 0.0

    def test_stats_after_one_cached_response(self, default_tracker: CacheTokenTracker) -> None:
        usage = {
            "input_tokens": 1000,
            "cache_read_input_tokens": 800,
            "output_tokens": 200,
        }
        default_tracker.record_response(usage)
        stats = default_tracker.stats()
        assert stats.total_responses == 1
        assert stats.cached_responses == 1
        assert stats.cache_hit_rate == 1.0
        assert stats.total_cache_read_tokens == 800

    def test_cache_hit_rate_mixed(self, default_tracker: CacheTokenTracker) -> None:
        # One cached, one non-cached
        default_tracker.record_response({"input_tokens": 100, "cache_read_input_tokens": 80})
        default_tracker.record_response({"input_tokens": 100})
        stats = default_tracker.stats()
        assert stats.total_responses == 2
        assert stats.cached_responses == 1
        assert abs(stats.cache_hit_rate - 0.5) < 1e-6

    def test_token_cache_hit_rate(self, default_tracker: CacheTokenTracker) -> None:
        default_tracker.record_response({"input_tokens": 1000, "cache_read_input_tokens": 700})
        stats = default_tracker.stats()
        assert abs(stats.token_cache_hit_rate - 0.7) < 1e-6

    def test_savings_accumulate(self, default_tracker: CacheTokenTracker) -> None:
        for _ in range(5):
            default_tracker.record_response(
                {"input_tokens": 1000, "cache_read_input_tokens": 900}
            )
        stats = default_tracker.stats()
        assert stats.total_savings_usd > 0

    def test_returns_stats_type(self, default_tracker: CacheTokenTracker) -> None:
        stats = default_tracker.stats()
        assert isinstance(stats, CacheTrackerStats)


# ===========================================================================
# reset and record_count
# ===========================================================================


class TestResetAndCount:
    def test_reset_clears_records(self, default_tracker: CacheTokenTracker) -> None:
        default_tracker.record_response({"input_tokens": 100})
        default_tracker.reset()
        assert default_tracker.record_count() == 0
        stats = default_tracker.stats()
        assert stats.total_responses == 0

    def test_record_count(self, default_tracker: CacheTokenTracker) -> None:
        for _ in range(7):
            default_tracker.record_response({"input_tokens": 10})
        assert default_tracker.record_count() == 7


# ===========================================================================
# output price override
# ===========================================================================


class TestOutputPriceOverride:
    def test_config_output_price_used_when_no_param(self) -> None:
        config = CachePricingConfig(
            base_input_price_per_million=0.0,
            output_price_override_per_million=10.0,
        )
        tracker = CacheTokenTracker(config)
        record = tracker.record_response({"input_tokens": 0, "output_tokens": 1_000_000})
        assert abs(record.cost_usd - 10.0) < 0.001

    def test_call_param_overrides_config(self) -> None:
        config = CachePricingConfig(
            base_input_price_per_million=0.0,
            output_price_override_per_million=5.0,
        )
        tracker = CacheTokenTracker(config)
        # Pass 20.0 explicitly — should override the config's 5.0
        record = tracker.record_response(
            {"input_tokens": 0, "output_tokens": 1_000_000},
            base_output_price_per_million=20.0,
        )
        assert abs(record.cost_usd - 20.0) < 0.001
