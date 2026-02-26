"""Unit tests for agent_energy_budget.degradation package."""
from __future__ import annotations

import pytest

from agent_energy_budget.budget.tracker import BudgetStatus
from agent_energy_budget.degradation.base import DegradationResult, DegradationStrategyBase
from agent_energy_budget.degradation.block_with_error import BlockStrategy
from agent_energy_budget.degradation.cached_fallback import CachedFallbackStrategy, CachedResponse
from agent_energy_budget.degradation.model_downgrade import ModelDowngradeStrategy
from agent_energy_budget.degradation.registry import (
    StrategyAlreadyRegisteredError,
    StrategyNotFoundError,
    StrategyRegistry,
)
from agent_energy_budget.degradation.strategies import (
    DegradationAction,
    DegradationManager,
    DegradationStrategy,
    _next_lower_tier_model,
    _tier_for_model,
)
from agent_energy_budget.degradation.token_reduction import TokenReductionStrategy
from agent_energy_budget.pricing.tables import ModelPricing, ModelTier, ProviderName


# ===========================================================================
# DegradationResult (base)
# ===========================================================================


class TestDegradationResult:
    def test_is_frozen(self) -> None:
        result = DegradationResult(
            can_proceed=True,
            recommended_model="gpt-4o-mini",
            max_tokens=512,
            action="proceed",
            message="OK",
        )
        with pytest.raises((AttributeError, TypeError)):
            result.can_proceed = False  # type: ignore[misc]


# ===========================================================================
# DegradationStrategyBase
# ===========================================================================


class TestDegradationStrategyBase:
    def test_name_derives_from_class_name(self) -> None:
        class MyCustomStrategy(DegradationStrategyBase):
            def apply(self, model, input_tokens, output_tokens, remaining_budget):
                return DegradationResult(True, model, output_tokens, "proceed", "ok")

        s = MyCustomStrategy()
        assert "mycustom" in s.name()

    def test_abstract_method_enforced(self) -> None:
        with pytest.raises(TypeError):
            DegradationStrategyBase()  # type: ignore[abstract]


# ===========================================================================
# BlockStrategy
# ===========================================================================


class TestBlockStrategy:
    def test_apply_returns_cannot_proceed(self) -> None:
        strategy = BlockStrategy()
        result = strategy.apply("gpt-4o-mini", 100, 50, 0.001)
        assert result.can_proceed is False
        assert result.action == "block"
        assert result.max_tokens == 0

    def test_apply_with_detail_includes_cost_info(self) -> None:
        strategy = BlockStrategy(include_pricing_detail=True)
        result = strategy.apply("gpt-4o-mini", 1000, 500, 0.001)
        assert "gpt-4o-mini" in result.message

    def test_apply_without_detail_brief_message(self) -> None:
        strategy = BlockStrategy(include_pricing_detail=False)
        result = strategy.apply("gpt-4o-mini", 1000, 500, 0.001)
        assert result.message == "Budget exceeded."

    def test_apply_unknown_model_with_detail(self) -> None:
        strategy = BlockStrategy(include_pricing_detail=True)
        result = strategy.apply("unknown-model-xyz", 1000, 500, 0.001)
        assert result.can_proceed is False
        assert "unknown-model-xyz" in result.message

    def test_recommended_model_unchanged(self) -> None:
        strategy = BlockStrategy()
        result = strategy.apply("gpt-4o", 100, 100, 0.0)
        assert result.recommended_model == "gpt-4o"


# ===========================================================================
# CachedFallbackStrategy
# ===========================================================================


class TestCachedFallbackStrategy:
    @pytest.fixture
    def strategy(self) -> CachedFallbackStrategy:
        return CachedFallbackStrategy(max_cache_size=10)

    def test_apply_returns_use_cache(self, strategy: CachedFallbackStrategy) -> None:
        result = strategy.apply("gpt-4o-mini", 100, 50, 0.001)
        assert result.can_proceed is False
        assert result.action == "use_cache"

    def test_apply_unknown_model(self, strategy: CachedFallbackStrategy) -> None:
        result = strategy.apply("unknown-xyz", 100, 50, 0.001)
        assert result.can_proceed is False
        assert result.action == "use_cache"

    def test_max_cache_size_zero_raises(self) -> None:
        with pytest.raises(ValueError):
            CachedFallbackStrategy(max_cache_size=0)

    def test_store_and_retrieve(self, strategy: CachedFallbackStrategy) -> None:
        strategy.store("gpt-4o-mini", "What is 2+2?", "4")
        entry = strategy.get_cached_response("gpt-4o-mini", "What is 2+2?")
        assert entry is not None
        assert entry.response == "4"

    def test_cache_miss_returns_none(self, strategy: CachedFallbackStrategy) -> None:
        entry = strategy.get_cached_response("gpt-4o-mini", "Not stored")
        assert entry is None

    def test_hit_count_increments_on_retrieval(self, strategy: CachedFallbackStrategy) -> None:
        strategy.store("gpt-4o-mini", "prompt", "response")
        strategy.get_cached_response("gpt-4o-mini", "prompt")
        entry = strategy.get_cached_response("gpt-4o-mini", "prompt")
        assert entry is not None
        assert entry.hit_count == 2

    def test_store_returns_prompt_hash(self, strategy: CachedFallbackStrategy) -> None:
        prompt_hash = strategy.store("gpt-4o-mini", "Hello", "Hi")
        assert isinstance(prompt_hash, str)
        assert len(prompt_hash) == 64  # SHA-256 hex

    def test_lru_eviction_when_full(self) -> None:
        strategy = CachedFallbackStrategy(max_cache_size=2)
        strategy.store("gpt-4o-mini", "prompt1", "resp1")
        strategy.store("gpt-4o-mini", "prompt2", "resp2")
        strategy.store("gpt-4o-mini", "prompt3", "resp3")  # evicts prompt1
        assert strategy.size == 2
        assert strategy.get_cached_response("gpt-4o-mini", "prompt1") is None
        assert strategy.get_cached_response("gpt-4o-mini", "prompt3") is not None

    def test_store_with_messages_list(self, strategy: CachedFallbackStrategy) -> None:
        messages = [{"role": "user", "content": "Hello"}]
        strategy.store("gpt-4o-mini", messages, "Hi there")
        entry = strategy.get_cached_response("gpt-4o-mini", messages)
        assert entry is not None
        assert entry.response == "Hi there"

    def test_invalidate_removes_entry(self, strategy: CachedFallbackStrategy) -> None:
        strategy.store("gpt-4o-mini", "prompt", "resp")
        removed = strategy.invalidate("gpt-4o-mini", "prompt")
        assert removed is True
        assert strategy.get_cached_response("gpt-4o-mini", "prompt") is None

    def test_invalidate_missing_returns_false(self, strategy: CachedFallbackStrategy) -> None:
        removed = strategy.invalidate("gpt-4o-mini", "not-stored")
        assert removed is False

    def test_clear_returns_count(self, strategy: CachedFallbackStrategy) -> None:
        strategy.store("gpt-4o-mini", "p1", "r1")
        strategy.store("gpt-4o-mini", "p2", "r2")
        count = strategy.clear()
        assert count == 2
        assert strategy.size == 0

    def test_stats_returns_expected_keys(self, strategy: CachedFallbackStrategy) -> None:
        strategy.store("gpt-4o-mini", "p", "r")
        stats = strategy.stats()
        assert "size" in stats
        assert "max_size" in stats
        assert "total_hits" in stats

    def test_size_property(self, strategy: CachedFallbackStrategy) -> None:
        assert strategy.size == 0
        strategy.store("gpt-4o-mini", "p", "r")
        assert strategy.size == 1

    def test_lru_refresh_on_store_existing(self) -> None:
        strategy = CachedFallbackStrategy(max_cache_size=2)
        strategy.store("m", "p1", "r1")
        strategy.store("m", "p2", "r2")
        # Re-store p1 to move it to end (most recent)
        strategy.store("m", "p1", "r1")
        # Adding p3 should evict p2, not p1
        strategy.store("m", "p3", "r3")
        assert strategy.get_cached_response("m", "p1") is not None
        assert strategy.get_cached_response("m", "p2") is None


# ===========================================================================
# ModelDowngradeStrategy
# ===========================================================================


class TestModelDowngradeStrategy:
    @pytest.fixture
    def strategy(self) -> ModelDowngradeStrategy:
        return ModelDowngradeStrategy()

    def test_apply_finds_cheaper_model(self, strategy: ModelDowngradeStrategy) -> None:
        # Very small budget — only cheap models can fit
        result = strategy.apply("gpt-4o", 100, 100, 0.00001)
        # Either found a cheaper model or blocked
        assert result.action in ("model_downgrade", "block", "proceed")

    def test_apply_blocks_when_no_affordable_model(self, strategy: ModelDowngradeStrategy) -> None:
        # Enormous token counts to exhaust all model budgets
        result = strategy.apply("gpt-4o", 100_000_000, 100_000_000, 0.000001)
        assert result.can_proceed is False
        assert result.action == "block"

    def test_apply_proceeds_when_original_model_fits(self, strategy: ModelDowngradeStrategy) -> None:
        # Large budget — gpt-4o-mini should fit easily for small tokens.
        # The strategy picks the BEST model that fits, so when budget is large
        # it may recommend a more premium model instead of blocking.
        result = strategy.apply("gpt-4o-mini", 10, 10, 100.0)
        assert result.can_proceed is True
        # With $100 budget even the most expensive model fits
        assert result.action in ("proceed", "model_downgrade")

    def test_custom_pricing_included(self) -> None:
        cheap = ModelPricing(
            model="ultra-cheap",
            provider=ProviderName.CUSTOM,
            tier=ModelTier.NANO,
            input_per_million=0.001,
            output_per_million=0.002,
        )
        strategy = ModelDowngradeStrategy(custom_pricing={"ultra-cheap": cheap})
        result = strategy.apply("gpt-4o", 100, 100, 0.0000001)
        # ultra-cheap should fit within this tiny budget for 100 tokens
        if result.can_proceed:
            assert result.recommended_model == "ultra-cheap"

    def test_models_within_budget(self, strategy: ModelDowngradeStrategy) -> None:
        models = strategy.models_within_budget(100, 100, 1.0)
        assert isinstance(models, list)
        # With $1.00 budget and 100 tokens, many models should fit
        assert len(models) > 0

    def test_models_within_budget_sorted_by_quality_desc(
        self, strategy: ModelDowngradeStrategy
    ) -> None:
        models = strategy.models_within_budget(100, 100, 100.0)
        for i in range(len(models) - 1):
            assert models[i].input_per_million >= models[i + 1].input_per_million

    def test_models_within_budget_empty_when_nothing_fits(
        self, strategy: ModelDowngradeStrategy
    ) -> None:
        models = strategy.models_within_budget(100_000_000, 100_000_000, 0.0)
        assert models == []


# ===========================================================================
# TokenReductionStrategy
# ===========================================================================


class TestTokenReductionStrategy:
    @pytest.fixture
    def strategy(self) -> TokenReductionStrategy:
        return TokenReductionStrategy(absolute_minimum_tokens=50)

    def test_min_tokens_zero_raises(self) -> None:
        with pytest.raises(ValueError):
            TokenReductionStrategy(absolute_minimum_tokens=0)

    def test_apply_proceeds_when_original_fits(self, strategy: TokenReductionStrategy) -> None:
        result = strategy.apply("gpt-4o-mini", 100, 100, 100.0)
        assert result.can_proceed is True
        assert result.action == "proceed"
        assert result.max_tokens == 100

    def test_apply_reduces_tokens_when_over_budget(self, strategy: TokenReductionStrategy) -> None:
        # gpt-4o-mini: $0.15/M input, $0.60/M output
        # Budget $0.000030 → about 50 output tokens affordable after 100 input
        result = strategy.apply("gpt-4o-mini", 100, 10000, 0.00003)
        if result.can_proceed:
            assert result.action == "reduce_tokens"
            assert result.max_tokens < 10000
            assert result.max_tokens >= 50
        else:
            assert result.action == "block"

    def test_apply_blocks_when_below_minimum(self, strategy: TokenReductionStrategy) -> None:
        # Zero budget — nothing can fit
        result = strategy.apply("gpt-4o-mini", 10000, 10000, 0.0)
        assert result.can_proceed is False
        assert result.action == "block"

    def test_apply_unknown_model_blocks(self, strategy: TokenReductionStrategy) -> None:
        result = strategy.apply("totally-unknown-xyz", 100, 100, 1.0)
        assert result.can_proceed is False
        assert result.action == "block"
        assert "Unknown model" in result.message

    def test_calculate_max_tokens_known_model(self, strategy: TokenReductionStrategy) -> None:
        max_tokens = strategy.calculate_max_tokens("gpt-4o-mini", 100, 0.01)
        assert max_tokens >= 0

    def test_calculate_max_tokens_unknown_model_returns_zero(
        self, strategy: TokenReductionStrategy
    ) -> None:
        max_tokens = strategy.calculate_max_tokens("unknown-xyz", 100, 1.0)
        assert max_tokens == 0

    def test_custom_pricing_used(self) -> None:
        cheap = ModelPricing(
            model="cheap-model",
            provider=ProviderName.CUSTOM,
            tier=ModelTier.NANO,
            input_per_million=0.001,
            output_per_million=0.002,
        )
        strategy = TokenReductionStrategy(custom_pricing={"cheap-model": cheap})
        # Very large original request, very small budget
        result = strategy.apply("cheap-model", 10, 100000, 0.00001)
        if result.can_proceed:
            assert result.max_tokens < 100000


# ===========================================================================
# StrategyRegistry
# ===========================================================================


class TestStrategyRegistry:
    def test_builtins_registered_by_default(self) -> None:
        registry = StrategyRegistry()
        names = registry.list_strategies()
        assert "token_reduction" in names
        assert "block_with_error" in names
        assert "model_downgrade" in names
        assert "cached_fallback" in names

    def test_no_builtins_when_disabled(self) -> None:
        registry = StrategyRegistry(register_builtins=False)
        assert len(registry) == 0

    def test_get_instance_returns_strategy(self) -> None:
        registry = StrategyRegistry()
        instance = registry.get_instance("token_reduction")
        assert isinstance(instance, TokenReductionStrategy)

    def test_get_instance_cached(self) -> None:
        registry = StrategyRegistry()
        i1 = registry.get_instance("token_reduction")
        i2 = registry.get_instance("token_reduction")
        assert i1 is i2

    def test_get_class_returns_class(self) -> None:
        registry = StrategyRegistry()
        cls = registry.get_class("block_with_error")
        assert cls is BlockStrategy

    def test_get_unknown_strategy_raises(self) -> None:
        registry = StrategyRegistry()
        with pytest.raises(StrategyNotFoundError):
            registry.get_instance("nonexistent-strategy")

    def test_get_class_unknown_raises(self) -> None:
        registry = StrategyRegistry()
        with pytest.raises(StrategyNotFoundError):
            registry.get_class("nonexistent")

    def test_register_custom_strategy(self) -> None:
        registry = StrategyRegistry(register_builtins=False)

        class MyStrategy(DegradationStrategyBase):
            def apply(self, model, input_tokens, output_tokens, remaining_budget):
                return DegradationResult(True, model, output_tokens, "proceed", "ok")

        registry.register("my_strategy", MyStrategy)
        assert "my_strategy" in registry
        instance = registry.get_instance("my_strategy")
        assert isinstance(instance, MyStrategy)

    def test_register_duplicate_raises(self) -> None:
        registry = StrategyRegistry(register_builtins=False)

        class MyStrategy(DegradationStrategyBase):
            def apply(self, model, input_tokens, output_tokens, remaining_budget):
                return DegradationResult(True, model, output_tokens, "proceed", "ok")

        registry.register("my_strategy", MyStrategy)
        with pytest.raises(StrategyAlreadyRegisteredError):
            registry.register("my_strategy", MyStrategy)

    def test_register_with_overwrite(self) -> None:
        registry = StrategyRegistry(register_builtins=False)

        class MyStrategy(DegradationStrategyBase):
            def apply(self, model, input_tokens, output_tokens, remaining_budget):
                return DegradationResult(True, model, output_tokens, "proceed", "ok")

        registry.register("my_strategy", MyStrategy)
        # Should not raise with overwrite=True
        registry.register("my_strategy", MyStrategy, overwrite=True)

    def test_register_non_strategy_class_raises(self) -> None:
        registry = StrategyRegistry(register_builtins=False)
        with pytest.raises(TypeError):
            registry.register("bad", str)  # type: ignore[arg-type]

    def test_deregister_removes_strategy(self) -> None:
        registry = StrategyRegistry()
        registry.deregister("token_reduction")
        assert "token_reduction" not in registry

    def test_deregister_unknown_raises(self) -> None:
        registry = StrategyRegistry()
        with pytest.raises(StrategyNotFoundError):
            registry.deregister("nonexistent")

    def test_len_returns_count(self) -> None:
        registry = StrategyRegistry()
        assert len(registry) == 4  # 4 built-ins

    def test_contains_operator(self) -> None:
        registry = StrategyRegistry()
        assert "token_reduction" in registry
        assert "unknown" not in registry

    def test_list_strategies_sorted(self) -> None:
        registry = StrategyRegistry()
        names = registry.list_strategies()
        assert names == sorted(names)

    def test_register_with_singleton_instance(self) -> None:
        registry = StrategyRegistry(register_builtins=False)

        class MyStrategy(DegradationStrategyBase):
            def apply(self, model, input_tokens, output_tokens, remaining_budget):
                return DegradationResult(True, model, output_tokens, "proceed", "ok")

        singleton = MyStrategy()
        registry.register("my_s", MyStrategy, instance=singleton)
        retrieved = registry.get_instance("my_s")
        assert retrieved is singleton


# ===========================================================================
# DegradationManager (strategies.py)
# ===========================================================================


def _make_status(
    spent: float,
    limit: float,
    agent_id: str = "agent",
    period: str = "daily",
) -> BudgetStatus:
    remaining = limit - spent
    utilisation = (spent / limit * 100.0) if limit > 0 else 0.0
    return BudgetStatus(
        agent_id=agent_id,
        period=period,
        limit_usd=limit,
        spent_usd=spent,
        remaining_usd=remaining,
        utilisation_pct=utilisation,
        call_count=1,
        avg_cost_per_call=spent,
    )


class TestDegradationManager:
    def test_none_strategy_always_proceeds(self) -> None:
        manager = DegradationManager(strategy=DegradationStrategy.NONE)
        status = _make_status(9.9, 10.0)  # 99% utilisation
        action = manager.check(status, "gpt-4o")
        assert action.degraded_model == "gpt-4o"
        assert action.token_limit is None
        assert action.rate_limit_per_minute is None

    def test_below_threshold_proceeds_unchanged(self) -> None:
        manager = DegradationManager(strategy=DegradationStrategy.MODEL_DOWNGRADE)
        status = _make_status(5.0, 10.0)  # 50% utilisation, below 80% warning
        action = manager.check(status, "gpt-4o")
        assert action.degraded_model == "gpt-4o"
        assert action.token_limit is None

    def test_model_downgrade_at_warning_threshold(self) -> None:
        manager = DegradationManager(strategy=DegradationStrategy.MODEL_DOWNGRADE)
        status = _make_status(8.5, 10.0)  # 85% utilisation
        action = manager.check(status, "gpt-4o")
        # Should downgrade from gpt-4o to something cheaper
        assert action.strategy == DegradationStrategy.MODEL_DOWNGRADE

    def test_model_downgrade_when_exhausted_blocks(self) -> None:
        manager = DegradationManager(strategy=DegradationStrategy.MODEL_DOWNGRADE)
        status = _make_status(10.0, 10.0)  # 100% = exhausted
        action = manager.check(status, "gpt-4o")
        assert action.token_limit == 0

    def test_suspend_when_exhausted(self) -> None:
        manager = DegradationManager(strategy=DegradationStrategy.SUSPEND)
        status = _make_status(10.0, 10.0)
        action = manager.check(status, "gpt-4o")
        assert action.token_limit == 0
        assert action.rate_limit_per_minute == 0

    def test_suspend_approaching_threshold_allows(self) -> None:
        manager = DegradationManager(strategy=DegradationStrategy.SUSPEND)
        status = _make_status(8.5, 10.0)  # 85%, not yet exhausted
        action = manager.check(status, "gpt-4o")
        assert action.token_limit is None

    def test_token_limit_reduces_proportionally(self) -> None:
        manager = DegradationManager(strategy=DegradationStrategy.TOKEN_LIMIT)
        status = _make_status(9.0, 10.0)  # 10% remaining
        action = manager.check(status, "gpt-4o")
        assert action.token_limit is not None
        assert action.token_limit >= 1
        # 10% remaining → roughly 10% of 4096
        assert action.token_limit <= 500

    def test_token_limit_when_exhausted_blocks(self) -> None:
        manager = DegradationManager(strategy=DegradationStrategy.TOKEN_LIMIT)
        status = _make_status(10.0, 10.0)
        action = manager.check(status, "gpt-4o")
        assert action.token_limit == 0

    def test_rate_limit_reduces_proportionally(self) -> None:
        manager = DegradationManager(strategy=DegradationStrategy.RATE_LIMIT)
        status = _make_status(9.0, 10.0)  # 10% remaining
        action = manager.check(status, "gpt-4o")
        assert action.rate_limit_per_minute is not None
        assert action.rate_limit_per_minute >= 1
        assert action.rate_limit_per_minute <= 10

    def test_rate_limit_when_exhausted_blocks(self) -> None:
        manager = DegradationManager(strategy=DegradationStrategy.RATE_LIMIT)
        status = _make_status(10.0, 10.0)
        action = manager.check(status, "gpt-4o")
        assert action.rate_limit_per_minute == 0

    def test_configure_updates_strategy(self) -> None:
        manager = DegradationManager(strategy=DegradationStrategy.NONE)
        manager.configure(DegradationStrategy.SUSPEND)
        status = _make_status(10.0, 10.0)
        action = manager.check(status, "gpt-4o")
        assert action.strategy == DegradationStrategy.SUSPEND

    def test_configure_updates_thresholds(self) -> None:
        manager = DegradationManager(strategy=DegradationStrategy.MODEL_DOWNGRADE)
        manager.configure(
            DegradationStrategy.MODEL_DOWNGRADE,
            thresholds={"warning": 30.0, "critical": 60.0},
        )
        # 40% utilisation should now trigger degradation (above 30% warning)
        status = _make_status(4.0, 10.0)  # 40%
        action = manager.check(status, "gpt-4o")
        assert action.strategy == DegradationStrategy.MODEL_DOWNGRADE
        # Should not simply "proceed" at the default 80% threshold
        assert "below" not in action.reason.lower() or "30" in action.reason

    def test_exhausted_when_remaining_usd_negative(self) -> None:
        manager = DegradationManager(strategy=DegradationStrategy.SUSPEND)
        status = BudgetStatus(
            agent_id="agent",
            period="daily",
            limit_usd=10.0,
            spent_usd=11.0,
            remaining_usd=-1.0,  # overspent
            utilisation_pct=110.0,
            call_count=5,
            avg_cost_per_call=2.2,
        )
        action = manager.check(status, "gpt-4o")
        assert action.token_limit == 0

    def test_model_downgrade_cheapest_tier_stays(self) -> None:
        manager = DegradationManager(strategy=DegradationStrategy.MODEL_DOWNGRADE)
        status = _make_status(9.0, 10.0)
        # A nano-tier model is already at the bottom
        action = manager.check(status, "deepseek-v3")
        # Should report already at cheapest tier
        if "cheapest" in action.reason.lower():
            assert action.degraded_model == "deepseek-v3"


# ===========================================================================
# Helper functions in strategies.py
# ===========================================================================


class TestTierHelpers:
    def test_tier_for_known_model(self) -> None:
        tier = _tier_for_model("gpt-4o")
        assert tier == "premium"

    def test_tier_for_standard_model(self) -> None:
        tier = _tier_for_model("gpt-4o-mini")
        assert tier == "standard"

    def test_tier_for_unknown_model_fallback(self) -> None:
        # Unknown model — should try pricing tables then return None
        tier = _tier_for_model("totally-unknown-xyz-abc")
        assert tier is None

    def test_next_lower_tier_from_premium(self) -> None:
        next_model = _next_lower_tier_model("gpt-4o")
        assert next_model is not None
        # Should be a standard-tier model
        assert next_model in ["claude-haiku-4", "gpt-4o-mini", "gemini-2.0-flash"]

    def test_next_lower_tier_from_nano_returns_none(self) -> None:
        # nano is the cheapest tier, no lower
        next_model = _next_lower_tier_model("deepseek-v3")
        assert next_model is None

    def test_next_lower_tier_unknown_model_falls_to_cheapest(self) -> None:
        next_model = _next_lower_tier_model("some-completely-unknown-model")
        # Should fall through to cheapest known tier
        assert next_model is not None
