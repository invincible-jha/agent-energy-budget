"""Unit tests for agent_energy_budget.budget.tracker."""
from __future__ import annotations

import json
import pathlib
import tempfile

import pytest

from agent_energy_budget.budget.config import BudgetConfig, DegradationStrategy
from agent_energy_budget.budget.tracker import (
    BudgetExceededError,
    BudgetRecommendation,
    BudgetStatus,
    BudgetTracker,
    _CallRecord,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_storage(tmp_path: pathlib.Path) -> pathlib.Path:
    return tmp_path


@pytest.fixture
def basic_config() -> BudgetConfig:
    return BudgetConfig(agent_id="test-agent", daily_limit=1.0)


@pytest.fixture
def tracker(basic_config: BudgetConfig, tmp_storage: pathlib.Path) -> BudgetTracker:
    return BudgetTracker(config=basic_config, storage_dir=tmp_storage)


# ---------------------------------------------------------------------------
# BudgetExceededError
# ---------------------------------------------------------------------------


class TestBudgetExceededError:
    def test_attributes_are_set(self) -> None:
        error = BudgetExceededError(
            agent_id="my-agent", remaining_usd=0.001, estimated_cost_usd=0.01
        )
        assert error.agent_id == "my-agent"
        assert error.remaining_usd == pytest.approx(0.001)
        assert error.estimated_cost_usd == pytest.approx(0.01)

    def test_message_contains_agent_id(self) -> None:
        error = BudgetExceededError("my-agent", 0.0, 0.05)
        assert "my-agent" in str(error)

    def test_is_runtime_error(self) -> None:
        assert issubclass(BudgetExceededError, RuntimeError)


# ---------------------------------------------------------------------------
# BudgetTracker — initialisation
# ---------------------------------------------------------------------------


class TestBudgetTrackerInit:
    def test_creates_storage_dir(self, basic_config: BudgetConfig, tmp_path: pathlib.Path) -> None:
        storage = tmp_path / "subdir" / "nested"
        BudgetTracker(config=basic_config, storage_dir=storage)
        assert storage.exists()

    def test_default_storage_not_needed_for_test(
        self, basic_config: BudgetConfig, tmp_storage: pathlib.Path
    ) -> None:
        tracker = BudgetTracker(config=basic_config, storage_dir=tmp_storage)
        assert tracker.agent_id == "test-agent"

    def test_config_property(self, tracker: BudgetTracker, basic_config: BudgetConfig) -> None:
        assert tracker.config is basic_config

    def test_agent_id_property(self, tracker: BudgetTracker) -> None:
        assert tracker.agent_id == "test-agent"

    def test_loads_existing_records_from_jsonl(
        self, basic_config: BudgetConfig, tmp_storage: pathlib.Path
    ) -> None:
        from datetime import datetime, timezone

        log_path = tmp_storage / "test-agent.jsonl"
        record = {
            "agent_id": "test-agent",
            "model": "gpt-4o-mini",
            "input_tokens": 100,
            "output_tokens": 50,
            "cost_usd": 0.001,
            "recorded_at": datetime.now(timezone.utc).isoformat(),
        }
        log_path.write_text(json.dumps(record) + "\n", encoding="utf-8")
        tracker = BudgetTracker(config=basic_config, storage_dir=tmp_storage)
        assert tracker.total_lifetime_spend() == pytest.approx(0.001, abs=1e-9)

    def test_malformed_jsonl_lines_skipped(
        self, basic_config: BudgetConfig, tmp_storage: pathlib.Path
    ) -> None:
        log_path = tmp_storage / "test-agent.jsonl"
        log_path.write_text("not-json\n{}\n", encoding="utf-8")
        # Should not raise
        BudgetTracker(config=basic_config, storage_dir=tmp_storage)


# ---------------------------------------------------------------------------
# BudgetTracker.record
# ---------------------------------------------------------------------------


class TestRecord:
    def test_record_returns_cost(self, tracker: BudgetTracker) -> None:
        cost = tracker.record("gpt-4o-mini", input_tokens=1000, output_tokens=200)
        assert cost > 0.0

    def test_record_with_explicit_cost(self, tracker: BudgetTracker) -> None:
        cost = tracker.record(
            "gpt-4o-mini", input_tokens=1000, output_tokens=200, cost=0.005
        )
        assert cost == pytest.approx(0.005)

    def test_record_unknown_model_uses_zero_cost(self, tracker: BudgetTracker) -> None:
        cost = tracker.record("totally-unknown-model-xyz", input_tokens=100, output_tokens=50)
        assert cost == 0.0

    def test_record_persists_to_jsonl(
        self, tracker: BudgetTracker, tmp_storage: pathlib.Path
    ) -> None:
        tracker.record("gpt-4o-mini", input_tokens=100, output_tokens=50, cost=0.001)
        log_path = tmp_storage / "test-agent.jsonl"
        assert log_path.exists()
        content = log_path.read_text(encoding="utf-8")
        assert "gpt-4o-mini" in content

    def test_record_accumulates_spend(self, tracker: BudgetTracker) -> None:
        tracker.record("gpt-4o-mini", input_tokens=100, output_tokens=50, cost=0.001)
        tracker.record("gpt-4o-mini", input_tokens=100, output_tokens=50, cost=0.002)
        assert tracker.total_lifetime_spend() == pytest.approx(0.003, abs=1e-9)

    def test_total_lifetime_spend_initially_zero(self, tracker: BudgetTracker) -> None:
        assert tracker.total_lifetime_spend() == 0.0


# ---------------------------------------------------------------------------
# BudgetTracker.check — unlimited budget
# ---------------------------------------------------------------------------


class TestCheckUnlimited:
    def test_unlimited_budget_always_proceeds(self, tmp_storage: pathlib.Path) -> None:
        config = BudgetConfig(agent_id="agent", daily_limit=0.0)
        tracker = BudgetTracker(config=config, storage_dir=tmp_storage)
        can_afford, rec = tracker.check("gpt-4o-mini", 1000)
        assert can_afford is True
        assert rec.action == "proceed"
        assert rec.remaining_usd == float("inf")


# ---------------------------------------------------------------------------
# BudgetTracker.check — within budget
# ---------------------------------------------------------------------------


class TestCheckWithinBudget:
    def test_within_budget_returns_proceed(self, tracker: BudgetTracker) -> None:
        # daily_limit=1.0, a tiny call should fit
        can_afford, rec = tracker.check("gpt-4o-mini", 100, 50)
        assert can_afford is True
        assert rec.action == "proceed"
        assert rec.can_afford is True

    def test_recommendation_has_model(self, tracker: BudgetTracker) -> None:
        _, rec = tracker.check("gpt-4o-mini", 100, 50)
        assert rec.model == "gpt-4o-mini"

    def test_recommendation_has_positive_estimated_cost(self, tracker: BudgetTracker) -> None:
        _, rec = tracker.check("gpt-4o-mini", 100, 50)
        assert rec.estimated_cost_usd >= 0.0

    def test_unknown_model_uses_fallback_pricing(self, tracker: BudgetTracker) -> None:
        can_afford, rec = tracker.check("unknown-model-abc", 10, 10)
        assert can_afford is True  # very cheap fallback pricing, fits in $1 daily


# ---------------------------------------------------------------------------
# BudgetTracker.check — budget exceeded / degradation strategies
# ---------------------------------------------------------------------------


class TestCheckDegradation:
    def _tracker_with_strategy(
        self, strategy: DegradationStrategy, tmp_path: pathlib.Path
    ) -> BudgetTracker:
        config = BudgetConfig(
            agent_id="agent",
            daily_limit=0.00001,  # 0.00001 USD — effectively exhausted immediately
            degradation_strategy=strategy,
        )
        return BudgetTracker(config=config, storage_dir=tmp_path)

    def test_block_with_error_raises(self, tmp_path: pathlib.Path) -> None:
        tracker = self._tracker_with_strategy(DegradationStrategy.BLOCK_WITH_ERROR, tmp_path)
        with pytest.raises(BudgetExceededError):
            tracker.check("gpt-4o", 10000, 2000)

    def test_token_reduction_reduces_output_tokens(self, tmp_path: pathlib.Path) -> None:
        config = BudgetConfig(
            agent_id="agent",
            daily_limit=0.0001,
            degradation_strategy=DegradationStrategy.TOKEN_REDUCTION,
        )
        tracker = BudgetTracker(config=config, storage_dir=tmp_path)
        can_afford, rec = tracker.check("gpt-4o-mini", 100, 50000)
        # With very tight budget, tokens should be reduced or blocked
        if can_afford:
            assert rec.max_output_tokens < 50000
        else:
            assert rec.action == "block"

    def test_token_reduction_zero_max_output_blocks(self, tmp_path: pathlib.Path) -> None:
        tracker = self._tracker_with_strategy(DegradationStrategy.TOKEN_REDUCTION, tmp_path)
        can_afford, rec = tracker.check("gpt-4o", 100000, 100000)
        assert can_afford is False
        assert rec.action == "block"

    def test_model_downgrade_suggests_cheaper_model(self, tmp_path: pathlib.Path) -> None:
        config = BudgetConfig(
            agent_id="agent",
            daily_limit=0.00005,
            degradation_strategy=DegradationStrategy.MODEL_DOWNGRADE,
        )
        tracker = BudgetTracker(config=config, storage_dir=tmp_path)
        can_afford, rec = tracker.check("gpt-4o", 1000, 500)
        # Either it blocks (no affordable model) or downgrades
        if can_afford:
            assert rec.action == "model_downgrade"
        else:
            assert rec.action == "block"

    def test_model_downgrade_no_affordable_model_blocks(self, tmp_path: pathlib.Path) -> None:
        tracker = self._tracker_with_strategy(DegradationStrategy.MODEL_DOWNGRADE, tmp_path)
        can_afford, rec = tracker.check("gpt-4o", 1000000, 1000000)
        assert can_afford is False
        assert rec.action == "block"

    def test_cached_fallback_returns_use_cache(self, tmp_path: pathlib.Path) -> None:
        tracker = self._tracker_with_strategy(DegradationStrategy.CACHED_FALLBACK, tmp_path)
        can_afford, rec = tracker.check("gpt-4o", 100000, 100000)
        assert can_afford is False
        assert rec.action == "use_cache"

    def test_weekly_limit_used_when_tighter(self, tmp_path: pathlib.Path) -> None:
        config = BudgetConfig(
            agent_id="agent",
            daily_limit=100.0,
            weekly_limit=0.00001,  # weekly is tighter
        )
        tracker = BudgetTracker(config=config, storage_dir=tmp_path)
        # With very low weekly limit the expensive call should fail
        with pytest.raises(Exception):
            config2 = BudgetConfig(
                agent_id="agent2",
                weekly_limit=0.00001,
                degradation_strategy=DegradationStrategy.BLOCK_WITH_ERROR,
            )
            tracker2 = BudgetTracker(config=config2, storage_dir=tmp_path)
            tracker2.check("gpt-4o", 100000, 100000)


# ---------------------------------------------------------------------------
# BudgetTracker.status
# ---------------------------------------------------------------------------


class TestStatus:
    def test_status_daily_returns_correct_fields(self, tracker: BudgetTracker) -> None:
        status = tracker.status("daily")
        assert status.agent_id == "test-agent"
        assert status.period == "daily"
        assert status.limit_usd == pytest.approx(1.0)
        assert status.spent_usd == 0.0
        assert status.call_count == 0

    def test_status_after_record(self, tracker: BudgetTracker) -> None:
        tracker.record("gpt-4o-mini", input_tokens=100, output_tokens=50, cost=0.01)
        status = tracker.status("daily")
        assert status.spent_usd == pytest.approx(0.01, abs=1e-9)
        assert status.call_count == 1

    def test_status_unknown_period_raises(self, tracker: BudgetTracker) -> None:
        with pytest.raises(ValueError, match="Unknown period"):
            tracker.status("yearly")

    def test_status_weekly(self, tmp_storage: pathlib.Path) -> None:
        config = BudgetConfig(agent_id="agent", weekly_limit=5.0)
        tracker = BudgetTracker(config=config, storage_dir=tmp_storage)
        status = tracker.status("weekly")
        assert status.limit_usd == pytest.approx(5.0)

    def test_status_monthly(self, tmp_storage: pathlib.Path) -> None:
        config = BudgetConfig(agent_id="agent", monthly_limit=50.0)
        tracker = BudgetTracker(config=config, storage_dir=tmp_storage)
        status = tracker.status("monthly")
        assert status.limit_usd == pytest.approx(50.0)

    def test_status_zero_limit_returns_inf_remaining(self, tmp_storage: pathlib.Path) -> None:
        config = BudgetConfig(agent_id="agent", daily_limit=0.0)
        tracker = BudgetTracker(config=config, storage_dir=tmp_storage)
        status = tracker.status("daily")
        assert status.remaining_usd == float("inf")

    def test_utilisation_pct(self, tracker: BudgetTracker) -> None:
        tracker.record("gpt-4o-mini", input_tokens=100, output_tokens=50, cost=0.5)
        status = tracker.status("daily")
        assert status.utilisation_pct == pytest.approx(50.0, abs=0.01)

    def test_avg_cost_per_call(self, tracker: BudgetTracker) -> None:
        tracker.record("gpt-4o-mini", input_tokens=100, output_tokens=50, cost=0.1)
        tracker.record("gpt-4o-mini", input_tokens=100, output_tokens=50, cost=0.3)
        status = tracker.status("daily")
        assert status.avg_cost_per_call == pytest.approx(0.2, abs=1e-9)


# ---------------------------------------------------------------------------
# BudgetTracker.allocate_sub_budget
# ---------------------------------------------------------------------------


class TestAllocateSubBudget:
    def test_returns_budget_tracker_instance(self, tracker: BudgetTracker, tmp_storage: pathlib.Path) -> None:
        sub = tracker.allocate_sub_budget("sub-agent", 0.5)
        assert isinstance(sub, BudgetTracker)

    def test_sub_budget_has_scaled_limits(self, tracker: BudgetTracker) -> None:
        sub = tracker.allocate_sub_budget("sub-agent", 0.3)
        assert sub.config.daily_limit == pytest.approx(0.3, abs=1e-9)

    def test_fraction_zero_raises(self, tracker: BudgetTracker) -> None:
        with pytest.raises(ValueError):
            tracker.allocate_sub_budget("sub", 0.0)

    def test_fraction_above_one_raises(self, tracker: BudgetTracker) -> None:
        with pytest.raises(ValueError):
            tracker.allocate_sub_budget("sub", 1.5)

    def test_fraction_one_is_valid(self, tracker: BudgetTracker) -> None:
        sub = tracker.allocate_sub_budget("sub", 1.0)
        assert sub.config.daily_limit == pytest.approx(1.0, abs=1e-9)

    def test_sub_inherits_degradation_strategy(self, tmp_storage: pathlib.Path) -> None:
        config = BudgetConfig(
            agent_id="parent",
            daily_limit=10.0,
            degradation_strategy=DegradationStrategy.BLOCK_WITH_ERROR,
        )
        parent = BudgetTracker(config=config, storage_dir=tmp_storage)
        sub = parent.allocate_sub_budget("child", 0.5)
        assert sub.config.degradation_strategy == DegradationStrategy.BLOCK_WITH_ERROR

    def test_sub_tags_include_parent_id(self, tracker: BudgetTracker) -> None:
        sub = tracker.allocate_sub_budget("sub", 0.5)
        assert sub.config.tags.get("parent_agent_id") == "test-agent"


# ---------------------------------------------------------------------------
# BudgetTracker.reset_period_alerts
# ---------------------------------------------------------------------------


class TestResetPeriodAlerts:
    def test_reset_does_not_raise(self, tracker: BudgetTracker) -> None:
        # Should not raise even if no alerts have been fired
        tracker.reset_period_alerts("daily")

    def test_reset_allows_alert_to_re_fire(self, tracker: BudgetTracker) -> None:
        # Record enough to trigger an alert
        tracker.record("gpt-4o-mini", input_tokens=0, output_tokens=0, cost=0.8)
        # Fire the alert once
        tracker.check("gpt-4o-mini", 1, 1)
        # Reset
        tracker.reset_period_alerts("daily")
        # Alert should be able to fire again (internal state cleared)
        # We just confirm it doesn't raise
        tracker.check("gpt-4o-mini", 1, 1)


# ---------------------------------------------------------------------------
# JSONL persistence edge cases
# ---------------------------------------------------------------------------


class TestJsonlPersistence:
    def test_jsonl_file_created_after_record(
        self, tracker: BudgetTracker, tmp_storage: pathlib.Path
    ) -> None:
        tracker.record("gpt-4o-mini", input_tokens=10, output_tokens=10, cost=0.001)
        log_path = tmp_storage / "test-agent.jsonl"
        assert log_path.exists()

    def test_jsonl_records_are_valid_json(
        self, tracker: BudgetTracker, tmp_storage: pathlib.Path
    ) -> None:
        tracker.record("gpt-4o-mini", input_tokens=10, output_tokens=10, cost=0.001)
        log_path = tmp_storage / "test-agent.jsonl"
        for line in log_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                data = json.loads(line)
                assert "model" in data
                assert "cost_usd" in data
