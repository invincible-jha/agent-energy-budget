"""Tests for agent_energy_budget.enforcement.enforcer — BudgetEnforcer."""
from __future__ import annotations

import threading

import pytest

from agent_energy_budget.enforcement.enforcer import (
    BudgetEnforcer,
    EnforcerConfig,
    EnforcerStatus,
    EnforcementResult,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def config() -> EnforcerConfig:
    return EnforcerConfig(limit_usd=1.00, agent_id="test-agent")


@pytest.fixture()
def enforcer(config: EnforcerConfig) -> BudgetEnforcer:
    return BudgetEnforcer(config)


# ===========================================================================
# EnforcerConfig validation
# ===========================================================================


class TestEnforcerConfig:
    def test_basic_creation(self) -> None:
        config = EnforcerConfig(limit_usd=5.00)
        assert config.limit_usd == 5.00
        assert config.period_label == "daily"
        assert config.allow_overrun_fraction == 0.0

    def test_negative_limit_raises(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            EnforcerConfig(limit_usd=-1.0)

    def test_zero_limit_raises(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            EnforcerConfig(limit_usd=0.0)

    def test_invalid_overrun_fraction(self) -> None:
        with pytest.raises(ValueError, match="allow_overrun_fraction"):
            EnforcerConfig(limit_usd=1.0, allow_overrun_fraction=1.5)

    def test_valid_overrun_fraction(self) -> None:
        config = EnforcerConfig(limit_usd=1.0, allow_overrun_fraction=0.05)
        assert config.allow_overrun_fraction == 0.05


# ===========================================================================
# check_and_reserve — basic allow/deny
# ===========================================================================


class TestCheckAndReserve:
    def test_allows_affordable_call(self, enforcer: BudgetEnforcer) -> None:
        result = enforcer.check_and_reserve(0.50)
        assert result.allowed is True
        assert result.reservation_id != ""
        assert result.reserved_usd == 0.50

    def test_blocks_unaffordable_call(self, enforcer: BudgetEnforcer) -> None:
        result = enforcer.check_and_reserve(2.00)
        assert result.allowed is False
        assert result.reservation_id == ""
        assert result.rejection_reason != ""
        assert "test-agent" in result.rejection_reason

    def test_zero_cost_always_allowed(self, enforcer: BudgetEnforcer) -> None:
        result = enforcer.check_and_reserve(0.0)
        assert result.allowed is True

    def test_negative_cost_raises(self, enforcer: BudgetEnforcer) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            enforcer.check_and_reserve(-0.01)

    def test_exact_limit_allowed(self, enforcer: BudgetEnforcer) -> None:
        result = enforcer.check_and_reserve(1.00)
        assert result.allowed is True

    def test_just_over_limit_blocked(self, enforcer: BudgetEnforcer) -> None:
        result = enforcer.check_and_reserve(1.001)
        assert result.allowed is False

    def test_remaining_reflects_reservation(self, enforcer: BudgetEnforcer) -> None:
        result = enforcer.check_and_reserve(0.30)
        assert result.allowed
        assert abs(result.remaining_after_reservation_usd - 0.70) < 1e-9

    def test_rejection_updates_rejected_counter(self, enforcer: BudgetEnforcer) -> None:
        enforcer.check_and_reserve(5.00)
        status = enforcer.status()
        assert status.total_calls_rejected == 1

    def test_allow_updates_allowed_counter(self, enforcer: BudgetEnforcer) -> None:
        enforcer.check_and_reserve(0.10)
        status = enforcer.status()
        assert status.total_calls_allowed == 1


# ===========================================================================
# confirm
# ===========================================================================


class TestConfirm:
    def test_confirm_with_reserved_amount(self, enforcer: BudgetEnforcer) -> None:
        result = enforcer.check_and_reserve(0.20)
        cost = enforcer.confirm(result.reservation_id)
        assert cost == 0.20
        status = enforcer.status()
        assert abs(status.spent_usd - 0.20) < 1e-9
        assert status.active_reservations == 0

    def test_confirm_with_actual_cost(self, enforcer: BudgetEnforcer) -> None:
        result = enforcer.check_and_reserve(0.20)
        cost = enforcer.confirm(result.reservation_id, actual_cost_usd=0.18)
        assert cost == 0.18
        status = enforcer.status()
        assert abs(status.spent_usd - 0.18) < 1e-9

    def test_confirm_unknown_id_raises(self, enforcer: BudgetEnforcer) -> None:
        with pytest.raises(KeyError):
            enforcer.confirm("nonexistent-id")

    def test_confirm_negative_cost_raises(self, enforcer: BudgetEnforcer) -> None:
        result = enforcer.check_and_reserve(0.10)
        with pytest.raises(ValueError, match="non-negative"):
            enforcer.confirm(result.reservation_id, actual_cost_usd=-0.01)

    def test_double_confirm_raises(self, enforcer: BudgetEnforcer) -> None:
        result = enforcer.check_and_reserve(0.10)
        enforcer.confirm(result.reservation_id)
        with pytest.raises(KeyError):
            enforcer.confirm(result.reservation_id)


# ===========================================================================
# release
# ===========================================================================


class TestRelease:
    def test_release_restores_budget(self, enforcer: BudgetEnforcer) -> None:
        result = enforcer.check_and_reserve(0.50)
        enforcer.release(result.reservation_id)
        status = enforcer.status()
        assert abs(status.remaining_usd - 1.00) < 1e-9
        assert status.active_reservations == 0

    def test_release_unknown_id_raises(self, enforcer: BudgetEnforcer) -> None:
        with pytest.raises(KeyError):
            enforcer.release("unknown-id")

    def test_release_allows_subsequent_call(self, enforcer: BudgetEnforcer) -> None:
        """After releasing a large reservation, the next call can succeed."""
        big = enforcer.check_and_reserve(0.80)
        assert big.allowed
        enforcer.release(big.reservation_id)
        # Now 0.80 is free again
        result = enforcer.check_and_reserve(0.80)
        assert result.allowed


# ===========================================================================
# record_direct
# ===========================================================================


class TestRecordDirect:
    def test_record_direct_reduces_remaining(self, enforcer: BudgetEnforcer) -> None:
        enforcer.record_direct(0.30)
        status = enforcer.status()
        assert abs(status.spent_usd - 0.30) < 1e-9

    def test_record_direct_negative_raises(self, enforcer: BudgetEnforcer) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            enforcer.record_direct(-0.01)


# ===========================================================================
# reset
# ===========================================================================


class TestReset:
    def test_reset_clears_spend(self, enforcer: BudgetEnforcer) -> None:
        enforcer.record_direct(0.50)
        enforcer.reset()
        status = enforcer.status()
        assert status.spent_usd == 0.0

    def test_reset_clears_reservations(self, enforcer: BudgetEnforcer) -> None:
        enforcer.check_and_reserve(0.30)
        enforcer.reset()
        status = enforcer.status()
        assert status.active_reservations == 0
        assert abs(status.remaining_usd - 1.00) < 1e-9

    def test_reset_clears_counters(self, enforcer: BudgetEnforcer) -> None:
        enforcer.check_and_reserve(0.10)
        enforcer.check_and_reserve(5.00)
        enforcer.reset()
        status = enforcer.status()
        assert status.total_calls_allowed == 0
        assert status.total_calls_rejected == 0


# ===========================================================================
# status
# ===========================================================================


class TestStatus:
    def test_status_returns_enforcer_status(self, enforcer: BudgetEnforcer) -> None:
        status = enforcer.status()
        assert isinstance(status, EnforcerStatus)

    def test_initial_status(self, enforcer: BudgetEnforcer) -> None:
        status = enforcer.status()
        assert status.agent_id == "test-agent"
        assert status.limit_usd == 1.00
        assert status.spent_usd == 0.0
        assert status.reserved_usd == 0.0
        assert abs(status.remaining_usd - 1.00) < 1e-9
        assert status.active_reservations == 0

    def test_status_after_reservation(self, enforcer: BudgetEnforcer) -> None:
        enforcer.check_and_reserve(0.40)
        status = enforcer.status()
        assert abs(status.reserved_usd - 0.40) < 1e-9
        assert abs(status.remaining_usd - 0.60) < 1e-9
        assert status.active_reservations == 1


# ===========================================================================
# Overrun grace
# ===========================================================================


class TestOverrunGrace:
    def test_grace_allows_slight_overrun(self) -> None:
        config = EnforcerConfig(
            limit_usd=1.00, allow_overrun_fraction=0.10, agent_id="grace-agent"
        )
        enforcer = BudgetEnforcer(config)
        enforcer.record_direct(0.95)
        # Remaining = 0.05; grace = 0.10; allowable = 0.15
        result = enforcer.check_and_reserve(0.12)
        assert result.allowed

    def test_grace_still_blocks_excessive_overrun(self) -> None:
        config = EnforcerConfig(
            limit_usd=1.00, allow_overrun_fraction=0.05, agent_id="grace-agent2"
        )
        enforcer = BudgetEnforcer(config)
        enforcer.record_direct(0.99)
        # Remaining = 0.01; grace = 0.05; allowable = 0.06
        result = enforcer.check_and_reserve(0.10)
        assert not result.allowed


# ===========================================================================
# Thread safety
# ===========================================================================


class TestThreadSafety:
    def test_concurrent_reservations_respect_limit(self) -> None:
        """Multiple threads should not collectively exceed the limit."""
        config = EnforcerConfig(limit_usd=1.00, agent_id="concurrent")
        enforcer = BudgetEnforcer(config)
        allowed_results: list[EnforcementResult] = []
        lock = threading.Lock()

        def try_reserve() -> None:
            result = enforcer.check_and_reserve(0.20)
            if result.allowed:
                with lock:
                    allowed_results.append(result)

        threads = [threading.Thread(target=try_reserve) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # At most 5 calls of $0.20 fit into $1.00
        assert len(allowed_results) <= 5

    def test_concurrent_confirm_and_reserve(self) -> None:
        """Confirm+reserve interleaved from multiple threads is safe."""
        config = EnforcerConfig(limit_usd=10.00, agent_id="interleave")
        enforcer = BudgetEnforcer(config)
        errors: list[Exception] = []

        def worker() -> None:
            try:
                result = enforcer.check_and_reserve(0.10)
                if result.allowed:
                    enforcer.confirm(result.reservation_id, actual_cost_usd=0.09)
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"Thread errors: {errors}"
