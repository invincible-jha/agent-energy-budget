"""Unit tests for agent_energy_budget.budget.sub_budget."""
from __future__ import annotations

import pathlib

import pytest

from agent_energy_budget.budget.config import BudgetConfig
from agent_energy_budget.budget.sub_budget import SubBudget
from agent_energy_budget.budget.tracker import BudgetTracker


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_storage(tmp_path: pathlib.Path) -> pathlib.Path:
    return tmp_path


@pytest.fixture
def parent_tracker(tmp_storage: pathlib.Path) -> BudgetTracker:
    config = BudgetConfig(agent_id="parent", daily_limit=10.0)
    return BudgetTracker(config=config, storage_dir=tmp_storage)


@pytest.fixture
def sub_budget(parent_tracker: BudgetTracker) -> SubBudget:
    return SubBudget(parent_tracker=parent_tracker, sub_id="researcher", allocated_usd=3.0)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestSubBudgetInit:
    def test_negative_allocated_raises(self, parent_tracker: BudgetTracker) -> None:
        with pytest.raises(ValueError, match="allocated_usd"):
            SubBudget(parent_tracker=parent_tracker, sub_id="sub", allocated_usd=-1.0)

    def test_zero_allocated_is_valid(self, parent_tracker: BudgetTracker) -> None:
        sub = SubBudget(parent_tracker=parent_tracker, sub_id="sub", allocated_usd=0.0)
        assert sub.allocated_usd == 0.0

    def test_sub_id_property(self, sub_budget: SubBudget) -> None:
        assert sub_budget.sub_id == "researcher"

    def test_allocated_usd_property(self, sub_budget: SubBudget) -> None:
        assert sub_budget.allocated_usd == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# record_cost
# ---------------------------------------------------------------------------


class TestRecordCost:
    def test_negative_amount_raises(self, sub_budget: SubBudget) -> None:
        with pytest.raises(ValueError, match="amount_usd"):
            sub_budget.record_cost(-0.01)

    def test_zero_amount_is_valid(self, sub_budget: SubBudget) -> None:
        sub_budget.record_cost(0.0)
        assert sub_budget.status().spent_usd == 0.0

    def test_record_cost_accumulates_spend(self, sub_budget: SubBudget) -> None:
        sub_budget.record_cost(0.10)
        sub_budget.record_cost(0.20)
        assert sub_budget.status().spent_usd == pytest.approx(0.30, abs=1e-9)

    def test_record_cost_forwards_to_parent(
        self, sub_budget: SubBudget, parent_tracker: BudgetTracker
    ) -> None:
        sub_budget.record_cost(0.50)
        # Parent should have received the forwarded cost
        assert parent_tracker.total_lifetime_spend() == pytest.approx(0.50, abs=1e-9)

    def test_record_cost_with_model(self, sub_budget: SubBudget) -> None:
        sub_budget.record_cost(0.01, model="gpt-4o-mini", operation="summarise")
        assert sub_budget.status().spent_usd == pytest.approx(0.01, abs=1e-9)

    def test_record_cost_with_unknown_model(self, sub_budget: SubBudget) -> None:
        # Should not raise for unknown model
        sub_budget.record_cost(0.01, model="totally-unknown-xyz")
        assert sub_budget.status().spent_usd == pytest.approx(0.01, abs=1e-9)

    def test_record_cost_creates_entry(self, sub_budget: SubBudget) -> None:
        sub_budget.record_cost(0.05, model="gpt-4o-mini", operation="test")
        entries = sub_budget.entries()
        assert len(entries) == 1
        assert entries[0].amount_usd == pytest.approx(0.05)
        assert entries[0].model == "gpt-4o-mini"
        assert entries[0].operation == "test"


# ---------------------------------------------------------------------------
# status
# ---------------------------------------------------------------------------


class TestStatus:
    def test_status_initial_state(self, sub_budget: SubBudget) -> None:
        status = sub_budget.status()
        assert status.agent_id == "researcher"
        assert status.period == "sub_budget"
        assert status.limit_usd == pytest.approx(3.0)
        assert status.spent_usd == 0.0
        assert status.remaining_usd == pytest.approx(3.0)
        assert status.utilisation_pct == pytest.approx(0.0)
        assert status.call_count == 0
        assert status.avg_cost_per_call == 0.0

    def test_status_after_spend(self, sub_budget: SubBudget) -> None:
        sub_budget.record_cost(1.5)
        status = sub_budget.status()
        assert status.spent_usd == pytest.approx(1.5, abs=1e-9)
        assert status.remaining_usd == pytest.approx(1.5, abs=1e-9)
        assert status.utilisation_pct == pytest.approx(50.0, abs=0.01)
        assert status.call_count == 1
        assert status.avg_cost_per_call == pytest.approx(1.5, abs=1e-9)

    def test_status_zero_allocation_utilisation(self, parent_tracker: BudgetTracker) -> None:
        sub = SubBudget(parent_tracker=parent_tracker, sub_id="sub", allocated_usd=0.0)
        status = sub.status()
        assert status.utilisation_pct == 0.0


# ---------------------------------------------------------------------------
# is_within_budget
# ---------------------------------------------------------------------------


class TestIsWithinBudget:
    def test_zero_spent_within_budget(self, sub_budget: SubBudget) -> None:
        assert sub_budget.is_within_budget(2.0) is True

    def test_exact_fit_is_within_budget(self, sub_budget: SubBudget) -> None:
        assert sub_budget.is_within_budget(3.0) is True

    def test_over_budget_returns_false(self, sub_budget: SubBudget) -> None:
        assert sub_budget.is_within_budget(3.01) is False

    def test_after_partial_spend(self, sub_budget: SubBudget) -> None:
        sub_budget.record_cost(1.0)
        assert sub_budget.is_within_budget(2.0) is True
        assert sub_budget.is_within_budget(2.01) is False


# ---------------------------------------------------------------------------
# remaining_usd
# ---------------------------------------------------------------------------


class TestRemainingUsd:
    def test_initial_remaining_equals_allocated(self, sub_budget: SubBudget) -> None:
        assert sub_budget.remaining_usd() == pytest.approx(3.0)

    def test_remaining_decreases_after_spend(self, sub_budget: SubBudget) -> None:
        sub_budget.record_cost(1.0)
        assert sub_budget.remaining_usd() == pytest.approx(2.0, abs=1e-9)

    def test_remaining_can_go_negative(self, sub_budget: SubBudget) -> None:
        # record_cost doesn't enforce the limit
        sub_budget.record_cost(5.0)
        assert sub_budget.remaining_usd() < 0


# ---------------------------------------------------------------------------
# entries
# ---------------------------------------------------------------------------


class TestEntries:
    def test_entries_initially_empty(self, sub_budget: SubBudget) -> None:
        assert sub_budget.entries() == []

    def test_entries_returns_copy(self, sub_budget: SubBudget) -> None:
        sub_budget.record_cost(0.01)
        entries1 = sub_budget.entries()
        entries2 = sub_budget.entries()
        assert entries1 is not entries2

    def test_multiple_entries_recorded(self, sub_budget: SubBudget) -> None:
        sub_budget.record_cost(0.01, model="gpt-4o-mini")
        sub_budget.record_cost(0.02, model="claude-haiku-4")
        entries = sub_budget.entries()
        assert len(entries) == 2
        models = {e.model for e in entries}
        assert "gpt-4o-mini" in models
        assert "claude-haiku-4" in models
