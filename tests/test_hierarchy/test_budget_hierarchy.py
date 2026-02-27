"""Tests for agent_energy_budget.hierarchy.budget_hierarchy — HierarchicalBudget."""
from __future__ import annotations

import threading

import pytest

from agent_energy_budget.hierarchy.budget_hierarchy import (
    HierarchicalBudget,
    BudgetNode,
    HierarchyConfig,
    NodeStatus,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def simple_tree() -> HierarchicalBudget:
    """org(1000) -> eng_team(300) -> agent_1(50)."""
    budget = HierarchicalBudget(root_id="org", root_limit=1000.0)
    budget.add_node("eng_team", parent_id="org", limit=300.0)
    budget.add_node("agent_1", parent_id="eng_team", limit=50.0)
    return budget


# ===========================================================================
# HierarchyConfig
# ===========================================================================


class TestHierarchyConfig:
    def test_defaults(self) -> None:
        config = HierarchyConfig()
        assert config.allow_child_to_exceed_parent is False
        assert config.rollup_mode == "strict"

    def test_invalid_rollup_mode(self) -> None:
        with pytest.raises(ValueError, match="rollup_mode"):
            HierarchyConfig(rollup_mode="invalid")


# ===========================================================================
# HierarchicalBudget construction
# ===========================================================================


class TestHierarchicalBudgetConstruction:
    def test_basic_creation(self) -> None:
        budget = HierarchicalBudget(root_id="root", root_limit=500.0)
        assert "root" in budget
        assert len(budget) == 1

    def test_root_zero_limit_raises(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            HierarchicalBudget(root_id="root", root_limit=0.0)

    def test_root_negative_limit_raises(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            HierarchicalBudget(root_id="root", root_limit=-100.0)

    def test_root_status(self) -> None:
        budget = HierarchicalBudget(root_id="acme", root_limit=10_000.0, root_label="Acme Corp")
        status = budget.node_status("acme")
        assert status.limit_usd == 10_000.0
        assert status.spent_usd == 0.0
        assert status.parent_id is None
        assert status.label == "Acme Corp"

    def test_root_id_property(self) -> None:
        budget = HierarchicalBudget(root_id="root", root_limit=100.0)
        assert budget.root_id() == "root"


# ===========================================================================
# add_node
# ===========================================================================


class TestAddNode:
    def test_add_child_to_root(self) -> None:
        budget = HierarchicalBudget(root_id="org", root_limit=1000.0)
        budget.add_node("team_a", parent_id="org", limit=200.0)
        assert "team_a" in budget
        assert len(budget) == 2

    def test_add_grandchild(self) -> None:
        budget = HierarchicalBudget(root_id="org", root_limit=1000.0)
        budget.add_node("team", parent_id="org", limit=200.0)
        budget.add_node("agent", parent_id="team", limit=50.0)
        assert len(budget) == 3
        assert "agent" in budget

    def test_add_node_unknown_parent_raises(self) -> None:
        budget = HierarchicalBudget(root_id="org", root_limit=1000.0)
        with pytest.raises(KeyError, match="nonexistent"):
            budget.add_node("child", parent_id="nonexistent", limit=100.0)

    def test_add_duplicate_node_raises(self) -> None:
        budget = HierarchicalBudget(root_id="org", root_limit=1000.0)
        budget.add_node("team", parent_id="org", limit=100.0)
        with pytest.raises(ValueError, match="already exists"):
            budget.add_node("team", parent_id="org", limit=50.0)

    def test_child_exceeds_parent_raises_by_default(self) -> None:
        budget = HierarchicalBudget(root_id="org", root_limit=100.0)
        with pytest.raises(ValueError, match="exceeds parent"):
            budget.add_node("team", parent_id="org", limit=200.0)

    def test_child_exceeds_parent_allowed_with_config(self) -> None:
        config = HierarchyConfig(allow_child_to_exceed_parent=True)
        budget = HierarchicalBudget(root_id="org", root_limit=100.0, config=config)
        budget.add_node("team", parent_id="org", limit=200.0)  # should not raise
        assert "team" in budget

    def test_zero_limit_raises(self) -> None:
        budget = HierarchicalBudget(root_id="org", root_limit=1000.0)
        with pytest.raises(ValueError, match="positive"):
            budget.add_node("team", parent_id="org", limit=0.0)

    def test_parent_child_relationship(self) -> None:
        budget = HierarchicalBudget(root_id="org", root_limit=1000.0)
        budget.add_node("team", parent_id="org", limit=200.0)
        status = budget.node_status("team")
        assert status.parent_id == "org"
        children = budget.children("org")
        assert "team" in children


# ===========================================================================
# check_spend
# ===========================================================================


class TestCheckSpend:
    def test_check_affordable_spend(self, simple_tree: HierarchicalBudget) -> None:
        allowed, reason = simple_tree.check_spend("agent_1", 10.0)
        assert allowed
        assert reason == ""

    def test_check_blocks_exceeding_node_limit(self, simple_tree: HierarchicalBudget) -> None:
        allowed, reason = simple_tree.check_spend("agent_1", 100.0)
        assert not allowed
        assert "agent_1" in reason

    def test_check_blocks_exceeding_parent_limit(self) -> None:
        budget = HierarchicalBudget(root_id="org", root_limit=1000.0)
        budget.add_node("team", parent_id="org", limit=100.0)
        budget.add_node("agent", parent_id="team", limit=90.0)
        # Spend 95 at team level first
        budget.record_spend("team", 90.0)  # rolls up to org too
        # Now agent has 90 limit but team only has 10 remaining
        allowed, reason = budget.check_spend("agent", 20.0)
        assert not allowed
        assert "team" in reason

    def test_check_unknown_node_raises(self, simple_tree: HierarchicalBudget) -> None:
        with pytest.raises(KeyError):
            simple_tree.check_spend("nonexistent", 10.0)

    def test_check_negative_amount_raises(self, simple_tree: HierarchicalBudget) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            simple_tree.check_spend("agent_1", -5.0)

    def test_check_zero_amount_allowed(self, simple_tree: HierarchicalBudget) -> None:
        allowed, _ = simple_tree.check_spend("agent_1", 0.0)
        assert allowed

    def test_advisory_mode_always_allows(self) -> None:
        config = HierarchyConfig(rollup_mode="advisory")
        budget = HierarchicalBudget(root_id="org", root_limit=10.0, config=config)
        budget.add_node("team", parent_id="org", limit=5.0)
        allowed, _ = budget.check_spend("team", 100.0)
        assert allowed


# ===========================================================================
# record_spend
# ===========================================================================


class TestRecordSpend:
    def test_record_rolls_up_to_root(self, simple_tree: HierarchicalBudget) -> None:
        simple_tree.record_spend("agent_1", 10.0)
        assert simple_tree.node_status("agent_1").spent_usd == 10.0
        assert simple_tree.node_status("eng_team").spent_usd == 10.0
        assert simple_tree.node_status("org").spent_usd == 10.0

    def test_record_at_team_level_rolls_to_root(self, simple_tree: HierarchicalBudget) -> None:
        simple_tree.record_spend("eng_team", 50.0)
        assert simple_tree.node_status("eng_team").spent_usd == 50.0
        assert simple_tree.node_status("org").spent_usd == 50.0
        # agent_1 not in the spend path
        assert simple_tree.node_status("agent_1").spent_usd == 0.0

    def test_record_multiple_agents_accumulates_at_parent(self) -> None:
        budget = HierarchicalBudget(root_id="org", root_limit=1000.0)
        budget.add_node("team", parent_id="org", limit=500.0)
        budget.add_node("agent_a", parent_id="team", limit=100.0)
        budget.add_node("agent_b", parent_id="team", limit=100.0)
        budget.record_spend("agent_a", 30.0)
        budget.record_spend("agent_b", 40.0)
        assert budget.node_status("team").spent_usd == 70.0
        assert budget.node_status("org").spent_usd == 70.0

    def test_record_unknown_node_raises(self, simple_tree: HierarchicalBudget) -> None:
        with pytest.raises(KeyError):
            simple_tree.record_spend("nonexistent", 10.0)

    def test_record_negative_raises(self, simple_tree: HierarchicalBudget) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            simple_tree.record_spend("agent_1", -1.0)


# ===========================================================================
# node_status
# ===========================================================================


class TestNodeStatus:
    def test_utilisation_zero_initially(self, simple_tree: HierarchicalBudget) -> None:
        status = simple_tree.node_status("agent_1")
        assert status.utilisation_pct == 0.0
        assert not status.is_exhausted

    def test_utilisation_after_spend(self, simple_tree: HierarchicalBudget) -> None:
        simple_tree.record_spend("agent_1", 25.0)
        status = simple_tree.node_status("agent_1")
        assert abs(status.utilisation_pct - 50.0) < 0.01
        assert not status.is_exhausted

    def test_is_exhausted_when_full(self, simple_tree: HierarchicalBudget) -> None:
        simple_tree.record_spend("agent_1", 50.0)
        status = simple_tree.node_status("agent_1")
        assert status.is_exhausted

    def test_remaining_usd(self, simple_tree: HierarchicalBudget) -> None:
        simple_tree.record_spend("agent_1", 20.0)
        status = simple_tree.node_status("agent_1")
        assert abs(status.remaining_usd - 30.0) < 1e-9

    def test_unknown_node_raises(self, simple_tree: HierarchicalBudget) -> None:
        with pytest.raises(KeyError):
            simple_tree.node_status("nonexistent")

    def test_child_count(self, simple_tree: HierarchicalBudget) -> None:
        root_status = simple_tree.node_status("org")
        assert root_status.child_count == 1
        agent_status = simple_tree.node_status("agent_1")
        assert agent_status.child_count == 0


# ===========================================================================
# reset operations
# ===========================================================================


class TestReset:
    def test_reset_node(self, simple_tree: HierarchicalBudget) -> None:
        simple_tree.record_spend("agent_1", 20.0)
        simple_tree.reset_node("agent_1")
        assert simple_tree.node_status("agent_1").spent_usd == 0.0
        # Parent still has the rolled-up spend
        assert simple_tree.node_status("eng_team").spent_usd == 20.0

    def test_reset_all(self, simple_tree: HierarchicalBudget) -> None:
        simple_tree.record_spend("agent_1", 20.0)
        simple_tree.reset_all()
        assert simple_tree.node_status("org").spent_usd == 0.0
        assert simple_tree.node_status("eng_team").spent_usd == 0.0
        assert simple_tree.node_status("agent_1").spent_usd == 0.0

    def test_reset_unknown_raises(self, simple_tree: HierarchicalBudget) -> None:
        with pytest.raises(KeyError):
            simple_tree.reset_node("nonexistent")


# ===========================================================================
# list_nodes and children
# ===========================================================================


class TestListNodes:
    def test_list_nodes_count(self, simple_tree: HierarchicalBudget) -> None:
        nodes = simple_tree.list_nodes()
        assert len(nodes) == 3

    def test_list_nodes_sorted(self, simple_tree: HierarchicalBudget) -> None:
        nodes = simple_tree.list_nodes()
        ids = [n.node_id for n in nodes]
        assert ids == sorted(ids)

    def test_children_returns_direct_children(self, simple_tree: HierarchicalBudget) -> None:
        children = simple_tree.children("org")
        assert children == ["eng_team"]
        children_of_team = simple_tree.children("eng_team")
        assert children_of_team == ["agent_1"]
        children_of_agent = simple_tree.children("agent_1")
        assert children_of_agent == []

    def test_children_unknown_raises(self, simple_tree: HierarchicalBudget) -> None:
        with pytest.raises(KeyError):
            simple_tree.children("unknown")


# ===========================================================================
# Thread safety
# ===========================================================================


class TestThreadSafety:
    def test_concurrent_record_spend(self) -> None:
        budget = HierarchicalBudget(root_id="org", root_limit=1_000_000.0)
        budget.add_node("team", parent_id="org", limit=1_000_000.0)
        budget.add_node("agent", parent_id="team", limit=1_000_000.0)
        errors: list[Exception] = []

        def spend_worker() -> None:
            try:
                budget.record_spend("agent", 1.0)
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=spend_worker) for _ in range(100)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []
        status = budget.node_status("org")
        assert abs(status.spent_usd - 100.0) < 1e-9
