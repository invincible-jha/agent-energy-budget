"""Test that the 3-line quickstart API works for agent-energy-budget."""
from __future__ import annotations


def test_quickstart_import() -> None:
    from agent_energy_budget import Budget

    budget = Budget()
    assert budget is not None


def test_quickstart_unlimited_budget() -> None:
    from agent_energy_budget import Budget

    budget = Budget()
    can_afford, rec = budget.check("claude-haiku-4", 1000)
    assert can_afford is True
    assert rec is not None


def test_quickstart_with_limit() -> None:
    from agent_energy_budget import Budget

    budget = Budget(limit=10.00)
    can_afford, rec = budget.check("claude-haiku-4", 100)
    assert isinstance(can_afford, bool)
    assert rec is not None


def test_quickstart_record() -> None:
    from agent_energy_budget import Budget

    budget = Budget(limit=5.00)
    cost = budget.record("claude-haiku-4", 100, 50, cost=0.001)
    assert cost == 0.001


def test_quickstart_status() -> None:
    from agent_energy_budget import Budget

    budget = Budget(limit=5.00, agent_id="test-qs-agent")
    status = budget.status("daily")
    assert status is not None
    assert status.agent_id == "test-qs-agent"


def test_quickstart_repr() -> None:
    from agent_energy_budget import Budget

    budget = Budget(limit=3.00, agent_id="my-agent")
    text = repr(budget)
    assert "Budget" in text
    assert "3.0" in text
