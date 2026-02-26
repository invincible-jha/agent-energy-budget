"""Unit tests for agent_energy_budget.budget.allocator."""
from __future__ import annotations

import pytest

from agent_energy_budget.budget.allocator import AllocationResult, BudgetAllocator


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def allocator() -> BudgetAllocator:
    return BudgetAllocator()


# ---------------------------------------------------------------------------
# AllocationResult dataclass
# ---------------------------------------------------------------------------


class TestAllocationResult:
    def test_is_frozen(self) -> None:
        result = AllocationResult(
            allocations={"a": 1.0},
            total_allocated=1.0,
            total_budget=1.0,
            unallocated=0.0,
        )
        with pytest.raises((AttributeError, TypeError)):
            result.total_budget = 99.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# BudgetAllocator.allocate — validation
# ---------------------------------------------------------------------------


class TestAllocateValidation:
    def test_negative_total_raises(self, allocator: BudgetAllocator) -> None:
        with pytest.raises(ValueError, match="total budget"):
            allocator.allocate(total=-1.0, agents=["a"])

    def test_empty_agents_raises(self, allocator: BudgetAllocator) -> None:
        with pytest.raises(ValueError, match="empty"):
            allocator.allocate(total=10.0, agents=[])

    def test_negative_min_per_agent_raises(self, allocator: BudgetAllocator) -> None:
        with pytest.raises(ValueError, match="min_per_agent"):
            allocator.allocate(total=10.0, agents=["a"], min_per_agent=-1.0)

    def test_minimum_exceeds_total_raises(self, allocator: BudgetAllocator) -> None:
        with pytest.raises(ValueError, match="exceeds total budget"):
            allocator.allocate(total=5.0, agents=["a", "b"], min_per_agent=3.0)

    def test_zero_total_is_valid(self, allocator: BudgetAllocator) -> None:
        result = allocator.allocate(total=0.0, agents=["a", "b"])
        assert result.total_budget == 0.0


# ---------------------------------------------------------------------------
# BudgetAllocator.allocate — even split
# ---------------------------------------------------------------------------


class TestAllocateEvenSplit:
    def test_single_agent_gets_all(self, allocator: BudgetAllocator) -> None:
        result = allocator.allocate(total=10.0, agents=["a"])
        assert result.allocations["a"] == pytest.approx(10.0)

    def test_two_agents_equal_split(self, allocator: BudgetAllocator) -> None:
        result = allocator.allocate(total=10.0, agents=["a", "b"])
        assert result.allocations["a"] == pytest.approx(5.0, abs=1e-8)
        assert result.allocations["b"] == pytest.approx(5.0, abs=1e-8)

    def test_three_agents_even_split(self, allocator: BudgetAllocator) -> None:
        result = allocator.allocate(total=9.0, agents=["a", "b", "c"])
        for agent in ["a", "b", "c"]:
            assert result.allocations[agent] == pytest.approx(3.0, abs=1e-8)


# ---------------------------------------------------------------------------
# BudgetAllocator.allocate — weighted split
# ---------------------------------------------------------------------------


class TestAllocateWeighted:
    def test_weighted_allocation(self, allocator: BudgetAllocator) -> None:
        result = allocator.allocate(
            total=10.0,
            agents=["researcher", "writer", "reviewer"],
            weights={"researcher": 2.0, "writer": 2.0, "reviewer": 1.0},
        )
        assert result.allocations["researcher"] == pytest.approx(4.0, abs=1e-8)
        assert result.allocations["writer"] == pytest.approx(4.0, abs=1e-8)
        assert result.allocations["reviewer"] == pytest.approx(2.0, abs=1e-8)

    def test_unlisted_agent_gets_weight_one(self, allocator: BudgetAllocator) -> None:
        result = allocator.allocate(
            total=10.0,
            agents=["a", "b"],
            weights={"a": 3.0},
        )
        # a has weight 3.0, b gets default 1.0 → a gets 7.5, b gets 2.5
        assert result.allocations["a"] == pytest.approx(7.5, abs=1e-8)
        assert result.allocations["b"] == pytest.approx(2.5, abs=1e-8)

    def test_all_zero_weights_falls_back_to_even(self, allocator: BudgetAllocator) -> None:
        result = allocator.allocate(
            total=10.0,
            agents=["a", "b"],
            weights={"a": 0.0, "b": 0.0},
        )
        assert result.allocations["a"] == pytest.approx(5.0, abs=1e-8)
        assert result.allocations["b"] == pytest.approx(5.0, abs=1e-8)


# ---------------------------------------------------------------------------
# BudgetAllocator.allocate — min_per_agent
# ---------------------------------------------------------------------------


class TestAllocateMinPerAgent:
    def test_min_per_agent_applied(self, allocator: BudgetAllocator) -> None:
        result = allocator.allocate(
            total=10.0,
            agents=["a", "b", "c"],
            min_per_agent=1.0,
        )
        for agent in ["a", "b", "c"]:
            assert result.allocations[agent] >= 1.0

    def test_total_allocated_equals_total_budget(self, allocator: BudgetAllocator) -> None:
        result = allocator.allocate(total=10.0, agents=["a", "b", "c"])
        assert result.total_allocated == pytest.approx(result.total_budget, abs=1e-7)


# ---------------------------------------------------------------------------
# BudgetAllocator.allocate — result metadata
# ---------------------------------------------------------------------------


class TestAllocateMetadata:
    def test_total_budget_preserved(self, allocator: BudgetAllocator) -> None:
        result = allocator.allocate(total=7.5, agents=["x"])
        assert result.total_budget == pytest.approx(7.5)

    def test_unallocated_near_zero_for_clean_split(self, allocator: BudgetAllocator) -> None:
        result = allocator.allocate(total=10.0, agents=["a", "b"])
        assert abs(result.unallocated) < 1e-6


# ---------------------------------------------------------------------------
# BudgetAllocator.rebalance
# ---------------------------------------------------------------------------


class TestRebalance:
    def test_rebalance_pools_unused_budget(self, allocator: BudgetAllocator) -> None:
        remaining = {"a": 2.0, "b": 0.0, "c": 3.0}
        result = allocator.rebalance(remaining_by_agent=remaining, agents=["a", "b", "c"])
        assert result.total_budget == pytest.approx(5.0, abs=1e-8)

    def test_rebalance_exclude_agents_with_remaining(self, allocator: BudgetAllocator) -> None:
        remaining = {"a": 2.0, "b": 0.0, "c": 3.0}
        result = allocator.rebalance(
            remaining_by_agent=remaining,
            agents=["a", "b", "c"],
            exclude_agents_with_remaining=True,
        )
        # Only "b" should receive budget (a and c have remaining > 0)
        assert result.allocations.get("a", 0.0) == pytest.approx(0.0)
        assert result.allocations.get("c", 0.0) == pytest.approx(0.0)
        assert result.allocations.get("b", 0.0) > 0

    def test_rebalance_all_agents_have_remaining_returns_zero(
        self, allocator: BudgetAllocator
    ) -> None:
        remaining = {"a": 1.0, "b": 2.0}
        result = allocator.rebalance(
            remaining_by_agent=remaining,
            agents=["a", "b"],
            exclude_agents_with_remaining=True,
        )
        assert result.total_allocated == pytest.approx(0.0)

    def test_rebalance_negative_remaining_treated_as_zero(
        self, allocator: BudgetAllocator
    ) -> None:
        remaining = {"a": -1.0, "b": 2.0}
        result = allocator.rebalance(remaining_by_agent=remaining, agents=["a", "b"])
        # negative remaining for "a" is treated as 0 in pool
        assert result.total_budget == pytest.approx(2.0, abs=1e-8)


# ---------------------------------------------------------------------------
# BudgetAllocator.fractional_allocation
# ---------------------------------------------------------------------------


class TestFractionalAllocation:
    def test_basic_fractional_allocation(self, allocator: BudgetAllocator) -> None:
        result = allocator.fractional_allocation(
            parent_budget=100.0,
            fractions={"a": 0.3, "b": 0.5},
        )
        assert result.allocations["a"] == pytest.approx(30.0, abs=1e-8)
        assert result.allocations["b"] == pytest.approx(50.0, abs=1e-8)
        assert result.unallocated == pytest.approx(20.0, abs=1e-6)

    def test_fractions_summing_to_one(self, allocator: BudgetAllocator) -> None:
        result = allocator.fractional_allocation(
            parent_budget=10.0,
            fractions={"a": 0.6, "b": 0.4},
        )
        assert result.total_allocated == pytest.approx(10.0, abs=1e-6)

    def test_negative_fraction_raises(self, allocator: BudgetAllocator) -> None:
        with pytest.raises(ValueError, match="fractions must be >= 0"):
            allocator.fractional_allocation(
                parent_budget=10.0,
                fractions={"a": -0.1},
            )

    def test_fractions_exceeding_one_raises(self, allocator: BudgetAllocator) -> None:
        with pytest.raises(ValueError, match="exceeds 1.0"):
            allocator.fractional_allocation(
                parent_budget=10.0,
                fractions={"a": 0.7, "b": 0.5},
            )

    def test_fractions_equal_to_one_exactly_is_valid(self, allocator: BudgetAllocator) -> None:
        result = allocator.fractional_allocation(
            parent_budget=10.0,
            fractions={"a": 0.5, "b": 0.5},
        )
        assert result.total_allocated == pytest.approx(10.0, abs=1e-6)

    def test_total_budget_preserved(self, allocator: BudgetAllocator) -> None:
        result = allocator.fractional_allocation(
            parent_budget=50.0,
            fractions={"x": 0.2},
        )
        assert result.total_budget == pytest.approx(50.0)
