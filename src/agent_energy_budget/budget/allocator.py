"""Budget allocation across multi-agent workflows.

BudgetAllocator distributes a total USD budget across a set of agents
using configurable weights. It also supports rebalancing when one or more
agents return unused budget to the pool.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AllocationResult:
    """The result of a single budget allocation run.

    Parameters
    ----------
    allocations:
        Mapping of agent_id -> allocated USD amount.
    total_allocated:
        Sum of all allocated amounts.
    total_budget:
        The original total budget that was distributed.
    unallocated:
        Any remainder due to rounding (typically < 1 cent).
    """

    allocations: dict[str, float]
    total_allocated: float
    total_budget: float
    unallocated: float


class BudgetAllocator:
    """Distribute and rebalance budgets across multi-agent workflows.

    The allocator is stateless — each call to :meth:`allocate` or
    :meth:`rebalance` is independent with no shared mutable state.

    Examples
    --------
    >>> allocator = BudgetAllocator()
    >>> result = allocator.allocate(
    ...     total=10.0,
    ...     agents=["researcher", "writer", "reviewer"],
    ...     weights={"researcher": 2.0, "writer": 2.0, "reviewer": 1.0},
    ... )
    >>> result.allocations
    {'researcher': 4.0, 'writer': 4.0, 'reviewer': 2.0}
    """

    # ------------------------------------------------------------------
    # Core allocation
    # ------------------------------------------------------------------

    def allocate(
        self,
        total: float,
        agents: list[str],
        weights: dict[str, float] | None = None,
        *,
        min_per_agent: float = 0.0,
    ) -> AllocationResult:
        """Distribute *total* USD across *agents* using *weights*.

        When *weights* is None, the budget is split evenly.
        Agents not present in *weights* receive a default weight of 1.0.

        Parameters
        ----------
        total:
            Total budget in USD to distribute.
        agents:
            List of agent identifiers that will receive an allocation.
        weights:
            Relative weight per agent. Values need not sum to 1 — they are
            normalised internally. Default weight for unlisted agents is 1.0.
        min_per_agent:
            Minimum guaranteed allocation per agent in USD. Applied before
            proportional distribution of the remainder.

        Returns
        -------
        AllocationResult
            Per-agent allocations and metadata.

        Raises
        ------
        ValueError
            If *total* is negative, *agents* is empty, or *min_per_agent*
            is negative or causes the total minimum to exceed *total*.
        """
        if total < 0:
            raise ValueError(f"total budget must be >= 0; got {total}")
        if not agents:
            raise ValueError("agents list must not be empty")
        if min_per_agent < 0:
            raise ValueError(f"min_per_agent must be >= 0; got {min_per_agent}")

        minimum_total = min_per_agent * len(agents)
        if minimum_total > total:
            raise ValueError(
                f"min_per_agent={min_per_agent} * {len(agents)} agents = "
                f"{minimum_total:.4f} exceeds total budget {total:.4f}"
            )

        effective_weights = weights or {}
        resolved: dict[str, float] = {
            agent: effective_weights.get(agent, 1.0) for agent in agents
        }
        weight_sum = sum(resolved.values())

        if weight_sum == 0:
            # All weights are zero — fall back to even split
            resolved = {agent: 1.0 for agent in agents}
            weight_sum = float(len(agents))

        distributable = total - minimum_total
        allocations: dict[str, float] = {}
        for agent in agents:
            proportional = (resolved[agent] / weight_sum) * distributable
            allocations[agent] = round(min_per_agent + proportional, 8)

        total_allocated = sum(allocations.values())
        unallocated = round(total - total_allocated, 8)

        return AllocationResult(
            allocations=allocations,
            total_allocated=round(total_allocated, 8),
            total_budget=total,
            unallocated=unallocated,
        )

    # ------------------------------------------------------------------
    # Rebalancing
    # ------------------------------------------------------------------

    def rebalance(
        self,
        remaining_by_agent: dict[str, float],
        agents: list[str],
        weights: dict[str, float] | None = None,
        *,
        exclude_agents_with_remaining: bool = False,
    ) -> AllocationResult:
        """Redistribute unused budget from over-allocated agents.

        Calculates the total pool of unused budget and distributes it
        to agents that need more resources.

        Parameters
        ----------
        remaining_by_agent:
            Current remaining budget per agent (may include agents not
            in the active *agents* list — their surplus is still pooled).
        agents:
            The active agents to redistribute budget to.
        weights:
            Optional weights for the redistribution (same semantics as
            :meth:`allocate`).
        exclude_agents_with_remaining:
            When True, agents that already have remaining budget > 0 are
            excluded from receiving more (avoids double-allocation).

        Returns
        -------
        AllocationResult
            New per-agent allocations from the rebalanced pool.

        Raises
        ------
        ValueError
            If *agents* is empty.
        """
        pool = sum(max(0.0, v) for v in remaining_by_agent.values())

        target_agents = agents
        if exclude_agents_with_remaining:
            target_agents = [
                a for a in agents if remaining_by_agent.get(a, 0.0) <= 0.0
            ]

        if not target_agents:
            # Everyone has budget — return zero distribution
            return AllocationResult(
                allocations={a: 0.0 for a in agents},
                total_allocated=0.0,
                total_budget=pool,
                unallocated=pool,
            )

        return self.allocate(pool, target_agents, weights)

    # ------------------------------------------------------------------
    # Fractional sub-budget helpers
    # ------------------------------------------------------------------

    def fractional_allocation(
        self,
        parent_budget: float,
        fractions: dict[str, float],
    ) -> AllocationResult:
        """Allocate sub-budgets as explicit fractions of a parent budget.

        Parameters
        ----------
        parent_budget:
            The parent budget in USD.
        fractions:
            Mapping of agent_id -> fraction (0.0 to 1.0). Must sum to <= 1.0.

        Returns
        -------
        AllocationResult
            Allocated amounts.

        Raises
        ------
        ValueError
            If any fraction is negative or the sum exceeds 1.0.
        """
        if any(f < 0 for f in fractions.values()):
            raise ValueError("All fractions must be >= 0.0")
        total_fraction = sum(fractions.values())
        if total_fraction > 1.0 + 1e-9:
            raise ValueError(
                f"Fractions sum to {total_fraction:.6f} which exceeds 1.0"
            )

        allocations: dict[str, float] = {
            agent: round(parent_budget * fraction, 8)
            for agent, fraction in fractions.items()
        }
        total_allocated = sum(allocations.values())
        return AllocationResult(
            allocations=allocations,
            total_allocated=round(total_allocated, 8),
            total_budget=parent_budget,
            unallocated=round(parent_budget - total_allocated, 8),
        )
