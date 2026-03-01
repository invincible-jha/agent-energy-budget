#!/usr/bin/env python3
"""Example: Multi-Agent Budget Allocation

Demonstrates allocating a shared budget across multiple agents
with per-agent limits and consolidated reporting.

Usage:
    python examples/05_multi_agent_budget.py

Requirements:
    pip install agent-energy-budget
"""
from __future__ import annotations

import agent_energy_budget
from agent_energy_budget import Budget


class AgentBudgetPool:
    """Shared budget pool with per-agent allocations."""

    def __init__(self, total_limit_usd: float, agents: list[str]) -> None:
        self._total = total_limit_usd
        per_agent = total_limit_usd / len(agents)
        self._budgets: dict[str, Budget] = {
            agent_id: Budget(limit=per_agent, agent_id=agent_id)
            for agent_id in agents
        }

    def check(self, agent_id: str, model: str, input_tokens: int, output_tokens: int) -> bool:
        budget = self._budgets.get(agent_id)
        if not budget:
            return False
        can_afford, _ = budget.check(model, input_tokens, output_tokens)
        return can_afford

    def record(self, agent_id: str, model: str, input_tokens: int, output_tokens: int) -> float:
        budget = self._budgets[agent_id]
        return budget.record(model, input_tokens, output_tokens)

    def summary(self) -> dict[str, dict[str, float]]:
        return {
            agent_id: {
                "spent": budget.status().spent_usd,
                "remaining": budget.status().remaining_usd,
                "utilisation_pct": budget.status().utilisation_pct,
            }
            for agent_id, budget in self._budgets.items()
        }

    @property
    def total_spent(self) -> float:
        return sum(budget.status().spent_usd for budget in self._budgets.values())


def main() -> None:
    print(f"agent-energy-budget version: {agent_energy_budget.__version__}")

    # Step 1: Create pool with $0.12 shared across 3 agents
    agents = ["researcher", "writer", "reviewer"]
    pool = AgentBudgetPool(total_limit_usd=0.12, agents=agents)
    print(f"Budget pool: ${pool._total:.2f} shared across {len(agents)} agents")

    # Step 2: Simulate agent activities
    activities: list[tuple[str, str, int, int]] = [
        ("researcher", "claude-haiku-4", 2000, 800),
        ("writer", "claude-sonnet-4", 3000, 1500),
        ("reviewer", "claude-haiku-4", 1000, 400),
        ("researcher", "claude-haiku-4", 1500, 600),
        ("writer", "claude-haiku-4", 2500, 1000),
        ("reviewer", "claude-sonnet-4", 4000, 2000),
    ]

    print("\nAgent activities:")
    for agent_id, model, input_tokens, output_tokens in activities:
        if pool.check(agent_id, model, input_tokens, output_tokens):
            cost = pool.record(agent_id, model, input_tokens, output_tokens)
            print(f"  [{agent_id}] {model}: ${cost:.6f}")
        else:
            print(f"  [{agent_id}] {model}: BUDGET EXCEEDED")

    # Step 3: Final summary
    summary = pool.summary()
    print(f"\nAgent budget summary:")
    for agent_id, stats in summary.items():
        print(f"  [{agent_id}] spent=${stats['spent']:.6f} "
              f"remaining=${stats['remaining']:.6f} "
              f"({stats['utilisation_pct']:.1f}%)")
    print(f"\nPool total spent: ${pool.total_spent:.6f} / ${pool._total:.2f}")


if __name__ == "__main__":
    main()
