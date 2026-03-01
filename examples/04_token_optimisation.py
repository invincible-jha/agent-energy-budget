#!/usr/bin/env python3
"""Example: Token Optimisation Strategy

Demonstrates how to use budget checks to automatically select the
most cost-efficient model that fits within the remaining budget.

Usage:
    python examples/04_token_optimisation.py

Requirements:
    pip install agent-energy-budget
"""
from __future__ import annotations

import agent_energy_budget
from agent_energy_budget import Budget


MODEL_PRIORITY: list[str] = [
    "claude-haiku-4",     # cheapest
    "gpt-4o-mini",
    "claude-sonnet-4",    # most capable
]


def select_model(
    budget: Budget,
    input_tokens: int,
    output_tokens: int,
    min_capability: str = "claude-haiku-4",
) -> str | None:
    """Select the cheapest model that fits the budget."""
    start_index = MODEL_PRIORITY.index(min_capability) if min_capability in MODEL_PRIORITY else 0
    candidates = MODEL_PRIORITY[start_index:]

    for model in candidates:
        can_afford, rec = budget.check(model, input_tokens, output_tokens)
        if can_afford:
            return model
    return None


def main() -> None:
    print(f"agent-energy-budget version: {agent_energy_budget.__version__}")

    budget = Budget(limit=0.08, agent_id="optimiser-agent")
    print(f"Budget: ${budget.limit:.2f}/day")

    # Step 2: Simulate tasks with automatic model selection
    tasks: list[tuple[str, int, int, str]] = [
        ("Simple lookup", 200, 100, "claude-haiku-4"),
        ("Document analysis", 3000, 1200, "claude-haiku-4"),
        ("Complex reasoning", 5000, 2000, "claude-sonnet-4"),
        ("Quick summary", 500, 200, "claude-haiku-4"),
        ("Research synthesis", 8000, 3000, "claude-sonnet-4"),
    ]

    print("\nAuto-selecting models based on budget:")
    for task_name, input_tokens, output_tokens, min_model in tasks:
        selected = select_model(budget, input_tokens, output_tokens, min_model)
        if selected:
            actual_cost = budget.record(selected, input_tokens, output_tokens)
            status = budget.status()
            print(f"  [{task_name}] model={selected} | cost=${actual_cost:.6f} | "
                  f"remaining=${status.remaining_usd:.6f}")
        else:
            status = budget.status()
            print(f"  [{task_name}] SKIPPED — budget exhausted (${status.spent_usd:.6f} spent)")

    # Step 3: Final summary
    final = budget.status()
    print(f"\nFinal: ${final.spent_usd:.6f} spent ({final.utilisation_pct:.1f}% utilisation)")


if __name__ == "__main__":
    main()
