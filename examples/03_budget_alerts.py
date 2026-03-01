#!/usr/bin/env python3
"""Example: Budget Alerts and Enforcement

Demonstrates setting up budget thresholds and enforcing spend
limits with soft warnings and hard blocks.

Usage:
    python examples/03_budget_alerts.py

Requirements:
    pip install agent-energy-budget
"""
from __future__ import annotations

import agent_energy_budget
from agent_energy_budget import Budget


_WARN_THRESHOLD = 0.75   # Warn at 75% utilisation
_HARD_BLOCK_THRESHOLD = 0.95  # Block at 95% utilisation


def check_with_alerts(
    budget: Budget,
    model: str,
    input_tokens: int,
    output_tokens: int,
) -> bool:
    """Check budget with soft warnings and hard blocks."""
    status = budget.status(period="daily")
    utilisation = status.utilisation_pct / 100.0

    if utilisation >= _HARD_BLOCK_THRESHOLD:
        print(f"  [HARD BLOCK] {utilisation:.1%} utilisation — call denied.")
        return False
    if utilisation >= _WARN_THRESHOLD:
        print(f"  [WARNING] {utilisation:.1%} utilisation — approaching limit.")

    can_afford, rec = budget.check(model, input_tokens, output_tokens)
    if not can_afford:
        print(f"  [DENIED] {model}: {rec.action}")
        return False
    return True


def main() -> None:
    print(f"agent-energy-budget version: {agent_energy_budget.__version__}")

    # Step 1: Create a tight $0.05 budget to trigger alerts quickly
    budget = Budget(limit=0.05, agent_id="alert-demo-agent")
    print(f"Budget: ${budget.limit:.2f}/day (warn at {_WARN_THRESHOLD:.0%}, block at {_HARD_BLOCK_THRESHOLD:.0%})")

    # Step 2: Simulate escalating calls until budget is exhausted
    models_and_tokens: list[tuple[str, int, int]] = [
        ("claude-haiku-4", 1000, 500),
        ("claude-haiku-4", 2000, 800),
        ("claude-sonnet-4", 3000, 1500),
        ("claude-sonnet-4", 4000, 2000),
        ("claude-sonnet-4", 5000, 2500),
    ]

    print("\nSimulating calls with budget alerts:")
    for model, input_tokens, output_tokens in models_and_tokens:
        allowed = check_with_alerts(budget, model, input_tokens, output_tokens)
        if allowed:
            actual_cost = budget.record(model, input_tokens, output_tokens)
            status = budget.status()
            print(f"  [RECORDED] {model}: ${actual_cost:.6f} | utilisation={status.utilisation_pct:.1f}%")

    # Step 3: Final state
    final_status = budget.status()
    print(f"\nFinal status:")
    print(f"  Spent: ${final_status.spent_usd:.6f} / ${budget.limit:.2f}")
    print(f"  Utilisation: {final_status.utilisation_pct:.1f}%")


if __name__ == "__main__":
    main()
