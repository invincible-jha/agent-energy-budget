#!/usr/bin/env python3
"""Example: Quickstart

Demonstrates the minimal setup for agent-energy-budget using the
Budget convenience class to check and record LLM spend.

Usage:
    python examples/01_quickstart.py

Requirements:
    pip install agent-energy-budget
"""
from __future__ import annotations

import agent_energy_budget
from agent_energy_budget import Budget


def main() -> None:
    print(f"agent-energy-budget version: {agent_energy_budget.__version__}")

    # Step 1: Create a daily budget of $1.00
    budget = Budget(limit=1.00, agent_id="quickstart-agent")
    print(f"Budget: ${budget.limit:.2f}/day for '{budget.agent_id}'")

    # Step 2: Check if a call fits the budget
    model = "claude-haiku-4"
    input_tokens = 1000
    output_tokens = 500

    can_afford, recommendation = budget.check(model, input_tokens, output_tokens)
    print(f"\nPre-call check:")
    print(f"  Model: {model}")
    print(f"  Tokens: {input_tokens} in / {output_tokens} out")
    print(f"  Can afford: {can_afford}")
    print(f"  Estimated cost: ${recommendation.estimated_cost_usd:.6f}")
    print(f"  Recommendation: {recommendation.action}")

    # Step 3: Record the actual spend after the call
    if can_afford:
        actual_cost = budget.record(model, input_tokens, output_tokens)
        print(f"\nRecorded cost: ${actual_cost:.6f}")

    # Step 4: Check budget status
    status = budget.status(period="daily")
    print(f"\nBudget status (daily):")
    print(f"  Spent: ${status.spent_usd:.6f}")
    print(f"  Remaining: ${status.remaining_usd:.6f}")
    print(f"  Utilisation: {status.utilisation_pct:.1f}%")

    print("\nQuickstart complete.")


if __name__ == "__main__":
    main()
