#!/usr/bin/env python3
"""Example: Multi-Model Cost Tracking

Demonstrates tracking costs across multiple LLM models with a
shared budget and per-model reporting.

Usage:
    python examples/02_multi_model_tracking.py

Requirements:
    pip install agent-energy-budget
"""
from __future__ import annotations

import agent_energy_budget
from agent_energy_budget import Budget


def simulate_llm_calls(budget: Budget) -> list[tuple[str, int, int, float]]:
    """Simulate a series of LLM calls across multiple models."""
    call_log: list[tuple[str, int, int, float]] = []
    calls: list[tuple[str, int, int]] = [
        ("claude-haiku-4", 500, 200),
        ("claude-haiku-4", 1200, 400),
        ("claude-sonnet-4", 800, 600),
        ("gpt-4o-mini", 300, 150),
        ("claude-haiku-4", 2000, 800),
        ("claude-sonnet-4", 1500, 900),
        ("gpt-4o-mini", 600, 300),
    ]

    for model, input_tokens, output_tokens in calls:
        can_afford, rec = budget.check(model, input_tokens, output_tokens)
        if can_afford:
            actual_cost = budget.record(model, input_tokens, output_tokens)
            call_log.append((model, input_tokens, output_tokens, actual_cost))
        else:
            print(f"  [SKIPPED] {model} — budget exceeded: {rec.action}")

    return call_log


def main() -> None:
    print(f"agent-energy-budget version: {agent_energy_budget.__version__}")

    # Step 1: Create a $0.10 budget for demo
    budget = Budget(limit=0.10, agent_id="multi-model-agent")
    print(f"Budget: ${budget.limit:.2f}/day")

    # Step 2: Run simulated calls
    print("\nSimulating LLM calls:")
    call_log = simulate_llm_calls(budget)

    # Step 3: Per-model breakdown
    model_costs: dict[str, float] = {}
    model_calls: dict[str, int] = {}
    for model, _, _, cost in call_log:
        model_costs[model] = model_costs.get(model, 0.0) + cost
        model_calls[model] = model_calls.get(model, 0) + 1

    print(f"\nPer-model breakdown ({len(call_log)} calls recorded):")
    for model in sorted(model_costs):
        print(f"  [{model}] ${model_costs[model]:.6f} over {model_calls[model]} call(s)")

    # Step 4: Final budget status
    status = budget.status(period="daily")
    print(f"\nFinal budget status:")
    print(f"  Spent: ${status.spent_usd:.6f} / ${budget.limit:.2f}")
    print(f"  Utilisation: {status.utilisation_pct:.1f}%")
    print(f"  Remaining: ${status.remaining_usd:.6f}")


if __name__ == "__main__":
    main()
