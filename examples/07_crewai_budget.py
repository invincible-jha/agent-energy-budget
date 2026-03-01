#!/usr/bin/env python3
"""Example: CrewAI Budget Integration

Demonstrates using agent-energy-budget to track and enforce cost
limits during CrewAI crew execution.

Usage:
    python examples/07_crewai_budget.py

Requirements:
    pip install agent-energy-budget crewai
"""
from __future__ import annotations

try:
    from crewai import Agent, Task, Crew, Process
    _CREWAI_AVAILABLE = True
except ImportError:
    _CREWAI_AVAILABLE = False

import agent_energy_budget
from agent_energy_budget import Budget


def run_budgeted_crew(
    task_description: str,
    budget: Budget,
    model: str = "claude-haiku-4",
    estimated_tokens_in: int = 500,
    estimated_tokens_out: int = 300,
) -> str | None:
    """Run a CrewAI task only if budget allows."""
    can_afford, rec = budget.check(model, estimated_tokens_in, estimated_tokens_out)
    if not can_afford:
        return f"[BUDGET DENIED] {rec.action}"

    if _CREWAI_AVAILABLE:
        agent = Agent(
            role="Analyst",
            goal="Complete analytical tasks efficiently",
            backstory="Expert AI analyst.",
            verbose=False,
        )
        task = Task(description=task_description, agent=agent, expected_output="Concise analysis")
        crew = Crew(agents=[agent], tasks=[task], process=Process.sequential, verbose=False)
        result = str(crew.kickoff())
    else:
        result = f"[stub] Completed: {task_description[:40]}"

    cost = budget.record(model, estimated_tokens_in, estimated_tokens_out)
    return f"[${cost:.6f}] {result[:60]}"


def main() -> None:
    print(f"agent-energy-budget version: {agent_energy_budget.__version__}")

    if not _CREWAI_AVAILABLE:
        print("crewai not installed — using stub execution.")
        print("Install with: pip install crewai")

    # Step 1: Set up budget
    budget = Budget(limit=0.06, agent_id="crewai-budget-demo")
    print(f"Budget: ${budget.limit:.2f}/day")

    # Step 2: Submit tasks to the budgeted crew runner
    tasks: list[tuple[str, int, int]] = [
        ("Analyse market trends in the renewable energy sector.", 600, 400),
        ("Draft a short executive summary of Q3 performance.", 800, 500),
        ("Identify top three risks in the current supply chain.", 500, 300),
        ("Produce a SWOT analysis for the new product line.", 1000, 600),
        ("Summarise key takeaways from the annual report.", 1200, 700),
    ]

    print("\nCrewAI task execution with budget:")
    for task_desc, est_in, est_out in tasks:
        result = run_budgeted_crew(task_desc, budget, estimated_tokens_in=est_in, estimated_tokens_out=est_out)
        print(f"  -> {result}")
        status = budget.status()
        if status.utilisation_pct >= 90:
            print(f"  [WARNING] Budget at {status.utilisation_pct:.1f}% utilisation")

    # Step 3: Final report
    final = budget.status()
    print(f"\nFinal budget: spent=${final.spent_usd:.6f} remaining=${final.remaining_usd:.6f} ({final.utilisation_pct:.1f}%)")


if __name__ == "__main__":
    main()
