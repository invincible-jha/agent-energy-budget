#!/usr/bin/env python3
"""Example: LangChain Budget Integration

Demonstrates gating LangChain LLM calls with agent-energy-budget
to prevent runaway costs during chain execution.

Usage:
    python examples/06_langchain_budget.py

Requirements:
    pip install agent-energy-budget langchain langchain-openai
"""
from __future__ import annotations

try:
    from langchain.schema import HumanMessage
    from langchain_openai import ChatOpenAI
    _LANGCHAIN_AVAILABLE = True
except ImportError:
    _LANGCHAIN_AVAILABLE = False

import agent_energy_budget
from agent_energy_budget import Budget


def budgeted_llm_call(
    budget: Budget,
    model: str,
    prompt: str,
    estimated_input_tokens: int = 100,
    estimated_output_tokens: int = 200,
) -> str | None:
    """Call an LLM only if budget allows, then record actual cost."""
    can_afford, rec = budget.check(model, estimated_input_tokens, estimated_output_tokens)
    if not can_afford:
        print(f"  [BUDGET] Skipping call: {rec.action}")
        return None

    if _LANGCHAIN_AVAILABLE:
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        response = llm.invoke([HumanMessage(content=prompt)])
        output = response.content
        usage = getattr(response, "usage_metadata", {})
        actual_in = usage.get("input_tokens", estimated_input_tokens) if usage else estimated_input_tokens
        actual_out = usage.get("output_tokens", estimated_output_tokens) if usage else estimated_output_tokens
    else:
        output = f"[stub] Response to: {prompt[:40]}"
        actual_in, actual_out = estimated_input_tokens, estimated_output_tokens

    actual_cost = budget.record(model, actual_in, actual_out)
    status = budget.status()
    print(f"  [RECORDED] ${actual_cost:.6f} | utilisation={status.utilisation_pct:.1f}%")
    return output


def main() -> None:
    print(f"agent-energy-budget version: {agent_energy_budget.__version__}")

    if not _LANGCHAIN_AVAILABLE:
        print("LangChain not installed — using stub responses.")
        print("Install with: pip install langchain langchain-openai")

    # Step 1: Create budget
    budget = Budget(limit=0.05, agent_id="langchain-agent")
    print(f"Budget: ${budget.limit:.2f}/day")

    # Step 2: Make budgeted LangChain calls
    prompts: list[tuple[str, int, int]] = [
        ("What are the benefits of test-driven development?", 50, 150),
        ("Explain the CAP theorem in distributed systems.", 60, 200),
        ("Summarise the main principles of clean code.", 55, 180),
        ("What is the difference between REST and GraphQL?", 70, 250),
        ("Describe microservices architecture briefly.", 60, 200),
    ]

    print("\nBudgeted LangChain calls:")
    for prompt, est_in, est_out in prompts:
        result = budgeted_llm_call(budget, "gpt-4o-mini", prompt, est_in, est_out)
        if result:
            print(f"  Q: {prompt[:50]} -> {result[:60]}")

    # Step 3: Final status
    final = budget.status()
    print(f"\nFinal budget: ${final.spent_usd:.6f} / ${budget.limit:.2f} ({final.utilisation_pct:.1f}%)")


if __name__ == "__main__":
    main()
