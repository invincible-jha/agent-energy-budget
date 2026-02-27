"""Example: Using agent-energy-budget with LiteLLM.

Install with:
    pip install "aumos-agent-energy-budget[litellm]"

This example demonstrates wrapping LiteLLM calls with budget tracking
and enforcement. The adapter predicts cost before each call and blocks
calls that would exceed the session budget.
"""
from __future__ import annotations

# from agent_energy_budget.integrations.litellm_adapter import (
#     LiteLLMBudgetWrapper,
#     SessionSummary,
# )

# --- Configuration ---
# wrapper = LiteLLMBudgetWrapper(
#     session_budget_usd=1.00,  # $1 budget for this session
#     model="gpt-4o-mini",      # default model (overridable per call)
# )

# --- Make a budget-aware LLM call ---
# response = wrapper.completion_with_budget(
#     messages=[{"role": "user", "content": "Explain quantum computing in 2 sentences."}],
# )
# print(response.choices[0].message.content)

# --- Check remaining budget ---
# summary: SessionSummary = wrapper.session_summary()
# print(f"Spent: ${summary.total_cost_usd:.4f}")
# print(f"Remaining: ${summary.remaining_budget_usd:.4f}")
# print(f"Calls made: {summary.call_count}")

# --- Predict cost before calling ---
# predicted = wrapper.predict_before_call(
#     messages=[{"role": "user", "content": "Write a haiku."}],
# )
# print(f"Predicted cost: ${predicted.estimated_cost_usd:.4f}")
# print(f"Would exceed budget: {predicted.would_exceed_budget}")

# --- Reset session for a new task ---
# wrapper.reset_session()

print("Example: use_with_litellm.py")
print("Uncomment the code above and install litellm to run.")
