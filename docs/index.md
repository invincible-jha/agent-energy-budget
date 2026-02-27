# agent-energy-budget

Agent Cost Control & Energy Budget SDK — budget management, token prediction, cost-aware routing.

[![CI](https://github.com/invincible-jha/agent-energy-budget/actions/workflows/ci.yaml/badge.svg)](https://github.com/invincible-jha/agent-energy-budget/actions/workflows/ci.yaml)
[![PyPI version](https://img.shields.io/pypi/v/agent-energy-budget.svg)](https://pypi.org/project/agent-energy-budget/)
[![Python versions](https://img.shields.io/pypi/pyversions/agent-energy-budget.svg)](https://pypi.org/project/agent-energy-budget/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)

---

## Installation

```bash
pip install agent-energy-budget
```

Verify the installation:

```bash
agent-energy-budget version
```

---

## Quick Start

```python
from agent_energy_budget import BudgetTracker, CostEstimator, DegradationStrategy

# Create a budget tracker with daily and monthly limits
tracker = BudgetTracker(
    agent_id="agent-alpha",
    daily_limit=5.00,
    monthly_limit=100.00,
    log_path="spend.jsonl",
)

# Register alert callbacks
tracker.on_warning(lambda used, limit: print(f"Warning: ${used:.2f} / ${limit:.2f}"))
tracker.on_critical(lambda used, limit: print(f"Critical: ${used:.2f} / ${limit:.2f}"))

# Estimate cost before making a call
estimator = CostEstimator()
estimated = estimator.estimate(
    model="gpt-4o",
    input_tokens=1024,
    output_tokens=256,
)
print(f"Estimated cost: ${estimated:.4f}")

# Check affordability and record actual spend
if tracker.can_afford(estimated):
    # Make your LLM call here
    actual_cost = make_llm_call()
    tracker.record_spend(actual_cost, model="gpt-4o")
else:
    # Apply a degradation strategy
    strategy = tracker.get_degradation_strategy()
    if strategy == DegradationStrategy.MODEL_DOWNGRADE:
        actual_cost = make_llm_call(model="gpt-4o-mini")
        tracker.record_spend(actual_cost, model="gpt-4o-mini")

# Generate a spend report
report = tracker.report()
print(f"Today: ${report.daily_spend:.4f} / ${report.daily_limit:.2f}")
print(f"Top model: {report.top_model} (${report.top_model_spend:.4f})")
```

---

## Key Features

- **`BudgetTracker`** — performs pre-call affordability checks and records actual spend to a per-agent JSONL log, supporting daily, weekly, and monthly limits simultaneously
- **Four degradation strategies** — `TOKEN_REDUCTION` (cap output tokens), `MODEL_DOWNGRADE` (switch to cheapest affordable model), `CACHED_FALLBACK` (use cached responses), and `BLOCK_WITH_ERROR` (raise `BudgetExceededError`)
- **Model pricing tables** — for OpenAI, Anthropic, Google, and custom models with a `CostEstimator` that projects spend before the call is made
- **Hierarchical sub-budget allocation** — a parent `BudgetTracker` can allocate fractional child budgets for sub-agents in a multi-agent team
- **Threshold-based alert system** — fires warning, critical, and exhausted callbacks at configurable utilization percentages
- **Middleware wrappers** — for LangChain, CrewAI, AutoGen, Anthropic SDK, and OpenAI SDK that intercept calls and apply budget checks transparently
- **Spend reporting** — per-model breakdowns, top-N cost attribution, and ASCII/matplotlib visualizations

---

## Links

- [GitHub Repository](https://github.com/invincible-jha/agent-energy-budget)
- [PyPI Package](https://pypi.org/project/agent-energy-budget/)
- [Architecture](architecture.md)
- [Migration from LiteLLM](migrate-from-litellm.md)
- [Changelog](https://github.com/invincible-jha/agent-energy-budget/blob/main/CHANGELOG.md)
- [Contributing](https://github.com/invincible-jha/agent-energy-budget/blob/main/CONTRIBUTING.md)

---

> Part of the [AumOS](https://github.com/aumos-ai) open-source agent infrastructure portfolio.
