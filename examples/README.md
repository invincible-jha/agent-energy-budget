# Examples

| # | Example | Description |
|---|---------|-------------|
| 01 | [Quickstart](01_quickstart.py) | Minimal working example with the Budget convenience class |
| 02 | [Multi-Model Tracking](02_multi_model_tracking.py) | Track costs across multiple LLM models |
| 03 | [Budget Alerts](03_budget_alerts.py) | Soft warnings and hard blocks at configurable thresholds |
| 04 | [Token Optimisation](04_token_optimisation.py) | Auto-select the cheapest model within budget |
| 05 | [Multi-Agent Budget](05_multi_agent_budget.py) | Shared budget pool with per-agent allocations |
| 06 | [LangChain Budget](06_langchain_budget.py) | Gate LangChain calls with cost enforcement |
| 07 | [CrewAI Budget](07_crewai_budget.py) | Track and enforce costs during CrewAI execution |

## Running the examples

```bash
pip install agent-energy-budget
python examples/01_quickstart.py
```

For framework integrations:

```bash
pip install langchain langchain-openai   # for example 06
pip install crewai                       # for example 07
```
