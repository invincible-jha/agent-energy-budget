# agent-energy-budget

Agent cost control, energy budget management, and token tracking

[![CI](https://github.com/aumos-ai/agent-energy-budget/actions/workflows/ci.yaml/badge.svg)](https://github.com/aumos-ai/agent-energy-budget/actions/workflows/ci.yaml)
[![PyPI version](https://img.shields.io/pypi/v/agent-energy-budget.svg)](https://pypi.org/project/agent-energy-budget/)
[![Python versions](https://img.shields.io/pypi/pyversions/agent-energy-budget.svg)](https://pypi.org/project/agent-energy-budget/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)

Part of the [AumOS](https://github.com/aumos-ai) open-source agent infrastructure portfolio.

---

## Features

- `BudgetTracker` performs pre-call affordability checks and records actual spend to a per-agent JSONL log, supporting daily, weekly, and monthly limits simultaneously
- Four degradation strategies when budgets are tight: `TOKEN_REDUCTION` (cap output tokens to fit remaining budget), `MODEL_DOWNGRADE` (switch to the cheapest affordable model), `CACHED_FALLBACK` (signal to use cached responses), and `BLOCK_WITH_ERROR` (raise `BudgetExceededError`)
- Model pricing tables for OpenAI, Anthropic, Google, and custom models with a `CostEstimator` that projects spend before the call is made
- Hierarchical sub-budget allocation — a parent `BudgetTracker` can allocate fractional child budgets for sub-agents in a multi-agent team
- Threshold-based alert system fires warning, critical, and exhausted callbacks at configurable utilization percentages
- Middleware wrappers for LangChain, CrewAI, AutoGen, Anthropic SDK, and OpenAI SDK that intercept calls and apply budget checks transparently
- Spend reporting with per-model breakdowns, top-N cost attribution, and ASCII/matplotlib visualizations

## Current Limitations

> **Transparency note**: We list known limitations to help you evaluate fit.

- **Pricing**: Static pricing tables. No real-time API price feeds.
- **Cache**: No semantic caching yet — exact match only.
- **Providers**: Manual cost model configuration per provider.

## Quick Start

Install from PyPI:

```bash
pip install agent-energy-budget
```

Verify the installation:

```bash
agent-energy-budget version
```

Basic usage:

```python
import agent_energy_budget

# See examples/01_quickstart.py for a working example
```

## Documentation

- [Architecture](docs/architecture.md)
- [Contributing](CONTRIBUTING.md)
- [Changelog](CHANGELOG.md)
- [Examples](examples/README.md)

## Enterprise Upgrade

For production deployments requiring SLA-backed support and advanced
integrations, contact the maintainers or see the commercial extensions documentation.

## Contributing

Contributions are welcome. Please read [CONTRIBUTING.md](CONTRIBUTING.md)
before opening a pull request.

## License

Apache 2.0 — see [LICENSE](LICENSE) for full terms.

---

Part of [AumOS](https://github.com/aumos-ai) — open-source agent infrastructure.
