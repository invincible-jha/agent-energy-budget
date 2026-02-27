# Migrating from LiteLLM to agent-energy-budget

agent-energy-budget is a cost control library focused on one problem LiteLLM does not address:
predicting and enforcing costs *before* an LLM call executes. LiteLLM excels at providing a
unified API interface across providers; agent-energy-budget adds pre-execution budget enforcement,
cost-aware routing, circuit breakers, and workload forecasting on top of any LLM client.

This guide covers full migration and coexistence patterns.

---

## Feature Comparison

| Capability | LiteLLM | agent-energy-budget |
|---|---|---|
| Universal LLM API (OpenAI-compatible) | Yes — primary feature | Not in scope — use any LLM client |
| Provider routing / fallbacks | Yes | Yes — cost-aware routing with `CostAwareRouter` (from agent-mesh-router) |
| Basic cost tracking (post-execution) | Yes — from API responses | Yes — `BudgetTracker.record()` |
| Pre-execution cost prediction | No | Yes — `CostPredictor.predict()` before a single token is sent |
| Budget enforcement with circuit breakers | No | Yes — `BudgetTracker.check()` blocks calls when budget is exceeded |
| Cost range estimates (low / high) | No | Yes — `PredictionResult.low_cost_usd` / `high_cost_usd` |
| Output token estimation by task type | No | Yes — `OutputEstimator` with task-type heuristics |
| Multi-agent sub-budget allocation | No | Yes — hierarchical sub-budgets per agent or team |
| Workload forecasting | No | Yes — project spend over a time window |
| Degradation strategies | No | Yes — auto-downgrade model when budget is tight |
| Audit trail (JSONL) | No | Yes — persistent JSONL spend records |
| Budget alerts | No | Yes — configurable thresholds with `BudgetAlertManager` |
| Python SDK | Yes | Yes |

---

## Installation

```bash
pip install agent-energy-budget

# With LiteLLM adapter for coexistence
pip install "agent-energy-budget[litellm]"
```

---

## Step 1 — Replace Post-Execution Cost Tracking

LiteLLM reports costs from the API response after tokens are consumed. agent-energy-budget records
the same data with `BudgetTracker.record()`.

**Before (LiteLLM cost tracking):**

```python
import litellm

response = litellm.completion(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Explain quantum entanglement"}],
)
# Cost is available only after the call completes
cost = response._hidden_params.get("response_cost", 0.0)
print(f"This call cost ${cost:.6f}")
```

**After (agent-energy-budget):**

```python
from agent_energy_budget import Budget

budget = Budget(limit=10.00, agent_id="research-agent")

# Make the call with your existing LLM client (LiteLLM, openai, anthropic, etc.)
response = litellm.completion(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Explain quantum entanglement"}],
)

# Record actual cost
input_tokens = response.usage.prompt_tokens
output_tokens = response.usage.completion_tokens
actual_cost = budget.record("gpt-4o", input_tokens, output_tokens)
print(f"Recorded ${actual_cost:.6f}. Status: {budget.status()}")
```

---

## Step 2 — Add Pre-Execution Cost Prediction

This is the central capability that agent-energy-budget provides and LiteLLM does not: knowing the
estimated cost before committing to a call. Agents can use predictions to route to cheaper models,
reduce context size, or halt gracefully.

```python
from agent_energy_budget.prediction.predictor import CostPredictor
from agent_energy_budget.prediction.output_estimator import TaskType

predictor = CostPredictor()

prompt = "Write a detailed technical report on transformer architectures, covering attention mechanisms, positional encoding, and training stability techniques."

# Predict before sending any tokens
prediction = predictor.predict(
    model="claude-sonnet-4",
    prompt=prompt,
    task_type=TaskType.LONG_FORM,
    budget_usd=0.05,
)

print(f"Estimated cost:  ${prediction.estimated_cost_usd:.6f}")
print(f"Low estimate:    ${prediction.low_cost_usd:.6f}")
print(f"High estimate:   ${prediction.high_cost_usd:.6f}")
print(f"Input tokens:    {prediction.input_tokens}")
print(f"Output tokens:   ~{prediction.estimated_output_tokens}")
print(f"Will exceed $0.05 budget: {prediction.will_exceed_budget}")
print(f"Prediction confidence: {prediction.confidence:.0%}")

if prediction.will_exceed_budget:
    # Route to a cheaper model or truncate the prompt
    print("Routing to cheaper model...")
    prediction_cheap = predictor.predict(
        model="claude-haiku-4",
        prompt=prompt,
        task_type=TaskType.LONG_FORM,
        budget_usd=0.05,
    )
    print(f"Haiku estimate: ${prediction_cheap.estimated_cost_usd:.6f}")
```

---

## Step 3 — Enforce Budget Before Every Call

`BudgetTracker.check()` is the core enforcement primitive. It checks the current spend against the
configured daily limit and returns a `BudgetRecommendation` that tells your agent exactly what to
do: proceed, downgrade, or halt.

```python
from agent_energy_budget import Budget

budget = Budget(limit=5.00, agent_id="pipeline-agent")

def call_llm_with_budget(prompt: str, model: str = "claude-sonnet-4") -> str:
    # Check BEFORE calling the API
    can_afford, recommendation = budget.check(
        model=model,
        input_tokens=len(prompt.split()) * 4,  # rough token estimate
        output_tokens=512,
    )

    if not can_afford:
        print(f"Budget check failed: {recommendation.message}")
        print(f"Action: {recommendation.action}")
        # recommendation.action is one of: "proceed", "downgrade", "halt"
        if recommendation.action == "halt":
            raise RuntimeError("Daily budget exhausted")
        # If "downgrade", use recommendation.model instead
        model = recommendation.model

    # Proceed with the recommended model
    response = your_llm_client.call(model=model, prompt=prompt)
    budget.record(model, response.input_tokens, response.output_tokens)
    return response.text
```

---

## Step 4 — Compare Models by Predicted Cost

When you have flexibility over which model to use, `CostPredictor.compare_models()` sorts
candidates by predicted cost so you can pick the most cost-effective option within your quality
requirements.

**Before (LiteLLM fallback — routes on error, not on cost):**

```python
import litellm

response = litellm.completion(
    model="claude-sonnet-4",
    fallbacks=["claude-haiku-4", "gpt-4o-mini"],
    messages=[{"role": "user", "content": prompt}],
)
```

**After (agent-energy-budget — routes on predicted cost before calling):**

```python
from agent_energy_budget.prediction.predictor import CostPredictor
from agent_energy_budget.prediction.output_estimator import TaskType

predictor = CostPredictor()

candidates = ["claude-sonnet-4", "claude-haiku-4", "gpt-4o", "gpt-4o-mini"]
ranked = predictor.compare_models(
    models=candidates,
    prompt=prompt,
    task_type=TaskType.CHAT,
    budget_usd=0.02,
)

for result in ranked:
    status = "OVER BUDGET" if result.will_exceed_budget else "within budget"
    print(f"{result.model}: ${result.estimated_cost_usd:.6f} ({status})")

# Pick the first model within budget
best = next((r for r in ranked if not r.will_exceed_budget), None)
if best:
    response = your_llm_client.call(model=best.model, prompt=prompt)
```

---

## Step 5 — Forecast Batch Costs Before Running a Pipeline

For batch processing jobs, predict the total spend before committing the entire workload.

```python
from agent_energy_budget.prediction.predictor import CostPredictor, BatchPredictionResult
from agent_energy_budget.prediction.output_estimator import TaskType

predictor = CostPredictor()

# A batch of planned calls
planned_calls = [
    {"model": "claude-sonnet-4", "prompt": doc, "task_type": TaskType.SUMMARIZATION}
    for doc in document_list
]

forecast: BatchPredictionResult = predictor.predict_batch(
    calls=planned_calls,
    budget_usd=50.00,
)

print(f"Total estimated cost: ${forecast.total_estimated_cost_usd:.4f}")
print(f"Low estimate:         ${forecast.total_low_cost_usd:.4f}")
print(f"High estimate:        ${forecast.total_high_cost_usd:.4f}")
print(f"Total input tokens:   {forecast.total_input_tokens:,}")
print(f"Any will exceed $50:  {forecast.any_will_exceed_budget}")

if forecast.any_will_exceed_budget:
    print("Batch exceeds budget — switch models or reduce batch size before proceeding.")
```

---

## Step 6 — Multi-Agent Sub-Budget Allocation

For teams of agents, allocate sub-budgets so individual agents cannot consume the full pool.

```python
from agent_energy_budget.budget.config import BudgetConfig
from agent_energy_budget.budget.tracker import BudgetTracker
from agent_energy_budget.budget.sub_budget import SubBudgetAllocator

import tempfile

# Root budget for the entire pipeline
root_config = BudgetConfig(agent_id="pipeline-root", daily_limit=20.00)
root_tracker = BudgetTracker(config=root_config, storage_dir=tempfile.mkdtemp())

# Allocate sub-budgets per agent
allocator = SubBudgetAllocator(root_tracker=root_tracker)
retrieval_budget = allocator.allocate(agent_id="retrieval-agent", fraction=0.3)  # $6/day
reasoning_budget = allocator.allocate(agent_id="reasoning-agent", fraction=0.5)  # $10/day
response_budget  = allocator.allocate(agent_id="response-agent",  fraction=0.2)  # $4/day
```

---

## Coexistence: Adding Pre-Execution Prediction on Top of LiteLLM

You do not need to replace LiteLLM to gain budget enforcement. The `[litellm]` adapter wraps
LiteLLM's `completion()` call with a pre-execution prediction and post-execution recording.

```bash
pip install "agent-energy-budget[litellm]"
```

```python
from agent_energy_budget.adapters.litellm import BudgetedLiteLLM
from agent_energy_budget import Budget

budget = Budget(limit=5.00, agent_id="my-agent")

# Drop-in replacement for litellm.completion with budget enforcement
llm = BudgetedLiteLLM(budget=budget)

response = llm.completion(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Summarize this article: ..."}],
)
# If the predicted cost would exceed the budget, completion() raises BudgetExceededError
# instead of sending the request. Actual cost is recorded automatically on success.
```

This pattern requires changing only the import. All existing LiteLLM parameters, fallbacks, and
retry logic continue to work unchanged.

---

## What You Gain by Switching

1. **Pre-execution cost prediction** — `CostPredictor.predict()` gives you a cost estimate with
   low/high bounds before a single token is sent to any API. Agents can make routing decisions
   without paying to find out they exceeded the budget.
2. **Budget enforcement with circuit breakers** — `BudgetTracker.check()` blocks calls that would
   exceed the daily limit and returns a structured recommendation (proceed, downgrade, halt).
3. **Degradation strategies** — when the budget is tight, the tracker can automatically suggest
   a cheaper model rather than failing the entire request.
4. **Workload forecasting** — `CostPredictor.predict_batch()` lets you forecast the cost of an
   entire batch job before committing, catching budget overruns before they happen.
5. **Audit trail** — every `record()` call appends to a JSONL file for compliance and cost
   attribution reporting.
6. **Multi-agent sub-budgets** — `SubBudgetAllocator` prevents individual agents in a pipeline
   from consuming the entire cost pool.

## What You Keep

- LiteLLM's universal API interface, fallback routing, and provider configuration are unchanged.
  agent-energy-budget wraps LiteLLM without modifying it.
- All existing LiteLLM model aliases and router configuration remain valid — the adapter passes
  through all parameters.
- Cost records stored in LiteLLM's internal tracking can be reconciled against agent-energy-budget
  JSONL records using the model and timestamp fields.
