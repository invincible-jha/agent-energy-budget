# agent-energy-budget Benchmarks

Reproducible benchmark suite for cost prediction accuracy, token counting, and cached token detection.

## Quick Start

```bash
cd repos/agent-energy-budget
python benchmarks/bench_prediction_accuracy.py
python benchmarks/bench_token_counting.py
python benchmarks/bench_cached_token_detection.py
python benchmarks/compare.py
```

## Benchmarks

### bench_prediction_accuracy.py

**What it measures:** Predicted cost vs chars/4 ground-truth approximation (% error).

**Method:**
- Uses `CostEstimator.estimate_llm_call()` on prompts from 5 categories.
- Ground truth = `len(text) // 4` (OpenAI tokeniser FAQ approximation).
- Reports mean and max % error per category.

**Key metrics:**
- `mean_error_pct` — primary accuracy metric (target: <15%)
- `max_error_pct` — worst-case estimate deviation

**Competitor reference:**
LiteLLM issue #13724 (2024-07): ~50% token-counting error on Anthropic models.
Source: https://github.com/BerriAI/litellm/issues/13724

---

### bench_token_counting.py

**What it measures:** Token count accuracy (% error vs chars/4) for `TokenCounter`.

**Method:**
Tests both tiktoken and heuristic backends on the same prompt dataset.
Reports mean % error and mean absolute error (in tokens).

**Key metrics:**
- `mean_pct_error` — average deviation from ground truth
- `backend` — which counting method was active (tiktoken or heuristic)

---

### bench_cached_token_detection.py

**What it measures:** Overprediction gap when cached tokens are not modelled.

**Method:**
Simulates cache-hit scenarios using Anthropic's published cache pricing:
- Cache reads: 10% of standard input rate
- Cache writes: 125% of standard input rate

`CostEstimator` currently bills all tokens at the full input rate, so this
benchmark documents the resulting overprediction percentage.

**Key metrics:**
- `cache_hit_mean_overprediction_pct` — how much the naive estimator overpredicts for cache hits

---

## Interpreting Results

- Results saved to `results/` as JSON files.
- Use `compare.py` to display all results in a formatted table.
- All data is synthetic (no API calls, no downloads).
- Ground truth: `chars / 4` (universally documented approximation).

## Competitor Numbers (public only)

| Competitor | Issue | Error | Source |
|------------|-------|-------|--------|
| LiteLLM | #13724 | ~50% token count error | GitHub, Jul 2024 |
| TokenCost | — | Silent zero for unknown models | Library source |
