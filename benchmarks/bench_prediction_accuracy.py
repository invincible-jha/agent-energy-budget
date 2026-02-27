"""Benchmark: Cost prediction error % for agent-energy-budget.

Compares predicted cost (from CostEstimator.estimate_llm_call) to a
synthetic "actual" cost calculated from exact character-derived token counts.

Competitor context
------------------
LiteLLM GitHub issue #13724 (2024-07): reported 50% token-counting error
on some Anthropic model outputs due to incorrect per-message overhead.
Source: https://github.com/BerriAI/litellm/issues/13724

TokenCost library uses static lookup tables; any model not in the table
returns zero cost (silent failure).

This benchmark measures the error % of CostEstimator predictions relative
to a character-count ground truth approximation. For pre-call estimation,
the goal is to be within 15% of actual cost.
"""
from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent / "datasets"))

from agent_energy_budget.estimator.cost_estimator import CostEstimator
from agent_energy_budget.pricing.tables import get_pricing
from datasets.sample_prompts import generate_prompt_dataset


def _char_based_token_count(text: str) -> int:
    """Simple chars/4 approximation used as a ground-truth proxy.

    This is the most widely cited rule-of-thumb (OpenAI tokeniser FAQ).
    It is not perfect, but provides a consistent reference for measuring
    how close the heuristic and tiktoken estimators are.
    """
    return max(1, len(text) // 4)


def _actual_cost(model: str, input_text: str, output_tokens: int) -> float:
    """Compute cost using the chars/4 ground truth approximation."""
    pricing = get_pricing(model)
    input_tokens = _char_based_token_count(input_text)
    return pricing.cost_for_tokens(input_tokens, output_tokens)


def run_benchmark(
    n_output_tokens: int = 512,
    seed: int = 42,
) -> dict[str, object]:
    """Measure cost prediction accuracy vs chars/4 ground truth.

    Parameters
    ----------
    n_output_tokens:
        Output token count used for all estimates (simulates known output size).
    seed:
        Reproducibility seed.

    Returns
    -------
    dict with error % stats per category and overall.
    """
    estimator = CostEstimator()
    prompts = generate_prompt_dataset(seed=seed)

    errors_by_category: dict[str, list[float]] = {}
    latencies_ms: list[float] = []

    for sample in prompts:
        category = sample.category
        actual = _actual_cost(sample.model, sample.text, n_output_tokens)
        if actual == 0.0:
            continue

        start = time.perf_counter()
        estimate = estimator.estimate_llm_call(sample.model, sample.text, n_output_tokens)
        elapsed_ms = (time.perf_counter() - start) * 1000
        latencies_ms.append(elapsed_ms)

        error_pct = abs(estimate.estimated_cost_usd - actual) / actual * 100
        errors_by_category.setdefault(category, []).append(error_pct)

    category_stats: dict[str, dict[str, float]] = {}
    all_errors: list[float] = []
    for category, errors in errors_by_category.items():
        all_errors.extend(errors)
        category_stats[category] = {
            "mean_error_pct": round(statistics.mean(errors), 2),
            "max_error_pct": round(max(errors), 2),
            "min_error_pct": round(min(errors), 2),
            "n_samples": len(errors),
        }

    sorted_lats = sorted(latencies_ms)
    n = len(sorted_lats)

    return {
        "benchmark": "cost_prediction_accuracy",
        "n_output_tokens_assumed": n_output_tokens,
        "backend": estimator._counter.backend,
        "seed": seed,
        "overall": {
            "mean_error_pct": round(statistics.mean(all_errors), 2) if all_errors else 0,
            "max_error_pct": round(max(all_errors), 2) if all_errors else 0,
            "n_samples": len(all_errors),
        },
        "by_category": category_stats,
        "estimation_latency_ms": {
            "p50": round(sorted_lats[int(n * 0.50)], 4) if n else 0,
            "p95": round(sorted_lats[min(int(n * 0.95), n - 1)], 4) if n else 0,
            "mean": round(statistics.mean(latencies_ms), 4) if latencies_ms else 0,
        },
        "note": (
            "Ground truth = chars/4 approximation (OpenAI tokeniser FAQ). "
            "LiteLLM #13724 (2024-07) showed 50% error on Anthropic models. "
            "Target: <15% mean error for pre-call estimation. "
            "Source: https://github.com/BerriAI/litellm/issues/13724"
        ),
    }


if __name__ == "__main__":
    print("Running cost prediction accuracy benchmark...")
    result = run_benchmark()
    print(json.dumps(result, indent=2))
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)
    output_path = results_dir / "baseline.json"
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)
    print(f"\nResults saved to {output_path}")
