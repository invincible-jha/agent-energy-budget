"""Benchmark: Token counting accuracy vs known ground truth.

Tests TokenCounter against prompts with a deterministic synthetic ground
truth (chars/4) and computes:
- Mean absolute error (tokens)
- Mean % error relative to ground truth
- Backend detection (tiktoken vs heuristic)

Competitor context
------------------
LiteLLM issue #13724 (2024-07): reported ~50% error on some Anthropic
model completions due to incorrect message-overhead accounting.
Source: https://github.com/BerriAI/litellm/issues/13724

TokenCounter uses tiktoken (BPE) when available, falling back to a
word-based heuristic. This benchmark quantifies the accuracy of both.
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

from agent_energy_budget.pricing.token_counter import TokenCounter
from datasets.sample_prompts import generate_prompt_dataset


def _ground_truth_tokens(text: str) -> int:
    """Chars/4 approximation as ground truth reference."""
    return max(1, len(text) // 4)


def run_benchmark(seed: int = 42) -> dict[str, object]:
    """Measure token counting accuracy across prompt categories.

    Parameters
    ----------
    seed:
        Reproducibility seed.

    Returns
    -------
    dict with accuracy stats per category and overall.
    """
    counter = TokenCounter()
    prompts = generate_prompt_dataset(seed=seed)

    errors_by_category: dict[str, list[float]] = {}
    abs_errors_by_category: dict[str, list[float]] = {}
    latencies_ms: list[float] = []

    for sample in prompts:
        ground_truth = _ground_truth_tokens(sample.text)

        start = time.perf_counter()
        predicted = counter.count(sample.text)
        elapsed_ms = (time.perf_counter() - start) * 1000
        latencies_ms.append(elapsed_ms)

        abs_error = abs(predicted - ground_truth)
        pct_error = abs_error / ground_truth * 100 if ground_truth > 0 else 0.0

        cat = sample.category
        errors_by_category.setdefault(cat, []).append(pct_error)
        abs_errors_by_category.setdefault(cat, []).append(float(abs_error))

    category_stats: dict[str, dict[str, float]] = {}
    all_pct_errors: list[float] = []
    all_abs_errors: list[float] = []
    for cat in errors_by_category:
        pct_errors = errors_by_category[cat]
        abs_errors = abs_errors_by_category[cat]
        all_pct_errors.extend(pct_errors)
        all_abs_errors.extend(abs_errors)
        category_stats[cat] = {
            "mean_pct_error": round(statistics.mean(pct_errors), 2),
            "mean_abs_error_tokens": round(statistics.mean(abs_errors), 2),
            "max_pct_error": round(max(pct_errors), 2),
            "n_samples": len(pct_errors),
        }

    sorted_lats = sorted(latencies_ms)
    n = len(sorted_lats)

    return {
        "benchmark": "token_counting_accuracy",
        "backend": counter.backend,
        "seed": seed,
        "overall": {
            "mean_pct_error": round(statistics.mean(all_pct_errors), 2) if all_pct_errors else 0,
            "mean_abs_error_tokens": round(statistics.mean(all_abs_errors), 2) if all_abs_errors else 0,
            "max_pct_error": round(max(all_pct_errors), 2) if all_pct_errors else 0,
            "n_samples": len(all_pct_errors),
        },
        "by_category": category_stats,
        "counting_latency_ms": {
            "p50": round(sorted_lats[int(n * 0.50)], 4) if n else 0,
            "p95": round(sorted_lats[min(int(n * 0.95), n - 1)], 4) if n else 0,
            "mean": round(statistics.mean(latencies_ms), 4) if latencies_ms else 0,
        },
        "note": (
            "Ground truth = chars/4 approximation. "
            "LiteLLM #13724 (2024-07): ~50% error on Anthropic outputs. "
            "tiktoken backend expected to be within 5-10%; heuristic within 20-30%."
        ),
    }


if __name__ == "__main__":
    print("Running token counting accuracy benchmark...")
    result = run_benchmark()
    print(json.dumps(result, indent=2))
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)
    output_path = results_dir / "token_counting_baseline.json"
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)
    print(f"\nResults saved to {output_path}")
