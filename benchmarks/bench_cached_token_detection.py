"""Benchmark: Cached vs non-cached cost estimation accuracy.

Simulates scenarios where a shared system prompt prefix is reused across
multiple calls (cache hit simulation). Measures whether cost estimates
reflect the lower effective cost of cached tokens.

Note: The current CostEstimator does not model cached token pricing as a
separate concept — it uses the full input rate for all tokens. This benchmark
documents that gap as a known baseline for future improvement.

Competitor context
------------------
LiteLLM #13724 (2024-07): cached token handling was one of the reported
failure modes. Anthropic's cache pricing is 10% of the standard input rate
for cache hits (read) and 125% for cache writes.
Source: https://github.com/BerriAI/litellm/issues/13724
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


# Anthropic cache pricing factors (published rates, Feb 2026)
_CACHE_READ_FACTOR = 0.10     # 10% of standard input rate for cache hits
_CACHE_WRITE_FACTOR = 1.25    # 125% of standard input rate for cache writes


def _cost_with_cache(
    model: str,
    prefix_tokens: int,
    suffix_tokens: int,
    output_tokens: int,
    is_cache_hit: bool,
) -> float:
    """Estimate cost accounting for cached prefix tokens.

    Parameters
    ----------
    model:
        Model identifier.
    prefix_tokens:
        Number of shared prefix tokens (the cached portion).
    suffix_tokens:
        Non-cached (unique per-call) suffix tokens.
    output_tokens:
        Output token count.
    is_cache_hit:
        If True, prefix_tokens are billed at cache-read rate (10%).
        If False, prefix_tokens are billed at cache-write rate (125%).

    Returns
    -------
    float
        Estimated cost in USD.
    """
    pricing = get_pricing(model)
    rate_per_million = pricing.input_per_million
    factor = _CACHE_READ_FACTOR if is_cache_hit else _CACHE_WRITE_FACTOR
    prefix_cost = (prefix_tokens / 1_000_000) * rate_per_million * factor
    suffix_cost = (suffix_tokens / 1_000_000) * rate_per_million
    output_cost = (output_tokens / 1_000_000) * pricing.output_per_million
    return round(prefix_cost + suffix_cost + output_cost, 8)


def run_benchmark(
    n_output_tokens: int = 256,
    seed: int = 42,
) -> dict[str, object]:
    """Measure the gap between naive cost estimates and cache-aware estimates.

    Parameters
    ----------
    n_output_tokens:
        Output tokens for all calls.
    seed:
        Reproducibility seed.

    Returns
    -------
    dict with overprediction % (naive vs cache-aware) and timing stats.
    """
    estimator = CostEstimator()
    prompts = [
        p for p in generate_prompt_dataset(seed=seed)
        if p.category == "cached_simulation"
    ]

    results: list[dict[str, float]] = []
    latencies_ms: list[float] = []

    shared_prefix = prompts[0].text if prompts else ""
    # Simulate: first call is a cache-write, subsequent calls are cache-hits
    for index, sample in enumerate(prompts):
        is_hit = index > 0
        char_count = len(sample.text)
        prefix_tokens = max(1, len(shared_prefix) // 4)
        suffix_tokens = max(1, (char_count - len(shared_prefix)) // 4)
        if suffix_tokens < 0:
            suffix_tokens = 1

        cache_aware_cost = _cost_with_cache(
            model=sample.model,
            prefix_tokens=prefix_tokens,
            suffix_tokens=suffix_tokens,
            output_tokens=n_output_tokens,
            is_cache_hit=is_hit,
        )

        start = time.perf_counter()
        naive_estimate = estimator.estimate_llm_call(
            sample.model, sample.text, n_output_tokens
        )
        elapsed_ms = (time.perf_counter() - start) * 1000
        latencies_ms.append(elapsed_ms)

        if cache_aware_cost > 0:
            overprediction_pct = (
                (naive_estimate.estimated_cost_usd - cache_aware_cost)
                / cache_aware_cost
                * 100
            )
        else:
            overprediction_pct = 0.0

        results.append(
            {
                "prompt_id": sample.prompt_id,
                "is_cache_hit": float(is_hit),
                "naive_cost_usd": naive_estimate.estimated_cost_usd,
                "cache_aware_cost_usd": cache_aware_cost,
                "overprediction_pct": round(overprediction_pct, 2),
            }
        )

    hit_overpredictions = [
        r["overprediction_pct"] for r in results if r["is_cache_hit"] > 0
    ]
    write_overpredictions = [
        r["overprediction_pct"] for r in results if r["is_cache_hit"] == 0
    ]

    sorted_lats = sorted(latencies_ms)
    n = len(sorted_lats)

    return {
        "benchmark": "cached_token_detection",
        "n_output_tokens": n_output_tokens,
        "seed": seed,
        "cache_read_factor": _CACHE_READ_FACTOR,
        "cache_write_factor": _CACHE_WRITE_FACTOR,
        "detection_summary": {
            "cache_hit_mean_overprediction_pct": (
                round(statistics.mean(hit_overpredictions), 2)
                if hit_overpredictions else 0
            ),
            "cache_write_mean_overprediction_pct": (
                round(statistics.mean(write_overpredictions), 2)
                if write_overpredictions else 0
            ),
            "n_cache_hits_tested": len(hit_overpredictions),
            "n_cache_writes_tested": len(write_overpredictions),
        },
        "per_prompt": results,
        "estimation_latency_ms": {
            "p50": round(sorted_lats[int(n * 0.50)], 4) if n else 0,
            "p95": round(sorted_lats[min(int(n * 0.95), n - 1)], 4) if n else 0,
            "mean": round(statistics.mean(latencies_ms), 4) if latencies_ms else 0,
        },
        "note": (
            "CostEstimator does not model cached token pricing. "
            "This benchmark documents the resulting overprediction gap. "
            "Anthropic cache read = 10% of input rate; cache write = 125%. "
            "Source: https://www.anthropic.com/pricing (Feb 2026). "
            "LiteLLM #13724: cached token handling was a reported failure mode."
        ),
    }


if __name__ == "__main__":
    print("Running cached token detection benchmark...")
    result = run_benchmark()
    print(json.dumps(result, indent=2))
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)
    output_path = results_dir / "cached_token_baseline.json"
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)
    print(f"\nResults saved to {output_path}")
