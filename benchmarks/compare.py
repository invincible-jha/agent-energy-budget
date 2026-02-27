"""Comparison visualiser for agent-energy-budget benchmark results."""
from __future__ import annotations

import json
from pathlib import Path


COMPETITOR_NOTES = {
    "cost_prediction": (
        "LiteLLM #13724 (2024-07): 50% token-counting error on some Anthropic model outputs. "
        "Source: https://github.com/BerriAI/litellm/issues/13724. "
        "TokenCost library: silent failure (zero) for unknown models."
    ),
    "token_counting": (
        "LiteLLM #13724: message-overhead errors caused 50% cost estimation failures. "
        "tiktoken backend: expected <10% error vs chars/4. Heuristic: ~20-30%."
    ),
    "cached_tokens": (
        "Anthropic cache read = 10% of input rate. Cache write = 125% of input rate. "
        "CostEstimator currently bills all tokens at full input rate (documents the gap). "
        "LiteLLM #13724 listed cached token handling as a failure mode."
    ),
}


def _fmt_table(rows: list[tuple[str, str]], title: str) -> None:
    col1_width = max(len(r[0]) for r in rows) + 2
    print(f"\n{'=' * 65}")
    print(f"  {title}")
    print(f"{'=' * 65}")
    for key, value in rows:
        print(f"  {key:<{col1_width}} {value}")


def _load(path: Path) -> dict[str, object] | None:
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)  # type: ignore[return-value]


def display_prediction(data: dict[str, object]) -> None:
    overall = data.get("overall", {})
    rows: list[tuple[str, str]] = [
        ("Backend", str(data.get("backend"))),
        ("Mean error %", f"{overall.get('mean_error_pct')}%"),
        ("Max error %", f"{overall.get('max_error_pct')}%"),
        ("N samples", str(overall.get("n_samples"))),
    ]
    by_cat = data.get("by_category", {})
    for cat, stats in by_cat.items():
        rows.append((f"{cat} mean error %", f"{stats.get('mean_error_pct')}%"))
    _fmt_table(rows, "Cost Prediction Accuracy")
    print(f"\n  Competitor: {COMPETITOR_NOTES['cost_prediction']}")


def display_token_counting(data: dict[str, object]) -> None:
    overall = data.get("overall", {})
    rows: list[tuple[str, str]] = [
        ("Backend", str(data.get("backend"))),
        ("Mean % error", f"{overall.get('mean_pct_error')}%"),
        ("Mean abs error (tokens)", str(overall.get("mean_abs_error_tokens"))),
        ("N samples", str(overall.get("n_samples"))),
    ]
    _fmt_table(rows, "Token Counting Accuracy")
    print(f"\n  Competitor: {COMPETITOR_NOTES['token_counting']}")


def display_cached_tokens(data: dict[str, object]) -> None:
    det = data.get("detection_summary", {})
    rows: list[tuple[str, str]] = [
        ("Cache hit overprediction %", f"{det.get('cache_hit_mean_overprediction_pct')}%"),
        ("Cache write overprediction %", f"{det.get('cache_write_mean_overprediction_pct')}%"),
        ("Cache hits tested", str(det.get("n_cache_hits_tested"))),
        ("Cache writes tested", str(det.get("n_cache_writes_tested"))),
        ("Cache read factor used", str(data.get("cache_read_factor"))),
        ("Cache write factor used", str(data.get("cache_write_factor"))),
    ]
    _fmt_table(rows, "Cached Token Detection")
    print(f"\n  Competitor: {COMPETITOR_NOTES['cached_tokens']}")


def main() -> None:
    results_dir = Path(__file__).parent / "results"
    for fname, display_fn in [
        ("baseline.json", display_prediction),
        ("token_counting_baseline.json", display_token_counting),
        ("cached_token_baseline.json", display_cached_tokens),
    ]:
        data = _load(results_dir / fname)
        if data:
            display_fn(data)  # type: ignore[arg-type]
        else:
            print(f"No {fname} found. Run the corresponding benchmark first.")

    print("\n" + "=" * 65)
    print("  Run all benchmarks:")
    print("    python benchmarks/bench_prediction_accuracy.py")
    print("    python benchmarks/bench_token_counting.py")
    print("    python benchmarks/bench_cached_token_detection.py")
    print("=" * 65)


if __name__ == "__main__":
    main()
