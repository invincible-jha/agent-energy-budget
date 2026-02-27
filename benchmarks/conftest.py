"""Shared bootstrap for agent-energy-budget benchmarks."""
from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).parent.parent
_SRC = _REPO_ROOT / "src"
_BENCHMARKS = _REPO_ROOT / "benchmarks"

for _path in [str(_SRC), str(_BENCHMARKS)]:
    if _path not in sys.path:
        sys.path.insert(0, _path)

from agent_energy_budget.estimator.cost_estimator import CostEstimator
from agent_energy_budget.pricing.token_counter import TokenCounter

__all__ = ["CostEstimator", "TokenCounter"]
