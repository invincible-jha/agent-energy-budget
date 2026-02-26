"""Degradation strategies for graceful budget enforcement."""
from __future__ import annotations

from agent_energy_budget.degradation.base import DegradationResult, DegradationStrategyBase
from agent_energy_budget.degradation.block_with_error import BlockStrategy
from agent_energy_budget.degradation.cached_fallback import CachedFallbackStrategy
from agent_energy_budget.degradation.model_downgrade import ModelDowngradeStrategy
from agent_energy_budget.degradation.registry import StrategyRegistry
from agent_energy_budget.degradation.token_reduction import TokenReductionStrategy

__all__ = [
    "DegradationStrategyBase",
    "DegradationResult",
    "ModelDowngradeStrategy",
    "TokenReductionStrategy",
    "BlockStrategy",
    "CachedFallbackStrategy",
    "StrategyRegistry",
]
