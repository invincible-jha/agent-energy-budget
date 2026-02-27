"""Real-time budget enforcement package.

Exports :class:`BudgetEnforcer` for synchronous pre-call budget checking
with atomic updates.
"""
from __future__ import annotations

from agent_energy_budget.enforcement.enforcer import (
    BudgetEnforcer,
    EnforcerConfig,
    EnforcerStatus,
    EnforcementResult,
)

__all__ = [
    "BudgetEnforcer",
    "EnforcerConfig",
    "EnforcerStatus",
    "EnforcementResult",
]
