"""Framework integration middleware for budget enforcement."""
from __future__ import annotations

from agent_energy_budget.middleware.generic import BudgetGuardError, budget_guard

__all__ = [
    "budget_guard",
    "BudgetGuardError",
]
