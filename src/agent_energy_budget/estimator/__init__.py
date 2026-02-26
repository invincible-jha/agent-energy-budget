"""Standalone cost estimator module.

Provides pre-execution cost estimation for LLM calls without
requiring a BudgetTracker instance.
"""
from __future__ import annotations

from agent_energy_budget.estimator.cost_estimator import CostEstimate, CostEstimator

__all__ = ["CostEstimator", "CostEstimate"]
