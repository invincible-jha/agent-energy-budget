"""Hierarchical budget allocation package.

Exports :class:`HierarchicalBudget` for org → team → agent budget trees.
"""
from __future__ import annotations

from agent_energy_budget.hierarchy.budget_hierarchy import (
    HierarchicalBudget,
    BudgetNode,
    HierarchyConfig,
    NodeStatus,
)

__all__ = [
    "HierarchicalBudget",
    "BudgetNode",
    "HierarchyConfig",
    "NodeStatus",
]
