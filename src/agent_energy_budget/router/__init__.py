"""Cost-aware model router for agent energy budget management.

This package provides the ``CostAwareRouter`` and supporting types for
routing LLM calls to the most cost-effective model given a budget constraint.

Public Surface
--------------
- ``CostAwareRouter``    — main router class
- ``ModelProfile``       — frozen dataclass describing a single model
- ``RouterBudgetConfig`` — router-level budget configuration
- ``BudgetConfig``       — alias for ``RouterBudgetConfig``
- ``RoutingDecision``    — result of a routing call
- ``RoutingStrategy``    — Protocol for custom strategy implementations
- ``CheapestFirstStrategy``
- ``QualityFirstStrategy``
- ``BalancedStrategy``
- ``BudgetAwareStrategy``
- ``NoAffordableModelError``
- ``DEFAULT_MODEL_PROFILES`` — built-in realistic model catalogue

Example
-------
::

    from agent_energy_budget.router import CostAwareRouter, RouterBudgetConfig

    budget = RouterBudgetConfig(total_budget_usd=10.0, min_quality_score=0.6)
    router = CostAwareRouter(budget=budget, strategy="balanced")
    decision = router.route("Write a haiku about Python.")
    print(decision.selected_model.name, f"${decision.estimated_cost:.6f}")
"""
from __future__ import annotations

from agent_energy_budget.router.cost_router import CostAwareRouter
from agent_energy_budget.router.models import (
    DEFAULT_MODEL_PROFILES,
    BudgetConfig,
    ModelProfile,
    RoutingDecision,
    RouterBudgetConfig,
    RoutingStrategy,
)
from agent_energy_budget.router.strategies import (
    BalancedStrategy,
    BudgetAwareStrategy,
    CheapestFirstStrategy,
    NoAffordableModelError,
    QualityFirstStrategy,
    RoutingStrategy as RoutingStrategyProtocol,
)

__all__ = [
    "CostAwareRouter",
    "ModelProfile",
    "RouterBudgetConfig",
    "BudgetConfig",
    "RoutingDecision",
    "RoutingStrategy",
    "RoutingStrategyProtocol",
    "CheapestFirstStrategy",
    "QualityFirstStrategy",
    "BalancedStrategy",
    "BudgetAwareStrategy",
    "NoAffordableModelError",
    "DEFAULT_MODEL_PROFILES",
]
