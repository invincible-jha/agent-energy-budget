"""prediction — pre-execution LLM cost prediction for agent-energy-budget.

This subpackage provides cost estimation before any tokens are sent to an
API endpoint. It combines token counting, output token heuristics, and
a pricing table to produce cost estimates with confidence scores.

Public surface
--------------
- :class:`CostPredictor` — the main prediction engine
- :class:`PricingTable` — model pricing registry with runtime updates
- :class:`ModelPricing` — per-model pricing dataclass (per 1K tokens)
- :class:`TokenCounter` — pre-call token estimation
- :class:`OutputEstimator` — output token heuristics by task type
- :class:`OutputEstimate` — result of output estimation
- :class:`PredictionResult` — result of a single cost prediction
- :class:`BatchPredictionResult` — aggregate result for multiple calls
- :class:`TaskType` — task type enum for output estimation

Quick start
-----------
::

    from agent_energy_budget.prediction import CostPredictor

    predictor = CostPredictor()
    result = predictor.predict(
        model="claude-sonnet-4",
        prompt="Explain quantum computing in simple terms.",
        task_type="qa",
        budget_usd=0.01,
    )
    print(f"Estimated cost: ${result.estimated_cost_usd:.6f}")
    print(f"Will exceed budget: {result.will_exceed_budget}")
"""
from __future__ import annotations

from agent_energy_budget.prediction.output_estimator import (
    OutputEstimate,
    OutputEstimator,
    TaskType,
)
from agent_energy_budget.prediction.predictor import (
    BatchPredictionResult,
    CostPredictor,
    PredictionResult,
)
from agent_energy_budget.prediction.pricing_table import ModelPricing, PricingTable
from agent_energy_budget.prediction.token_counter import TokenCounter

__all__ = [
    "BatchPredictionResult",
    "CostPredictor",
    "ModelPricing",
    "OutputEstimate",
    "OutputEstimator",
    "PredictionResult",
    "PricingTable",
    "TaskType",
    "TokenCounter",
]
