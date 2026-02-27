"""Adaptive model routing enhancements (E10.2).

Provides an AdaptiveLearner that tracks success rates per model per task type
and recommends models based on historical success/cost ratios, plus a
TaskClassifier for classifying task complexity from prompt characteristics.

Example
-------
::

    from agent_energy_budget.routing import AdaptiveLearner, TaskClassifier

    learner = AdaptiveLearner()
    learner.record("gpt-4o-mini", "simple", success=True, cost_usd=0.0002)
    learner.record("gpt-4o-mini", "simple", success=True, cost_usd=0.0002)
    learner.record("claude-sonnet-4-6", "simple", success=False, cost_usd=0.015)

    recommendations = learner.recommend("simple")
    # recommendations[0] is the model with best success_rate/cost ratio

    classifier = TaskClassifier()
    task_type = classifier.classify("What is 2+2?")  # "simple"
"""
from __future__ import annotations

from agent_energy_budget.routing.adaptive_learner import (
    AdaptiveLearner,
    ModelStats,
    RecommendedModel,
)
from agent_energy_budget.routing.task_classifier import (
    TaskClassifier,
    TaskType,
    ClassificationResult,
)

__all__ = [
    "AdaptiveLearner",
    "ClassificationResult",
    "ModelStats",
    "RecommendedModel",
    "TaskClassifier",
    "TaskType",
]
