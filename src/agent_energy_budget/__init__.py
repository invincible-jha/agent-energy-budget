"""agent-energy-budget — Agent cost control, energy budget management, and token tracking.

Public API
----------
The stable public surface is everything exported from this module.
Anything inside submodules not re-exported here is considered private
and may change without notice.

Example
-------
>>> import agent_energy_budget
>>> agent_energy_budget.__version__
'0.1.0'
"""
from __future__ import annotations

__version__: str = "0.1.0"

from agent_energy_budget.convenience import Budget
from agent_energy_budget.integrations.observability_bridge import (
    ObservabilityAlertBridge,
    SpanContextProtocol,
    TracerProtocol,
)

__all__ = [
    "__version__",
    "Budget",
    "ObservabilityAlertBridge",
    "SpanContextProtocol",
    "TracerProtocol",
]
