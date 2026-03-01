"""Integration adapters for agent-energy-budget.

Optional third-party integrations. Install extras to enable:

    pip install agent-energy-budget[litellm]

The observability bridge requires no extra dependencies — it uses
Protocol-based structural typing and works with any compatible tracer:

    from agent_energy_budget.integrations.observability_bridge import (
        ObservabilityAlertBridge,
        TracerProtocol,
        SpanContextProtocol,
    )
"""
from agent_energy_budget.integrations.observability_bridge import (
    ObservabilityAlertBridge,
    SpanContextProtocol,
    TracerProtocol,
)

__all__ = [
    "ObservabilityAlertBridge",
    "SpanContextProtocol",
    "TracerProtocol",
]
