"""Bridge from energy-budget events to observability tracing.

This module allows agent-energy-budget to emit spans and events into an
observability backend without taking a hard dependency on agent-observability.
The bridge accepts any object satisfying :class:`TracerProtocol` via
Protocol-based structural typing.

No direct import of agent-observability is performed. Either package can be
installed independently; the bridge becomes a no-op when no tracer is supplied.

Install both packages to enable full integration::

    pip install agent-energy-budget aumos-agent-observability

Usage
-----
::

    from agent_energy_budget.integrations.observability_bridge import (
        ObservabilityAlertBridge,
    )
    from agent_observability import AgentTracer  # satisfies TracerProtocol

    tracer = AgentTracer()
    bridge = ObservabilityAlertBridge(tracer=tracer)

    # Forward budget events to observability spans
    bridge.on_budget_warning(budget_id="agent-42", usage_percentage=85.5)
    bridge.on_budget_exceeded(budget_id="agent-42", overage_usd=0.042)
    bridge.on_cost_recorded(model="gpt-4o", tokens=1500, cost_usd=0.003)
"""
from __future__ import annotations

import logging
from typing import Optional, Protocol, runtime_checkable

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Threshold constants
# ---------------------------------------------------------------------------

_WARNING_EVENT_NAME: str = "energy.budget.warning"
_EXCEEDED_EVENT_NAME: str = "energy.budget.exceeded"
_COST_RECORDED_EVENT_NAME: str = "energy.cost.recorded"

# ---------------------------------------------------------------------------
# Protocol definitions
# ---------------------------------------------------------------------------


@runtime_checkable
class SpanContextProtocol(Protocol):
    """Structural interface for an active span context.

    Any object exposing ``end()`` satisfies this protocol and can be used
    as a span returned by :class:`TracerProtocol`.
    """

    def end(self) -> None:
        """End/close this span."""
        ...  # pragma: no cover


@runtime_checkable
class TracerProtocol(Protocol):
    """Structural interface for any observability tracer.

    Any object exposing :meth:`start_span` and :meth:`record_event` can be
    passed as the ``tracer`` argument to :class:`ObservabilityAlertBridge`.
    This deliberately mirrors the method signatures in agent-observability's
    tracer implementations without importing them.
    """

    def start_span(
        self,
        name: str,
        attributes: dict[str, object],
    ) -> SpanContextProtocol:
        """Start a new span and return a handle to it.

        Parameters
        ----------
        name:
            Human-readable span name.
        attributes:
            Key/value pairs attached to the span as metadata.

        Returns
        -------
        SpanContextProtocol
            A span handle whose :meth:`~SpanContextProtocol.end` must be
            called when the span is complete.
        """
        ...  # pragma: no cover

    def record_event(
        self,
        name: str,
        attributes: dict[str, object],
    ) -> None:
        """Record a one-shot event (no duration) into the tracer.

        Parameters
        ----------
        name:
            Event name (dot-separated convention, e.g. ``"energy.budget.warning"``).
        attributes:
            Arbitrary key/value context for the event.
        """
        ...  # pragma: no cover


# ---------------------------------------------------------------------------
# Bridge implementation
# ---------------------------------------------------------------------------


class ObservabilityAlertBridge:
    """Emits observability spans and events for energy budget state changes.

    The bridge is deliberately decoupled from agent-observability via
    :class:`TracerProtocol`. Passing ``tracer=None`` is valid — all calls
    become no-ops. This preserves existing behaviour in environments where
    agent-observability is not installed.

    Parameters
    ----------
    tracer:
        Any object satisfying :class:`TracerProtocol`, or ``None`` to operate
        in no-op mode.

    Examples
    --------
    ::

        bridge = ObservabilityAlertBridge(tracer=my_tracer)
        bridge.on_budget_warning("agent-1", usage_percentage=80.0)
        bridge.on_budget_exceeded("agent-1", overage_usd=0.05)
        bridge.on_cost_recorded("gpt-4o", tokens=1200, cost_usd=0.0025)
    """

    def __init__(
        self,
        tracer: Optional[TracerProtocol] = None,
    ) -> None:
        self._tracer = tracer

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def on_budget_warning(
        self,
        budget_id: str,
        usage_percentage: float,
    ) -> None:
        """Emit a warning event when a budget approaches its threshold.

        Fires the event ``"energy.budget.warning"`` with the ``budget_id``
        and ``usage_percentage`` attached as attributes. The call is a
        no-op when no tracer is configured.

        Parameters
        ----------
        budget_id:
            Identifier for the budget that is nearing its limit (typically
            the agent_id from the BudgetConfig).
        usage_percentage:
            Current consumption as a percentage of the total budget (0–100+).
        """
        if self._tracer is None:
            logger.debug(
                "ObservabilityAlertBridge: no tracer configured; warning event dropped "
                "(budget_id=%s usage_percentage=%.2f)",
                budget_id,
                usage_percentage,
            )
            return

        try:
            self._tracer.record_event(
                _WARNING_EVENT_NAME,
                {
                    "budget_id": budget_id,
                    "usage_percentage": usage_percentage,
                    "severity": "warning",
                },
            )
            logger.debug(
                "ObservabilityAlertBridge: emitted warning event "
                "(budget_id=%s usage_percentage=%.2f)",
                budget_id,
                usage_percentage,
            )
        except Exception as exc:
            logger.warning(
                "ObservabilityAlertBridge: tracer.record_event failed for warning: %s",
                exc,
            )

    def on_budget_exceeded(
        self,
        budget_id: str,
        overage_usd: float,
    ) -> None:
        """Emit an error event when a budget has been exceeded.

        Opens a span (``"energy.budget.exceeded"``) and immediately closes
        it so that the event is visible as a zero-duration span in the
        observability backend, making it easy to correlate with adjacent
        LLM call spans. Falls back to ``record_event`` gracefully.

        Parameters
        ----------
        budget_id:
            Identifier for the budget that was exceeded.
        overage_usd:
            Amount by which the budget was exceeded, in USD (always >= 0).
        """
        if self._tracer is None:
            logger.debug(
                "ObservabilityAlertBridge: no tracer configured; exceeded event dropped "
                "(budget_id=%s overage_usd=%.8f)",
                budget_id,
                overage_usd,
            )
            return

        attributes: dict[str, object] = {
            "budget_id": budget_id,
            "overage_usd": overage_usd,
            "severity": "error",
        }

        try:
            span = self._tracer.start_span(_EXCEEDED_EVENT_NAME, attributes)
            try:
                span.end()
            except Exception as end_exc:
                logger.warning(
                    "ObservabilityAlertBridge: span.end() failed: %s", end_exc
                )
            logger.debug(
                "ObservabilityAlertBridge: emitted exceeded span "
                "(budget_id=%s overage_usd=%.8f)",
                budget_id,
                overage_usd,
            )
        except AttributeError:
            # Tracer may not implement start_span — fall back to record_event
            try:
                self._tracer.record_event(_EXCEEDED_EVENT_NAME, attributes)
            except Exception as exc:
                logger.warning(
                    "ObservabilityAlertBridge: tracer.record_event failed for exceeded: %s",
                    exc,
                )
        except Exception as exc:
            logger.warning(
                "ObservabilityAlertBridge: tracer.start_span failed for exceeded: %s",
                exc,
            )

    def on_cost_recorded(
        self,
        model: str,
        tokens: int,
        cost_usd: float,
    ) -> None:
        """Record a cost event as span attributes in the observability backend.

        Emits the event ``"energy.cost.recorded"`` with model, token count,
        and cost data attached. This gives observability dashboards a
        cost-per-call breakdown that complements the energy-budget package's
        own JSONL persistence.

        Parameters
        ----------
        model:
            LLM model identifier (e.g. ``"gpt-4o-mini"``).
        tokens:
            Total tokens consumed (input + output combined).
        cost_usd:
            Estimated cost in US dollars.
        """
        if self._tracer is None:
            logger.debug(
                "ObservabilityAlertBridge: no tracer configured; cost event dropped "
                "(model=%s tokens=%d cost_usd=%.8f)",
                model,
                tokens,
                cost_usd,
            )
            return

        try:
            self._tracer.record_event(
                _COST_RECORDED_EVENT_NAME,
                {
                    "model": model,
                    "tokens": tokens,
                    "cost_usd": cost_usd,
                },
            )
            logger.debug(
                "ObservabilityAlertBridge: emitted cost event "
                "(model=%s tokens=%d cost_usd=%.8f)",
                model,
                tokens,
                cost_usd,
            )
        except Exception as exc:
            logger.warning(
                "ObservabilityAlertBridge: tracer.record_event failed for cost: %s",
                exc,
            )

    def __repr__(self) -> str:
        tracer_repr = repr(self._tracer) if self._tracer is not None else "None"
        return f"ObservabilityAlertBridge(tracer={tracer_repr})"
