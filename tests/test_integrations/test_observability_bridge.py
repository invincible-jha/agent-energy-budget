"""Tests for agent_energy_budget.integrations.observability_bridge.

All tests use hand-rolled fakes that satisfy TracerProtocol and
SpanContextProtocol via structural typing — no agent-observability
package is required.
"""
from __future__ import annotations

import logging
from typing import Any
from unittest.mock import MagicMock

import pytest

from agent_energy_budget.integrations.observability_bridge import (
    ObservabilityAlertBridge,
    SpanContextProtocol,
    TracerProtocol,
)


# ---------------------------------------------------------------------------
# Minimal fakes satisfying the Protocols
# ---------------------------------------------------------------------------


class _FakeSpan:
    """Fake span that records whether end() was called."""

    def __init__(self) -> None:
        self.ended: bool = False

    def end(self) -> None:
        self.ended = True


class _FakeTracer:
    """Fake tracer that records all calls without importing observability."""

    def __init__(self, span_to_return: _FakeSpan | None = None) -> None:
        self._span = span_to_return or _FakeSpan()
        self.events: list[tuple[str, dict[str, object]]] = []
        self.spans: list[tuple[str, dict[str, object]]] = []

    def start_span(
        self,
        name: str,
        attributes: dict[str, object],
    ) -> _FakeSpan:
        self.spans.append((name, attributes))
        return self._span

    def record_event(
        self,
        name: str,
        attributes: dict[str, object],
    ) -> None:
        self.events.append((name, attributes))


class _EventOnlyTracer:
    """Tracer stub that only implements record_event (no start_span)."""

    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, object]]] = []

    def record_event(
        self,
        name: str,
        attributes: dict[str, object],
    ) -> None:
        self.events.append((name, attributes))


class _BrokenTracer:
    """Tracer that raises on every call to test error handling."""

    def start_span(
        self,
        name: str,
        attributes: dict[str, object],
    ) -> _FakeSpan:
        raise RuntimeError("start_span broken")

    def record_event(
        self,
        name: str,
        attributes: dict[str, object],
    ) -> None:
        raise RuntimeError("record_event broken")


# ---------------------------------------------------------------------------
# Protocol compliance
# ---------------------------------------------------------------------------


class TestProtocolCompliance:
    def test_fake_tracer_satisfies_tracer_protocol(self) -> None:
        tracer = _FakeTracer()
        assert isinstance(tracer, TracerProtocol)

    def test_fake_span_satisfies_span_context_protocol(self) -> None:
        span = _FakeSpan()
        assert isinstance(span, SpanContextProtocol)

    def test_none_is_not_tracer_protocol(self) -> None:
        assert not isinstance(None, TracerProtocol)

    def test_object_without_methods_is_not_tracer_protocol(self) -> None:
        assert not isinstance(object(), TracerProtocol)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestObservabilityAlertBridgeConstruction:
    def test_default_construction_no_tracer(self) -> None:
        bridge = ObservabilityAlertBridge()
        assert bridge._tracer is None

    def test_construction_with_tracer(self) -> None:
        tracer = _FakeTracer()
        bridge = ObservabilityAlertBridge(tracer=tracer)
        assert bridge._tracer is tracer

    def test_repr_contains_class_name(self) -> None:
        bridge = ObservabilityAlertBridge()
        assert "ObservabilityAlertBridge" in repr(bridge)

    def test_repr_shows_none_when_no_tracer(self) -> None:
        bridge = ObservabilityAlertBridge()
        assert "None" in repr(bridge)

    def test_repr_shows_tracer_when_configured(self) -> None:
        tracer = _FakeTracer()
        bridge = ObservabilityAlertBridge(tracer=tracer)
        assert "None" not in repr(bridge)


# ---------------------------------------------------------------------------
# on_budget_warning — with tracer
# ---------------------------------------------------------------------------


class TestOnBudgetWarningWithTracer:
    def test_emits_warning_event(self) -> None:
        tracer = _FakeTracer()
        bridge = ObservabilityAlertBridge(tracer=tracer)
        bridge.on_budget_warning("agent-1", usage_percentage=85.0)
        assert len(tracer.events) == 1

    def test_event_name_is_warning(self) -> None:
        tracer = _FakeTracer()
        bridge = ObservabilityAlertBridge(tracer=tracer)
        bridge.on_budget_warning("agent-1", usage_percentage=85.0)
        event_name, _ = tracer.events[0]
        assert event_name == "energy.budget.warning"

    def test_event_contains_budget_id(self) -> None:
        tracer = _FakeTracer()
        bridge = ObservabilityAlertBridge(tracer=tracer)
        bridge.on_budget_warning("agent-42", usage_percentage=75.0)
        _, attrs = tracer.events[0]
        assert attrs["budget_id"] == "agent-42"

    def test_event_contains_usage_percentage(self) -> None:
        tracer = _FakeTracer()
        bridge = ObservabilityAlertBridge(tracer=tracer)
        bridge.on_budget_warning("agent-1", usage_percentage=91.5)
        _, attrs = tracer.events[0]
        assert attrs["usage_percentage"] == 91.5

    def test_event_has_warning_severity(self) -> None:
        tracer = _FakeTracer()
        bridge = ObservabilityAlertBridge(tracer=tracer)
        bridge.on_budget_warning("agent-1", usage_percentage=80.0)
        _, attrs = tracer.events[0]
        assert attrs["severity"] == "warning"

    def test_multiple_warnings_accumulate(self) -> None:
        tracer = _FakeTracer()
        bridge = ObservabilityAlertBridge(tracer=tracer)
        bridge.on_budget_warning("agent-1", usage_percentage=70.0)
        bridge.on_budget_warning("agent-1", usage_percentage=85.0)
        bridge.on_budget_warning("agent-1", usage_percentage=95.0)
        assert len(tracer.events) == 3


# ---------------------------------------------------------------------------
# on_budget_warning — no tracer (no-op)
# ---------------------------------------------------------------------------


class TestOnBudgetWarningNoTracer:
    def test_no_tracer_does_not_raise(self) -> None:
        bridge = ObservabilityAlertBridge()
        bridge.on_budget_warning("agent-1", usage_percentage=85.0)  # must not raise

    def test_no_tracer_logs_debug(self, caplog: pytest.LogCaptureFixture) -> None:
        bridge = ObservabilityAlertBridge()
        with caplog.at_level(logging.DEBUG, logger="agent_energy_budget.integrations.observability_bridge"):
            bridge.on_budget_warning("agent-1", usage_percentage=85.0)
        assert any("warning event dropped" in record.message for record in caplog.records)

    def test_broken_tracer_does_not_raise(self) -> None:
        bridge = ObservabilityAlertBridge(tracer=_BrokenTracer())  # type: ignore[arg-type]
        bridge.on_budget_warning("agent-1", usage_percentage=80.0)  # must not raise


# ---------------------------------------------------------------------------
# on_budget_exceeded — with tracer
# ---------------------------------------------------------------------------


class TestOnBudgetExceededWithTracer:
    def test_emits_exceeded_span(self) -> None:
        tracer = _FakeTracer()
        bridge = ObservabilityAlertBridge(tracer=tracer)
        bridge.on_budget_exceeded("agent-1", overage_usd=0.05)
        assert len(tracer.spans) == 1

    def test_span_name_is_exceeded(self) -> None:
        tracer = _FakeTracer()
        bridge = ObservabilityAlertBridge(tracer=tracer)
        bridge.on_budget_exceeded("agent-1", overage_usd=0.05)
        span_name, _ = tracer.spans[0]
        assert span_name == "energy.budget.exceeded"

    def test_span_contains_budget_id(self) -> None:
        tracer = _FakeTracer()
        bridge = ObservabilityAlertBridge(tracer=tracer)
        bridge.on_budget_exceeded("agent-99", overage_usd=0.12)
        _, attrs = tracer.spans[0]
        assert attrs["budget_id"] == "agent-99"

    def test_span_contains_overage_usd(self) -> None:
        tracer = _FakeTracer()
        bridge = ObservabilityAlertBridge(tracer=tracer)
        bridge.on_budget_exceeded("agent-1", overage_usd=0.042)
        _, attrs = tracer.spans[0]
        assert attrs["overage_usd"] == 0.042

    def test_span_has_error_severity(self) -> None:
        tracer = _FakeTracer()
        bridge = ObservabilityAlertBridge(tracer=tracer)
        bridge.on_budget_exceeded("agent-1", overage_usd=0.01)
        _, attrs = tracer.spans[0]
        assert attrs["severity"] == "error"

    def test_span_end_is_called(self) -> None:
        fake_span = _FakeSpan()
        tracer = _FakeTracer(span_to_return=fake_span)
        bridge = ObservabilityAlertBridge(tracer=tracer)
        bridge.on_budget_exceeded("agent-1", overage_usd=0.01)
        assert fake_span.ended is True

    def test_event_only_tracer_falls_back_to_record_event(self) -> None:
        """Tracer without start_span falls back to record_event gracefully."""
        tracer = _EventOnlyTracer()
        bridge = ObservabilityAlertBridge(tracer=tracer)  # type: ignore[arg-type]
        bridge.on_budget_exceeded("agent-1", overage_usd=0.05)
        # Should have fallen back to record_event
        assert len(tracer.events) == 1
        event_name, attrs = tracer.events[0]
        assert event_name == "energy.budget.exceeded"
        assert attrs["budget_id"] == "agent-1"


# ---------------------------------------------------------------------------
# on_budget_exceeded — no tracer (no-op)
# ---------------------------------------------------------------------------


class TestOnBudgetExceededNoTracer:
    def test_no_tracer_does_not_raise(self) -> None:
        bridge = ObservabilityAlertBridge()
        bridge.on_budget_exceeded("agent-1", overage_usd=0.05)  # must not raise

    def test_no_tracer_logs_debug(self, caplog: pytest.LogCaptureFixture) -> None:
        bridge = ObservabilityAlertBridge()
        with caplog.at_level(logging.DEBUG, logger="agent_energy_budget.integrations.observability_bridge"):
            bridge.on_budget_exceeded("agent-1", overage_usd=0.05)
        assert any("exceeded event dropped" in record.message for record in caplog.records)

    def test_broken_tracer_does_not_raise(self) -> None:
        bridge = ObservabilityAlertBridge(tracer=_BrokenTracer())  # type: ignore[arg-type]
        bridge.on_budget_exceeded("agent-1", overage_usd=0.05)  # must not raise


# ---------------------------------------------------------------------------
# on_cost_recorded — with tracer
# ---------------------------------------------------------------------------


class TestOnCostRecordedWithTracer:
    def test_emits_cost_event(self) -> None:
        tracer = _FakeTracer()
        bridge = ObservabilityAlertBridge(tracer=tracer)
        bridge.on_cost_recorded(model="gpt-4o", tokens=1500, cost_usd=0.003)
        assert len(tracer.events) == 1

    def test_event_name_is_cost_recorded(self) -> None:
        tracer = _FakeTracer()
        bridge = ObservabilityAlertBridge(tracer=tracer)
        bridge.on_cost_recorded(model="gpt-4o", tokens=1500, cost_usd=0.003)
        event_name, _ = tracer.events[0]
        assert event_name == "energy.cost.recorded"

    def test_event_contains_model(self) -> None:
        tracer = _FakeTracer()
        bridge = ObservabilityAlertBridge(tracer=tracer)
        bridge.on_cost_recorded(model="claude-haiku-4", tokens=800, cost_usd=0.001)
        _, attrs = tracer.events[0]
        assert attrs["model"] == "claude-haiku-4"

    def test_event_contains_tokens(self) -> None:
        tracer = _FakeTracer()
        bridge = ObservabilityAlertBridge(tracer=tracer)
        bridge.on_cost_recorded(model="gpt-4o-mini", tokens=2000, cost_usd=0.002)
        _, attrs = tracer.events[0]
        assert attrs["tokens"] == 2000

    def test_event_contains_cost_usd(self) -> None:
        tracer = _FakeTracer()
        bridge = ObservabilityAlertBridge(tracer=tracer)
        bridge.on_cost_recorded(model="gpt-4o", tokens=1000, cost_usd=0.0025)
        _, attrs = tracer.events[0]
        assert attrs["cost_usd"] == 0.0025

    def test_zero_cost_still_emitted(self) -> None:
        tracer = _FakeTracer()
        bridge = ObservabilityAlertBridge(tracer=tracer)
        bridge.on_cost_recorded(model="unknown-model", tokens=0, cost_usd=0.0)
        assert len(tracer.events) == 1

    def test_multiple_cost_events_accumulate(self) -> None:
        tracer = _FakeTracer()
        bridge = ObservabilityAlertBridge(tracer=tracer)
        bridge.on_cost_recorded(model="gpt-4o", tokens=500, cost_usd=0.001)
        bridge.on_cost_recorded(model="gpt-4o-mini", tokens=1000, cost_usd=0.0005)
        assert len(tracer.events) == 2


# ---------------------------------------------------------------------------
# on_cost_recorded — no tracer (no-op)
# ---------------------------------------------------------------------------


class TestOnCostRecordedNoTracer:
    def test_no_tracer_does_not_raise(self) -> None:
        bridge = ObservabilityAlertBridge()
        bridge.on_cost_recorded(model="gpt-4o", tokens=1000, cost_usd=0.002)

    def test_no_tracer_logs_debug(self, caplog: pytest.LogCaptureFixture) -> None:
        bridge = ObservabilityAlertBridge()
        with caplog.at_level(logging.DEBUG, logger="agent_energy_budget.integrations.observability_bridge"):
            bridge.on_cost_recorded(model="gpt-4o", tokens=1000, cost_usd=0.002)
        assert any("cost event dropped" in record.message for record in caplog.records)

    def test_broken_tracer_does_not_raise(self) -> None:
        bridge = ObservabilityAlertBridge(tracer=_BrokenTracer())  # type: ignore[arg-type]
        bridge.on_cost_recorded(model="gpt-4o", tokens=1000, cost_usd=0.002)


# ---------------------------------------------------------------------------
# Integration: mixed event sequence
# ---------------------------------------------------------------------------


class TestMixedEventSequence:
    def test_full_lifecycle_events_ordered(self) -> None:
        tracer = _FakeTracer()
        bridge = ObservabilityAlertBridge(tracer=tracer)

        bridge.on_cost_recorded(model="gpt-4o", tokens=500, cost_usd=0.001)
        bridge.on_budget_warning("agent-1", usage_percentage=80.0)
        bridge.on_cost_recorded(model="gpt-4o", tokens=800, cost_usd=0.0015)
        bridge.on_budget_exceeded("agent-1", overage_usd=0.002)

        # Two cost + one warning events; one exceeded span
        assert len(tracer.events) == 3
        assert len(tracer.spans) == 1

    def test_bridge_reassignment_uses_new_tracer(self) -> None:
        tracer_a = _FakeTracer()
        tracer_b = _FakeTracer()
        bridge = ObservabilityAlertBridge(tracer=tracer_a)
        bridge.on_budget_warning("agent-1", usage_percentage=70.0)
        # Swap tracer
        bridge._tracer = tracer_b
        bridge.on_budget_warning("agent-1", usage_percentage=90.0)
        assert len(tracer_a.events) == 1
        assert len(tracer_b.events) == 1
