"""Unit tests for agent_energy_budget.budget.alerts."""
from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch

import pytest

from agent_energy_budget.budget.alerts import (
    AlertEvent,
    AlertLevel,
    BudgetAlertManager,
)


# ---------------------------------------------------------------------------
# AlertEvent
# ---------------------------------------------------------------------------


class TestAlertEvent:
    def test_to_dict_contains_all_fields(self) -> None:
        manager = BudgetAlertManager()
        events = manager.check_and_fire(
            agent_id="agent1", period="daily", spent_usd=5.0, limit_usd=10.0
        )
        assert len(events) == 1
        d = events[0].to_dict()
        assert d["agent_id"] == "agent1"
        assert d["level"] == AlertLevel.WARNING.value
        assert "utilisation_pct" in d
        assert "spent_usd" in d
        assert "limit_usd" in d
        assert "period" in d
        assert "message" in d
        assert "fired_at" in d

    def test_to_dict_fired_at_is_iso_string(self) -> None:
        manager = BudgetAlertManager()
        events = manager.check_and_fire(
            agent_id="a", period="daily", spent_usd=6.0, limit_usd=10.0
        )
        d = events[0].to_dict()
        # Should parse back to datetime without error
        from datetime import datetime
        datetime.fromisoformat(d["fired_at"])  # type: ignore[arg-type]

    def test_alert_event_is_frozen(self) -> None:
        manager = BudgetAlertManager()
        events = manager.check_and_fire(
            agent_id="a", period="daily", spent_usd=6.0, limit_usd=10.0
        )
        event = events[0]
        with pytest.raises((AttributeError, TypeError)):
            event.agent_id = "other"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# AlertLevel enum
# ---------------------------------------------------------------------------


class TestAlertLevel:
    def test_values(self) -> None:
        assert AlertLevel.WARNING.value == "warning"
        assert AlertLevel.CRITICAL.value == "critical"
        assert AlertLevel.EXHAUSTED.value == "exhausted"


# ---------------------------------------------------------------------------
# BudgetAlertManager — basic instantiation
# ---------------------------------------------------------------------------


class TestBudgetAlertManagerInit:
    def test_default_thresholds(self) -> None:
        manager = BudgetAlertManager()
        # Internal thresholds should be readable via check behaviour
        # WARNING fires at 50%, CRITICAL at 80%, EXHAUSTED at 100%
        events = manager.check_and_fire("a", "daily", 5.0, 10.0)  # 50%
        levels = {e.level for e in events}
        assert AlertLevel.WARNING in levels

    def test_custom_thresholds(self) -> None:
        manager = BudgetAlertManager(
            warning_threshold=25.0,
            critical_threshold=50.0,
            exhausted_threshold=75.0,
        )
        events = manager.check_and_fire("a", "daily", 3.0, 10.0)  # 30%
        levels = {e.level for e in events}
        assert AlertLevel.WARNING in levels
        assert AlertLevel.CRITICAL not in levels


# ---------------------------------------------------------------------------
# BudgetAlertManager.check_and_fire
# ---------------------------------------------------------------------------


class TestCheckAndFire:
    def test_no_alerts_below_threshold(self) -> None:
        manager = BudgetAlertManager(warning_threshold=50.0)
        events = manager.check_and_fire("a", "daily", 4.0, 10.0)  # 40%
        assert events == []

    def test_warning_fires_at_exact_threshold(self) -> None:
        manager = BudgetAlertManager(warning_threshold=50.0)
        events = manager.check_and_fire("a", "daily", 5.0, 10.0)  # exactly 50%
        levels = [e.level for e in events]
        assert AlertLevel.WARNING in levels

    def test_critical_fires_at_threshold(self) -> None:
        manager = BudgetAlertManager()
        events = manager.check_and_fire("a", "daily", 8.0, 10.0)  # 80%
        levels = {e.level for e in events}
        assert AlertLevel.WARNING in levels
        assert AlertLevel.CRITICAL in levels

    def test_exhausted_fires_at_100_pct(self) -> None:
        manager = BudgetAlertManager()
        events = manager.check_and_fire("a", "daily", 10.0, 10.0)  # 100%
        levels = {e.level for e in events}
        assert AlertLevel.EXHAUSTED in levels

    def test_each_level_fires_at_most_once_per_period(self) -> None:
        manager = BudgetAlertManager()
        # First call fires warning
        events1 = manager.check_and_fire("a", "daily", 5.0, 10.0)
        assert any(e.level == AlertLevel.WARNING for e in events1)
        # Second call at same utilisation should not re-fire
        events2 = manager.check_and_fire("a", "daily", 5.0, 10.0)
        assert not any(e.level == AlertLevel.WARNING for e in events2)

    def test_zero_limit_returns_empty_list(self) -> None:
        manager = BudgetAlertManager()
        events = manager.check_and_fire("a", "daily", 1.0, 0.0)
        assert events == []

    def test_negative_limit_returns_empty_list(self) -> None:
        manager = BudgetAlertManager()
        events = manager.check_and_fire("a", "daily", 1.0, -5.0)
        assert events == []

    def test_multiple_agents_tracked_independently(self) -> None:
        manager = BudgetAlertManager()
        events_a = manager.check_and_fire("agent_a", "daily", 5.0, 10.0)
        events_b = manager.check_and_fire("agent_b", "daily", 5.0, 10.0)
        assert any(e.level == AlertLevel.WARNING for e in events_a)
        assert any(e.level == AlertLevel.WARNING for e in events_b)

    def test_multiple_periods_tracked_independently(self) -> None:
        manager = BudgetAlertManager()
        events_daily = manager.check_and_fire("a", "daily", 5.0, 10.0)
        events_weekly = manager.check_and_fire("a", "weekly", 5.0, 10.0)
        assert any(e.level == AlertLevel.WARNING for e in events_daily)
        assert any(e.level == AlertLevel.WARNING for e in events_weekly)

    def test_event_has_correct_utilisation_pct(self) -> None:
        manager = BudgetAlertManager()
        events = manager.check_and_fire("a", "daily", 5.0, 10.0)
        warning_events = [e for e in events if e.level == AlertLevel.WARNING]
        assert warning_events[0].utilisation_pct == pytest.approx(50.0, abs=0.01)

    def test_event_message_contains_agent_id(self) -> None:
        manager = BudgetAlertManager()
        events = manager.check_and_fire("my-agent", "daily", 5.0, 10.0)
        assert any("my-agent" in e.message for e in events)

    def test_over_100_pct_fires_all_three_levels(self) -> None:
        manager = BudgetAlertManager()
        events = manager.check_and_fire("a", "daily", 12.0, 10.0)  # 120%
        levels = {e.level for e in events}
        assert levels == {AlertLevel.WARNING, AlertLevel.CRITICAL, AlertLevel.EXHAUSTED}


# ---------------------------------------------------------------------------
# BudgetAlertManager.reset_period
# ---------------------------------------------------------------------------


class TestResetPeriod:
    def test_reset_allows_alerts_to_fire_again(self) -> None:
        manager = BudgetAlertManager()
        manager.check_and_fire("a", "daily", 5.0, 10.0)
        manager.reset_period("a", "daily")
        events = manager.check_and_fire("a", "daily", 5.0, 10.0)
        assert any(e.level == AlertLevel.WARNING for e in events)

    def test_reset_unknown_agent_does_not_raise(self) -> None:
        manager = BudgetAlertManager()
        # Should not raise even if agent was never seen
        manager.reset_period("nonexistent-agent", "daily")

    def test_reset_one_period_does_not_affect_another(self) -> None:
        manager = BudgetAlertManager()
        manager.check_and_fire("a", "daily", 5.0, 10.0)
        manager.check_and_fire("a", "weekly", 5.0, 10.0)
        manager.reset_period("a", "daily")
        # daily should fire again
        events_daily = manager.check_and_fire("a", "daily", 5.0, 10.0)
        assert any(e.level == AlertLevel.WARNING for e in events_daily)
        # weekly should NOT fire again
        events_weekly = manager.check_and_fire("a", "weekly", 5.0, 10.0)
        assert not any(e.level == AlertLevel.WARNING for e in events_weekly)


# ---------------------------------------------------------------------------
# Callback registration and dispatch
# ---------------------------------------------------------------------------


class TestCallbacks:
    def test_registered_callback_is_invoked(self) -> None:
        manager = BudgetAlertManager()
        received: list[AlertEvent] = []
        manager.register_callback(received.append)
        manager.check_and_fire("a", "daily", 5.0, 10.0)
        assert len(received) == 1
        assert received[0].level == AlertLevel.WARNING

    def test_multiple_callbacks_all_invoked(self) -> None:
        manager = BudgetAlertManager()
        calls1: list[AlertEvent] = []
        calls2: list[AlertEvent] = []
        manager.register_callback(calls1.append)
        manager.register_callback(calls2.append)
        manager.check_and_fire("a", "daily", 10.0, 10.0)
        assert len(calls1) == 3  # warning + critical + exhausted
        assert len(calls2) == 3

    def test_deregister_callback_stops_invocation(self) -> None:
        manager = BudgetAlertManager()
        calls: list[AlertEvent] = []
        manager.register_callback(calls.append)
        manager.deregister_callback(calls.append)
        manager.check_and_fire("a", "daily", 10.0, 10.0)
        assert calls == []

    def test_deregister_returns_true_when_found(self) -> None:
        manager = BudgetAlertManager()
        cb = MagicMock()
        manager.register_callback(cb)
        result = manager.deregister_callback(cb)
        assert result is True

    def test_deregister_returns_false_when_not_found(self) -> None:
        manager = BudgetAlertManager()
        cb = MagicMock()
        result = manager.deregister_callback(cb)
        assert result is False

    def test_callback_exception_does_not_stop_other_callbacks(self) -> None:
        manager = BudgetAlertManager()

        def bad_callback(event: AlertEvent) -> None:
            raise RuntimeError("boom")

        good_calls: list[AlertEvent] = []
        manager.register_callback(bad_callback)
        manager.register_callback(good_calls.append)
        # Should not raise, and good_calls should still receive events
        manager.check_and_fire("a", "daily", 5.0, 10.0)
        assert len(good_calls) >= 1


# ---------------------------------------------------------------------------
# Webhook dispatch
# ---------------------------------------------------------------------------


class TestWebhookDispatch:
    def test_webhook_post_is_called_on_alert(self) -> None:
        manager = BudgetAlertManager(webhook_url="http://example.com/hook")
        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_resp = MagicMock()
            mock_resp.__enter__ = MagicMock(return_value=mock_resp)
            mock_resp.__exit__ = MagicMock(return_value=False)
            mock_resp.status = 200
            mock_urlopen.return_value = mock_resp
            manager.check_and_fire("a", "daily", 5.0, 10.0)
            assert mock_urlopen.called

    def test_webhook_failure_logs_warning_not_raises(self) -> None:
        manager = BudgetAlertManager(webhook_url="http://bad.invalid/hook")
        with patch("urllib.request.urlopen", side_effect=OSError("network error")):
            # Should not raise
            manager.check_and_fire("a", "daily", 5.0, 10.0)

    def test_webhook_4xx_logs_warning(self) -> None:
        manager = BudgetAlertManager(webhook_url="http://example.com/hook")
        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_resp = MagicMock()
            mock_resp.__enter__ = MagicMock(return_value=mock_resp)
            mock_resp.__exit__ = MagicMock(return_value=False)
            mock_resp.status = 400
            mock_urlopen.return_value = mock_resp
            with patch.object(
                __import__("agent_energy_budget.budget.alerts", fromlist=["logger"]).logger,
                "warning",
            ) as mock_warn:
                manager.check_and_fire("a", "daily", 5.0, 10.0)
                # Warning should have been called for the 400 status
                assert mock_warn.called
