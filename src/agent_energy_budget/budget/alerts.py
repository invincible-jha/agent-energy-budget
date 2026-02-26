"""Budget alert management.

Fires alerts at configurable utilisation thresholds. Supports three
delivery mechanisms: console output, registered Python callbacks, and
HTTP webhook POST requests.
"""
from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Callable

logger = logging.getLogger(__name__)


class AlertLevel(str, Enum):
    """Severity level of a budget alert."""

    WARNING = "warning"
    CRITICAL = "critical"
    EXHAUSTED = "exhausted"


@dataclass(frozen=True)
class AlertEvent:
    """Immutable record of a fired budget alert.

    Parameters
    ----------
    agent_id:
        The agent whose budget triggered the alert.
    level:
        Alert severity.
    utilisation_pct:
        Current utilisation percentage at the time of the alert.
    spent_usd:
        Amount spent in USD at the time of the alert.
    limit_usd:
        The budget limit in USD that was approached/exceeded.
    period:
        Budget period label ("daily", "weekly", or "monthly").
    message:
        Human-readable description.
    fired_at:
        UTC timestamp of the alert.
    """

    agent_id: str
    level: AlertLevel
    utilisation_pct: float
    spent_usd: float
    limit_usd: float
    period: str
    message: str
    fired_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict[str, object]:
        """Serialise the event to a JSON-safe dict."""
        return {
            "agent_id": self.agent_id,
            "level": self.level.value,
            "utilisation_pct": self.utilisation_pct,
            "spent_usd": self.spent_usd,
            "limit_usd": self.limit_usd,
            "period": self.period,
            "message": self.message,
            "fired_at": self.fired_at.isoformat(),
        }


AlertCallback = Callable[[AlertEvent], None]


class BudgetAlertManager:
    """Manage and dispatch budget threshold alerts.

    Tracks which thresholds have already fired for each agent/period
    combination so that each level fires at most once per period.

    Parameters
    ----------
    warning_threshold:
        Utilisation % at which WARNING fires (default 50.0).
    critical_threshold:
        Utilisation % at which CRITICAL fires (default 80.0).
    exhausted_threshold:
        Utilisation % at which EXHAUSTED fires (default 100.0).
    webhook_url:
        Optional URL to POST alert JSON payloads to.
    webhook_timeout:
        Seconds before a webhook POST times out (default 5.0).
    """

    def __init__(
        self,
        warning_threshold: float = 50.0,
        critical_threshold: float = 80.0,
        exhausted_threshold: float = 100.0,
        webhook_url: str | None = None,
        webhook_timeout: float = 5.0,
    ) -> None:
        self._thresholds: dict[AlertLevel, float] = {
            AlertLevel.WARNING: warning_threshold,
            AlertLevel.CRITICAL: critical_threshold,
            AlertLevel.EXHAUSTED: exhausted_threshold,
        }
        self._webhook_url = webhook_url
        self._webhook_timeout = webhook_timeout
        self._callbacks: list[AlertCallback] = []
        # fired[agent_id][period][level] = True when already fired this period
        self._fired: dict[str, dict[str, set[AlertLevel]]] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Callback registration
    # ------------------------------------------------------------------

    def register_callback(self, callback: AlertCallback) -> None:
        """Register a Python callable to invoke on every alert.

        Parameters
        ----------
        callback:
            Function accepting a single :class:`AlertEvent` argument.
        """
        with self._lock:
            self._callbacks.append(callback)

    def deregister_callback(self, callback: AlertCallback) -> bool:
        """Remove a previously registered callback.

        Parameters
        ----------
        callback:
            The exact callable reference passed to :meth:`register_callback`.

        Returns
        -------
        bool
            True if the callback was found and removed; False otherwise.
        """
        with self._lock:
            try:
                self._callbacks.remove(callback)
                return True
            except ValueError:
                return False

    # ------------------------------------------------------------------
    # Alert evaluation
    # ------------------------------------------------------------------

    def check_and_fire(
        self,
        agent_id: str,
        period: str,
        spent_usd: float,
        limit_usd: float,
    ) -> list[AlertEvent]:
        """Evaluate utilisation and fire any due alerts.

        Parameters
        ----------
        agent_id:
            Agent identifier.
        period:
            Period label ("daily", "weekly", "monthly").
        spent_usd:
            Amount spent so far in this period.
        limit_usd:
            Period limit in USD.

        Returns
        -------
        list[AlertEvent]
            All alerts that were fired in this evaluation (may be empty).
        """
        if limit_usd <= 0:
            return []

        utilisation = (spent_usd / limit_usd) * 100.0
        fired_events: list[AlertEvent] = []

        with self._lock:
            agent_fired = self._fired.setdefault(agent_id, {})
            period_fired = agent_fired.setdefault(period, set())

            for level in (AlertLevel.WARNING, AlertLevel.CRITICAL, AlertLevel.EXHAUSTED):
                threshold = self._thresholds[level]
                if utilisation >= threshold and level not in period_fired:
                    event = AlertEvent(
                        agent_id=agent_id,
                        level=level,
                        utilisation_pct=round(utilisation, 2),
                        spent_usd=spent_usd,
                        limit_usd=limit_usd,
                        period=period,
                        message=(
                            f"Agent '{agent_id}' has reached {utilisation:.1f}% of its "
                            f"{period} budget (${spent_usd:.4f} of ${limit_usd:.2f})."
                        ),
                    )
                    period_fired.add(level)
                    fired_events.append(event)

        # Dispatch outside the lock to avoid holding it during I/O
        for event in fired_events:
            self._dispatch(event)

        return fired_events

    def reset_period(self, agent_id: str, period: str) -> None:
        """Clear fired-alert state for a new period.

        Call this at the start of each new day/week/month to allow
        thresholds to fire again.

        Parameters
        ----------
        agent_id:
            Agent identifier.
        period:
            Period label to reset.
        """
        with self._lock:
            self._fired.get(agent_id, {}).pop(period, None)

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------

    def _dispatch(self, event: AlertEvent) -> None:
        """Dispatch an alert to all configured delivery channels."""
        self._dispatch_console(event)
        self._dispatch_callbacks(event)
        if self._webhook_url:
            self._dispatch_webhook(event)

    def _dispatch_console(self, event: AlertEvent) -> None:
        level_map: dict[AlertLevel, int] = {
            AlertLevel.WARNING: logging.WARNING,
            AlertLevel.CRITICAL: logging.ERROR,
            AlertLevel.EXHAUSTED: logging.CRITICAL,
        }
        logger.log(level_map[event.level], "BUDGET ALERT [%s]: %s", event.level.value, event.message)

    def _dispatch_callbacks(self, event: AlertEvent) -> None:
        for callback in self._callbacks:
            try:
                callback(event)
            except Exception:
                logger.exception("Budget alert callback raised an exception")

    def _dispatch_webhook(self, event: AlertEvent) -> None:
        import urllib.request

        payload = json.dumps(event.to_dict()).encode("utf-8")
        req = urllib.request.Request(
            self._webhook_url,  # type: ignore[arg-type]
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self._webhook_timeout) as resp:
                status = resp.status
            if status >= 400:
                logger.warning(
                    "Webhook POST returned HTTP %d for alert %s", status, event.level.value
                )
        except Exception as exc:
            logger.warning("Failed to POST budget alert webhook: %s", exc)
