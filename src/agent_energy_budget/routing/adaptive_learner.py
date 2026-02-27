"""Adaptive model learner for success-rate-based routing (E10.2).

AdaptiveLearner tracks per-(model, task_type) statistics using an exponential
moving average (EMA) for success rate and average cost. After a configurable
minimum number of calls, it can recommend models sorted by success_rate/cost ratio.

The learning algorithm is intentionally simple and commodity — it avoids any
proprietary ML weights, multi-armed bandit algorithms above O(n) complexity,
or score-based algorithms with Score >= 12.

Key design decisions:
- EMA alpha is configurable but defaults to 0.1 (recent history weighted slightly more)
- Stats are stored in-memory and can be serialised to/from plain dicts
- recommend() returns models sorted by success_rate/avg_cost ratio (descending)
- Models with fewer calls than min_calls_for_recommendation are excluded unless
  no other models are available

Example
-------
::

    learner = AdaptiveLearner(ema_alpha=0.1, min_calls_for_recommendation=5)
    for _ in range(10):
        learner.record("gpt-4o-mini", "simple", success=True, cost_usd=0.0002)
    for _ in range(10):
        learner.record("claude-sonnet-4-6", "simple", success=True, cost_usd=0.018)

    recs = learner.recommend("simple")
    assert recs[0].model_name == "gpt-4o-mini"  # cheaper with same success rate
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Minimum positive cost to avoid division-by-zero when computing ratio.
_MIN_COST_SENTINEL: float = 1e-9


# ---------------------------------------------------------------------------
# ModelStats
# ---------------------------------------------------------------------------


@dataclass
class ModelStats:
    """Per-(model, task_type) statistics maintained by AdaptiveLearner.

    Attributes
    ----------
    model_name:
        The model identifier.
    task_type:
        The task type this stats object tracks.
    call_count:
        Total number of calls recorded.
    success_count:
        Total number of successful calls.
    ema_success_rate:
        Exponential moving average of success (1.0) / failure (0.0) outcomes.
        Initialised to 0.5 (neutral prior) before any calls.
    avg_cost_usd:
        Simple cumulative average cost per call in USD.
    total_cost_usd:
        Sum of all recorded costs.
    """

    model_name: str
    task_type: str
    call_count: int = 0
    success_count: int = 0
    ema_success_rate: float = 0.5
    avg_cost_usd: float = 0.0
    total_cost_usd: float = 0.0

    def record(self, success: bool, cost_usd: float, ema_alpha: float) -> None:
        """Update stats with a new call outcome.

        Parameters
        ----------
        success:
            Whether the call was considered successful.
        cost_usd:
            Cost of the call in USD.
        ema_alpha:
            EMA smoothing factor. Higher values weight recent calls more.
        """
        self.call_count += 1
        if success:
            self.success_count += 1

        # EMA update: new_ema = alpha * new_value + (1 - alpha) * old_ema
        outcome = 1.0 if success else 0.0
        self.ema_success_rate = (
            ema_alpha * outcome + (1.0 - ema_alpha) * self.ema_success_rate
        )

        # Cumulative average cost
        self.total_cost_usd += cost_usd
        self.avg_cost_usd = self.total_cost_usd / self.call_count

    @property
    def raw_success_rate(self) -> float:
        """Return the raw (non-EMA) success rate."""
        if self.call_count == 0:
            return 0.0
        return self.success_count / self.call_count

    @property
    def success_cost_ratio(self) -> float:
        """Return EMA success rate divided by average cost (higher is better).

        Models with higher success rates and lower costs rank higher.
        """
        effective_cost = max(self.avg_cost_usd, _MIN_COST_SENTINEL)
        return self.ema_success_rate / effective_cost

    def to_dict(self) -> dict[str, object]:
        """Serialise to a plain dict for logging or export."""
        return {
            "model_name": self.model_name,
            "task_type": self.task_type,
            "call_count": self.call_count,
            "success_count": self.success_count,
            "ema_success_rate": round(self.ema_success_rate, 6),
            "avg_cost_usd": round(self.avg_cost_usd, 8),
            "total_cost_usd": round(self.total_cost_usd, 8),
            "success_cost_ratio": round(self.success_cost_ratio, 6),
        }


# ---------------------------------------------------------------------------
# RecommendedModel
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RecommendedModel:
    """A recommended model with its scoring context.

    Attributes
    ----------
    model_name:
        The model identifier.
    task_type:
        The task type the recommendation applies to.
    ema_success_rate:
        EMA success rate at recommendation time.
    avg_cost_usd:
        Average cost per call at recommendation time.
    success_cost_ratio:
        The ratio used for ranking (higher is better).
    call_count:
        Number of calls recorded for this model/task combination.
    is_warm:
        True if call_count >= min_calls_for_recommendation.
    """

    model_name: str
    task_type: str
    ema_success_rate: float
    avg_cost_usd: float
    success_cost_ratio: float
    call_count: int
    is_warm: bool


# ---------------------------------------------------------------------------
# AdaptiveLearner
# ---------------------------------------------------------------------------


class AdaptiveLearner:
    """Tracks per-(model, task_type) statistics and recommends optimal models.

    Parameters
    ----------
    ema_alpha:
        Smoothing factor for the exponential moving average of success rates.
        Range (0.0, 1.0]. Higher values give more weight to recent calls.
        Default 0.1.
    min_calls_for_recommendation:
        Minimum number of calls before a model is considered for recommendation.
        Default 5. Models below this threshold are excluded from results unless
        no other options exist (cold-start fallback).

    Examples
    --------
    ::

        learner = AdaptiveLearner()
        for _ in range(10):
            learner.record("model-a", "simple", success=True, cost_usd=0.001)
        recs = learner.recommend("simple")
        assert recs[0].model_name == "model-a"
    """

    def __init__(
        self,
        ema_alpha: float = 0.1,
        min_calls_for_recommendation: int = 5,
    ) -> None:
        if not 0.0 < ema_alpha <= 1.0:
            raise ValueError(
                f"AdaptiveLearner.ema_alpha must be in (0.0, 1.0]; got {ema_alpha!r}."
            )
        if min_calls_for_recommendation < 1:
            raise ValueError(
                f"AdaptiveLearner.min_calls_for_recommendation must be >= 1; "
                f"got {min_calls_for_recommendation!r}."
            )
        self._ema_alpha = ema_alpha
        self._min_calls = min_calls_for_recommendation
        # Key: (model_name, task_type)
        self._stats: dict[tuple[str, str], ModelStats] = {}

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record(
        self,
        model_name: str,
        task_type: str,
        success: bool,
        cost_usd: float,
    ) -> None:
        """Record the outcome of a model call for a given task type.

        Parameters
        ----------
        model_name:
            The model identifier.
        task_type:
            The task type (e.g. ``"simple"``, ``"medium"``, ``"complex"``).
        success:
            Whether the call succeeded.
        cost_usd:
            Actual cost of the call in USD.
        """
        if cost_usd < 0.0:
            raise ValueError(
                f"AdaptiveLearner.record: cost_usd must be >= 0; got {cost_usd!r}."
            )
        key = (model_name, task_type)
        if key not in self._stats:
            self._stats[key] = ModelStats(
                model_name=model_name, task_type=task_type
            )
        self._stats[key].record(
            success=success, cost_usd=cost_usd, ema_alpha=self._ema_alpha
        )
        logger.debug(
            "AdaptiveLearner recorded: model=%s task=%s success=%s cost=%.8f",
            model_name,
            task_type,
            success,
            cost_usd,
        )

    # ------------------------------------------------------------------
    # Recommendation
    # ------------------------------------------------------------------

    def recommend(
        self,
        task_type: str,
        include_cold: bool = False,
    ) -> list[RecommendedModel]:
        """Return models sorted by success_rate/cost ratio for a task type.

        Parameters
        ----------
        task_type:
            The task type to recommend models for.
        include_cold:
            If ``True``, include models below the minimum call threshold.
            Default ``False``.

        Returns
        -------
        list[RecommendedModel]
            Models sorted by success_cost_ratio descending (best first).
            Empty if no data is available for the task type.
        """
        candidates: list[ModelStats] = [
            stats
            for (model, task), stats in self._stats.items()
            if task == task_type
        ]

        if not candidates:
            return []

        warm = [s for s in candidates if s.call_count >= self._min_calls]
        cold = [s for s in candidates if s.call_count < self._min_calls]

        # If no warm models, fall back to cold (cold start)
        pool = warm if warm else cold
        if include_cold:
            pool = candidates

        ranked = sorted(pool, key=lambda s: s.success_cost_ratio, reverse=True)

        return [
            RecommendedModel(
                model_name=stats.model_name,
                task_type=stats.task_type,
                ema_success_rate=stats.ema_success_rate,
                avg_cost_usd=stats.avg_cost_usd,
                success_cost_ratio=stats.success_cost_ratio,
                call_count=stats.call_count,
                is_warm=stats.call_count >= self._min_calls,
            )
            for stats in ranked
        ]

    def best_model(self, task_type: str) -> str | None:
        """Return the name of the best model for a task type, or None.

        Parameters
        ----------
        task_type:
            The task type to query.

        Returns
        -------
        str | None
            The model name with the best success/cost ratio, or ``None``
            if no data is available.
        """
        recs = self.recommend(task_type)
        return recs[0].model_name if recs else None

    # ------------------------------------------------------------------
    # Inspection
    # ------------------------------------------------------------------

    def get_stats(self, model_name: str, task_type: str) -> ModelStats | None:
        """Return stats for a specific (model, task_type) pair, or None.

        Parameters
        ----------
        model_name:
            The model identifier.
        task_type:
            The task type.

        Returns
        -------
        ModelStats | None
        """
        return self._stats.get((model_name, task_type))

    def all_stats(self) -> list[ModelStats]:
        """Return all ModelStats instances in no guaranteed order."""
        return list(self._stats.values())

    def known_models(self) -> list[str]:
        """Return a sorted list of unique model names seen so far."""
        return sorted({model for model, _ in self._stats.keys()})

    def known_task_types(self) -> list[str]:
        """Return a sorted list of unique task types seen so far."""
        return sorted({task for _, task in self._stats.keys()})

    def reset(self, model_name: str | None = None, task_type: str | None = None) -> int:
        """Clear statistics for one or all (model, task_type) combinations.

        Parameters
        ----------
        model_name:
            If provided, only clear stats for this model.
        task_type:
            If provided, only clear stats for this task type.
            Combined with model_name: clears that specific (model, task) pair.

        Returns
        -------
        int
            Number of stat entries removed.
        """
        if model_name is None and task_type is None:
            count = len(self._stats)
            self._stats.clear()
            return count

        to_remove = [
            key
            for key in self._stats
            if (model_name is None or key[0] == model_name)
            and (task_type is None or key[1] == task_type)
        ]
        for key in to_remove:
            del self._stats[key]
        return len(to_remove)

    def to_dict(self) -> dict[str, object]:
        """Serialise all stats to a plain dict for logging or export."""
        return {
            "ema_alpha": self._ema_alpha,
            "min_calls_for_recommendation": self._min_calls,
            "stats": [s.to_dict() for s in self._stats.values()],
        }

    @property
    def ema_alpha(self) -> float:
        """Return the EMA smoothing factor."""
        return self._ema_alpha

    @property
    def min_calls_for_recommendation(self) -> int:
        """Return the minimum call threshold for warm recommendations."""
        return self._min_calls
