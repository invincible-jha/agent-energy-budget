"""Tests for AdaptiveLearner and ModelStats (E10.2)."""
from __future__ import annotations

import pytest

from agent_energy_budget.routing.adaptive_learner import (
    AdaptiveLearner,
    ModelStats,
    RecommendedModel,
)


# ---------------------------------------------------------------------------
# ModelStats
# ---------------------------------------------------------------------------

class TestModelStats:
    def test_initial_call_count_zero(self) -> None:
        stats = ModelStats(model_name="model-a", task_type="simple")
        assert stats.call_count == 0

    def test_initial_success_count_zero(self) -> None:
        stats = ModelStats(model_name="model-a", task_type="simple")
        assert stats.success_count == 0

    def test_initial_ema_is_neutral_prior(self) -> None:
        stats = ModelStats(model_name="model-a", task_type="simple")
        assert stats.ema_success_rate == 0.5

    def test_record_increments_call_count(self) -> None:
        stats = ModelStats(model_name="model-a", task_type="simple")
        stats.record(success=True, cost_usd=0.001, ema_alpha=0.1)
        assert stats.call_count == 1

    def test_record_success_increments_success_count(self) -> None:
        stats = ModelStats(model_name="model-a", task_type="simple")
        stats.record(success=True, cost_usd=0.001, ema_alpha=0.1)
        assert stats.success_count == 1

    def test_record_failure_does_not_increment_success_count(self) -> None:
        stats = ModelStats(model_name="model-a", task_type="simple")
        stats.record(success=False, cost_usd=0.001, ema_alpha=0.1)
        assert stats.success_count == 0
        assert stats.call_count == 1

    def test_ema_updates_toward_success(self) -> None:
        stats = ModelStats(model_name="model-a", task_type="simple")
        initial_ema = stats.ema_success_rate
        stats.record(success=True, cost_usd=0.001, ema_alpha=0.5)
        assert stats.ema_success_rate > initial_ema

    def test_ema_updates_toward_failure(self) -> None:
        stats = ModelStats(model_name="model-a", task_type="simple")
        initial_ema = stats.ema_success_rate
        stats.record(success=False, cost_usd=0.001, ema_alpha=0.5)
        assert stats.ema_success_rate < initial_ema

    def test_avg_cost_computed_correctly(self) -> None:
        stats = ModelStats(model_name="model-a", task_type="simple")
        stats.record(success=True, cost_usd=0.001, ema_alpha=0.1)
        stats.record(success=True, cost_usd=0.003, ema_alpha=0.1)
        assert abs(stats.avg_cost_usd - 0.002) < 1e-9

    def test_raw_success_rate_zero_calls(self) -> None:
        stats = ModelStats(model_name="model-a", task_type="simple")
        assert stats.raw_success_rate == 0.0

    def test_raw_success_rate_all_success(self) -> None:
        stats = ModelStats(model_name="model-a", task_type="simple")
        for _ in range(10):
            stats.record(success=True, cost_usd=0.001, ema_alpha=0.1)
        assert stats.raw_success_rate == 1.0

    def test_raw_success_rate_half(self) -> None:
        stats = ModelStats(model_name="model-a", task_type="simple")
        for _ in range(5):
            stats.record(success=True, cost_usd=0.001, ema_alpha=0.1)
        for _ in range(5):
            stats.record(success=False, cost_usd=0.001, ema_alpha=0.1)
        assert abs(stats.raw_success_rate - 0.5) < 1e-9

    def test_success_cost_ratio_positive(self) -> None:
        stats = ModelStats(model_name="model-a", task_type="simple")
        stats.record(success=True, cost_usd=0.001, ema_alpha=0.1)
        assert stats.success_cost_ratio > 0.0

    def test_success_cost_ratio_cheaper_model_ranks_higher(self) -> None:
        cheap = ModelStats(model_name="cheap", task_type="simple")
        expensive = ModelStats(model_name="expensive", task_type="simple")
        for _ in range(5):
            cheap.record(success=True, cost_usd=0.001, ema_alpha=0.1)
            expensive.record(success=True, cost_usd=0.01, ema_alpha=0.1)
        assert cheap.success_cost_ratio > expensive.success_cost_ratio

    def test_to_dict_contains_all_fields(self) -> None:
        stats = ModelStats(model_name="model-a", task_type="simple")
        stats.record(success=True, cost_usd=0.001, ema_alpha=0.1)
        result = stats.to_dict()
        assert "model_name" in result
        assert "task_type" in result
        assert "call_count" in result
        assert "ema_success_rate" in result
        assert "avg_cost_usd" in result
        assert "success_cost_ratio" in result


# ---------------------------------------------------------------------------
# AdaptiveLearner — construction
# ---------------------------------------------------------------------------

class TestAdaptiveLearnerConstruction:
    def test_default_ema_alpha(self) -> None:
        learner = AdaptiveLearner()
        assert learner.ema_alpha == 0.1

    def test_custom_ema_alpha(self) -> None:
        learner = AdaptiveLearner(ema_alpha=0.3)
        assert learner.ema_alpha == 0.3

    def test_invalid_ema_alpha_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="ema_alpha"):
            AdaptiveLearner(ema_alpha=0.0)

    def test_invalid_ema_alpha_above_one_raises(self) -> None:
        with pytest.raises(ValueError, match="ema_alpha"):
            AdaptiveLearner(ema_alpha=1.1)

    def test_invalid_min_calls_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="min_calls_for_recommendation"):
            AdaptiveLearner(min_calls_for_recommendation=0)

    def test_default_min_calls(self) -> None:
        learner = AdaptiveLearner()
        assert learner.min_calls_for_recommendation == 5


# ---------------------------------------------------------------------------
# AdaptiveLearner — recording
# ---------------------------------------------------------------------------

class TestAdaptiveLearnerRecord:
    def test_record_creates_stats_entry(self) -> None:
        learner = AdaptiveLearner()
        learner.record("model-a", "simple", success=True, cost_usd=0.001)
        stats = learner.get_stats("model-a", "simple")
        assert stats is not None

    def test_record_increments_call_count(self) -> None:
        learner = AdaptiveLearner()
        learner.record("model-a", "simple", success=True, cost_usd=0.001)
        learner.record("model-a", "simple", success=True, cost_usd=0.001)
        stats = learner.get_stats("model-a", "simple")
        assert stats is not None
        assert stats.call_count == 2

    def test_record_negative_cost_raises(self) -> None:
        learner = AdaptiveLearner()
        with pytest.raises(ValueError, match="cost_usd"):
            learner.record("model-a", "simple", success=True, cost_usd=-0.001)

    def test_record_zero_cost_is_valid(self) -> None:
        learner = AdaptiveLearner()
        learner.record("model-a", "simple", success=True, cost_usd=0.0)
        stats = learner.get_stats("model-a", "simple")
        assert stats is not None
        assert stats.call_count == 1

    def test_records_separate_task_types(self) -> None:
        learner = AdaptiveLearner()
        learner.record("model-a", "simple", success=True, cost_usd=0.001)
        learner.record("model-a", "complex", success=False, cost_usd=0.01)
        assert learner.get_stats("model-a", "simple") is not None
        assert learner.get_stats("model-a", "complex") is not None

    def test_get_stats_unknown_returns_none(self) -> None:
        learner = AdaptiveLearner()
        assert learner.get_stats("unknown-model", "simple") is None


# ---------------------------------------------------------------------------
# AdaptiveLearner — recommendation
# ---------------------------------------------------------------------------

class TestAdaptiveLearnerRecommend:
    def _warm_learner(self) -> AdaptiveLearner:
        learner = AdaptiveLearner(min_calls_for_recommendation=5)
        # model-a: 10 calls, all success, cheap
        for _ in range(10):
            learner.record("model-a", "simple", success=True, cost_usd=0.001)
        # model-b: 10 calls, all success, expensive
        for _ in range(10):
            learner.record("model-b", "simple", success=True, cost_usd=0.01)
        # model-c: 10 calls, 50% success, cheap
        for _ in range(5):
            learner.record("model-c", "simple", success=True, cost_usd=0.001)
        for _ in range(5):
            learner.record("model-c", "simple", success=False, cost_usd=0.001)
        return learner

    def test_recommend_returns_list(self) -> None:
        learner = self._warm_learner()
        recs = learner.recommend("simple")
        assert isinstance(recs, list)

    def test_recommend_returns_all_warm_models(self) -> None:
        learner = self._warm_learner()
        recs = learner.recommend("simple")
        assert len(recs) == 3

    def test_cheaper_model_ranks_higher_than_expensive(self) -> None:
        learner = self._warm_learner()
        recs = learner.recommend("simple")
        model_names = [r.model_name for r in recs]
        assert model_names.index("model-a") < model_names.index("model-b")

    def test_higher_success_rate_ranks_better(self) -> None:
        learner = AdaptiveLearner(min_calls_for_recommendation=5)
        for _ in range(10):
            learner.record("high-success", "simple", success=True, cost_usd=0.005)
        for _ in range(5):
            learner.record("low-success", "simple", success=True, cost_usd=0.005)
        for _ in range(5):
            learner.record("low-success", "simple", success=False, cost_usd=0.005)
        recs = learner.recommend("simple")
        assert recs[0].model_name == "high-success"

    def test_cold_models_excluded_from_default_recommend(self) -> None:
        learner = AdaptiveLearner(min_calls_for_recommendation=5)
        learner.record("cold-model", "simple", success=True, cost_usd=0.001)
        recs = learner.recommend("simple")
        # cold model only: cold-start fallback returns it
        assert len(recs) == 1
        assert recs[0].model_name == "cold-model"

    def test_include_cold_flag(self) -> None:
        learner = AdaptiveLearner(min_calls_for_recommendation=5)
        for _ in range(10):
            learner.record("warm-model", "simple", success=True, cost_usd=0.001)
        learner.record("cold-model", "simple", success=True, cost_usd=0.0001)
        recs = learner.recommend("simple", include_cold=True)
        model_names = {r.model_name for r in recs}
        assert "cold-model" in model_names

    def test_recommend_unknown_task_type_empty(self) -> None:
        learner = AdaptiveLearner()
        recs = learner.recommend("nonexistent_task")
        assert recs == []

    def test_recommended_model_is_warm_flag(self) -> None:
        learner = AdaptiveLearner(min_calls_for_recommendation=5)
        for _ in range(10):
            learner.record("model-a", "simple", success=True, cost_usd=0.001)
        recs = learner.recommend("simple")
        assert recs[0].is_warm is True

    def test_best_model_returns_top(self) -> None:
        learner = self._warm_learner()
        best = learner.best_model("simple")
        assert best == "model-a"

    def test_best_model_no_data_returns_none(self) -> None:
        learner = AdaptiveLearner()
        assert learner.best_model("nonexistent") is None


# ---------------------------------------------------------------------------
# AdaptiveLearner — inspection and management
# ---------------------------------------------------------------------------

class TestAdaptiveLearnerInspection:
    def test_known_models_sorted(self) -> None:
        learner = AdaptiveLearner()
        learner.record("model-b", "simple", success=True, cost_usd=0.001)
        learner.record("model-a", "simple", success=True, cost_usd=0.001)
        assert learner.known_models() == ["model-a", "model-b"]

    def test_known_task_types_sorted(self) -> None:
        learner = AdaptiveLearner()
        learner.record("model-a", "simple", success=True, cost_usd=0.001)
        learner.record("model-a", "complex", success=True, cost_usd=0.01)
        assert learner.known_task_types() == ["complex", "simple"]

    def test_all_stats_returns_all_entries(self) -> None:
        learner = AdaptiveLearner()
        learner.record("model-a", "simple", success=True, cost_usd=0.001)
        learner.record("model-b", "complex", success=False, cost_usd=0.01)
        assert len(learner.all_stats()) == 2

    def test_reset_all_clears_stats(self) -> None:
        learner = AdaptiveLearner()
        learner.record("model-a", "simple", success=True, cost_usd=0.001)
        removed = learner.reset()
        assert removed == 1
        assert len(learner.all_stats()) == 0

    def test_reset_by_model_name(self) -> None:
        learner = AdaptiveLearner()
        learner.record("model-a", "simple", success=True, cost_usd=0.001)
        learner.record("model-b", "simple", success=True, cost_usd=0.001)
        learner.reset(model_name="model-a")
        assert learner.get_stats("model-a", "simple") is None
        assert learner.get_stats("model-b", "simple") is not None

    def test_reset_by_task_type(self) -> None:
        learner = AdaptiveLearner()
        learner.record("model-a", "simple", success=True, cost_usd=0.001)
        learner.record("model-a", "complex", success=True, cost_usd=0.01)
        learner.reset(task_type="simple")
        assert learner.get_stats("model-a", "simple") is None
        assert learner.get_stats("model-a", "complex") is not None

    def test_to_dict_contains_stats(self) -> None:
        learner = AdaptiveLearner()
        learner.record("model-a", "simple", success=True, cost_usd=0.001)
        result = learner.to_dict()
        assert "stats" in result
        assert len(result["stats"]) == 1  # type: ignore[arg-type]

    def test_to_dict_contains_ema_alpha(self) -> None:
        learner = AdaptiveLearner(ema_alpha=0.2)
        result = learner.to_dict()
        assert result["ema_alpha"] == 0.2
