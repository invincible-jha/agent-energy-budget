"""Tests for agent_energy_budget.integrations.litellm_adapter.

All LiteLLM calls are mocked — the litellm package is not required in CI.
"""
from __future__ import annotations

import sys
import types
from typing import Any
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Fixture: inject a fake litellm module before the adapter is imported
# ---------------------------------------------------------------------------


def _make_mock_litellm() -> types.ModuleType:
    """Build a minimal litellm stub."""
    mod = types.ModuleType("litellm")
    mock_response = MagicMock()
    mock_response.usage.prompt_tokens = 50
    mock_response.usage.completion_tokens = 100
    mod.completion = MagicMock(return_value=mock_response)
    return mod


@pytest.fixture(autouse=True)
def inject_litellm(monkeypatch: pytest.MonkeyPatch) -> None:
    """Inject fake litellm before each test and remove after."""
    fake_litellm = _make_mock_litellm()
    monkeypatch.setitem(sys.modules, "litellm", fake_litellm)
    # Ensure the adapter module is re-evaluated with the stub
    monkeypatch.delitem(
        sys.modules,
        "agent_energy_budget.integrations.litellm_adapter",
        raising=False,
    )


# ---------------------------------------------------------------------------
# Import adapter under test (after litellm stub is in place)
# ---------------------------------------------------------------------------


def _import_adapter() -> Any:
    from agent_energy_budget.integrations import litellm_adapter  # noqa: PLC0415

    return litellm_adapter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_wrapper(**kwargs: Any) -> Any:
    adapter = _import_adapter()
    return adapter.LiteLLMBudgetWrapper(**kwargs)


def _make_messages() -> list[dict[str, str]]:
    return [{"role": "user", "content": "Hello, world!"}]


# ---------------------------------------------------------------------------
# LiteLLMBudgetWrapper — construction
# ---------------------------------------------------------------------------


class TestLiteLLMBudgetWrapperConstruction:
    def test_default_construction(self) -> None:
        wrapper = _make_wrapper()
        assert wrapper is not None

    def test_session_budget_stored(self) -> None:
        wrapper = _make_wrapper(session_budget_usd=1.0)
        assert wrapper._session_budget == 1.0

    def test_no_session_budget_by_default(self) -> None:
        wrapper = _make_wrapper()
        assert wrapper._session_budget is None

    def test_repr_contains_class_name(self) -> None:
        wrapper = _make_wrapper()
        assert "LiteLLMBudgetWrapper" in repr(wrapper)


# ---------------------------------------------------------------------------
# predict_before_call
# ---------------------------------------------------------------------------


class TestPredictBeforeCall:
    def test_returns_prediction_result(self) -> None:
        wrapper = _make_wrapper()
        result = wrapper.predict_before_call(
            model="gpt-4o-mini",
            messages=_make_messages(),
        )
        from agent_energy_budget.prediction.predictor import PredictionResult  # noqa: PLC0415

        assert isinstance(result, PredictionResult)

    def test_model_stored_in_result(self) -> None:
        wrapper = _make_wrapper()
        result = wrapper.predict_before_call(
            model="gpt-4o-mini",
            messages=_make_messages(),
        )
        assert result.model == "gpt-4o-mini"

    def test_budget_sets_will_exceed_budget(self) -> None:
        wrapper = _make_wrapper()
        # Zero budget — any cost exceeds it
        result = wrapper.predict_before_call(
            model="gpt-4o-mini",
            messages=_make_messages(),
            budget_usd=0.0,
        )
        assert result.will_exceed_budget is True

    def test_high_budget_does_not_exceed(self) -> None:
        wrapper = _make_wrapper()
        result = wrapper.predict_before_call(
            model="gpt-4o-mini",
            messages=_make_messages(),
            budget_usd=100.0,
        )
        assert result.will_exceed_budget is False


# ---------------------------------------------------------------------------
# completion_with_budget — success path
# ---------------------------------------------------------------------------


class TestCompletionWithBudget:
    def test_returns_litellm_response_under_budget(self) -> None:
        wrapper = _make_wrapper()
        response = wrapper.completion_with_budget(
            model="gpt-4o-mini",
            messages=_make_messages(),
            budget_limit=100.0,
        )
        assert response is not None

    def test_calls_litellm_completion(self) -> None:
        # Verify the call was recorded (proves litellm.completion was reached
        # and the response was processed by the wrapper).
        wrapper = _make_wrapper()
        wrapper.completion_with_budget(
            model="gpt-4o-mini",
            messages=_make_messages(),
            budget_limit=100.0,
        )
        # A call record exists iff litellm.completion() was invoked and returned
        summary = wrapper.session_summary()
        assert summary.total_calls == 1

    def test_call_recorded_in_session(self) -> None:
        wrapper = _make_wrapper()
        wrapper.completion_with_budget(
            model="gpt-4o-mini",
            messages=_make_messages(),
            budget_limit=100.0,
        )
        summary = wrapper.session_summary()
        assert summary.total_calls == 1

    def test_cumulative_calls_tracked(self) -> None:
        wrapper = _make_wrapper()
        for _ in range(3):
            wrapper.completion_with_budget(
                model="gpt-4o-mini",
                messages=_make_messages(),
                budget_limit=100.0,
            )
        summary = wrapper.session_summary()
        assert summary.total_calls == 3


# ---------------------------------------------------------------------------
# completion_with_budget — budget exceeded
# ---------------------------------------------------------------------------


class TestBudgetExceeded:
    def test_raises_budget_exceeded_error(self) -> None:
        adapter = _import_adapter()
        wrapper = _make_wrapper()
        with pytest.raises(adapter.BudgetExceededError):
            wrapper.completion_with_budget(
                model="gpt-4o-mini",
                messages=_make_messages(),
                budget_limit=0.0,
            )

    def test_no_litellm_call_when_budget_exceeded(self) -> None:
        adapter = _import_adapter()
        wrapper = _make_wrapper()
        litellm_mod = sys.modules["litellm"]
        litellm_mod.completion.reset_mock()
        with pytest.raises(adapter.BudgetExceededError):
            wrapper.completion_with_budget(
                model="gpt-4o-mini",
                messages=_make_messages(),
                budget_limit=0.0,
            )
        litellm_mod.completion.assert_not_called()

    def test_budget_exceeded_error_attributes(self) -> None:
        adapter = _import_adapter()
        wrapper = _make_wrapper()
        with pytest.raises(adapter.BudgetExceededError) as exc_info:
            wrapper.completion_with_budget(
                model="gpt-4o-mini",
                messages=_make_messages(),
                budget_limit=0.0,
            )
        err = exc_info.value
        assert err.budget_limit_usd == 0.0
        assert err.predicted_cost_usd >= 0.0
        assert err.prediction is not None

    def test_session_not_polluted_when_exceeded(self) -> None:
        adapter = _import_adapter()
        wrapper = _make_wrapper()
        try:
            wrapper.completion_with_budget(
                model="gpt-4o-mini",
                messages=_make_messages(),
                budget_limit=0.0,
            )
        except adapter.BudgetExceededError:
            pass
        summary = wrapper.session_summary()
        assert summary.total_calls == 0


# ---------------------------------------------------------------------------
# Session budget enforcement
# ---------------------------------------------------------------------------


class TestSessionBudget:
    def test_session_budget_exhausted_blocks_calls(self) -> None:
        adapter = _import_adapter()
        # Session budget of zero — first call should be blocked
        wrapper = _make_wrapper(session_budget_usd=0.0)
        with pytest.raises(adapter.BudgetExceededError):
            wrapper.completion_with_budget(
                model="gpt-4o-mini",
                messages=_make_messages(),
                budget_limit=100.0,
            )

    def test_session_budget_large_allows_calls(self) -> None:
        wrapper = _make_wrapper(session_budget_usd=1000.0)
        response = wrapper.completion_with_budget(
            model="gpt-4o-mini",
            messages=_make_messages(),
            budget_limit=100.0,
        )
        assert response is not None


# ---------------------------------------------------------------------------
# session_summary and reset
# ---------------------------------------------------------------------------


class TestSessionSummary:
    def test_initial_summary_is_zero(self) -> None:
        wrapper = _make_wrapper()
        summary = wrapper.session_summary()
        assert summary.total_calls == 0
        assert summary.total_actual_cost_usd == 0.0
        assert summary.total_predicted_cost_usd == 0.0

    def test_summary_has_call_records(self) -> None:
        wrapper = _make_wrapper()
        wrapper.completion_with_budget(
            model="gpt-4o-mini",
            messages=_make_messages(),
            budget_limit=100.0,
        )
        summary = wrapper.session_summary()
        assert len(summary.call_records) == 1
        record = summary.call_records[0]
        assert record.model == "gpt-4o-mini"

    def test_reset_clears_records(self) -> None:
        wrapper = _make_wrapper()
        wrapper.completion_with_budget(
            model="gpt-4o-mini",
            messages=_make_messages(),
            budget_limit=100.0,
        )
        wrapper.reset_session()
        summary = wrapper.session_summary()
        assert summary.total_calls == 0

    def test_token_totals_accumulated(self) -> None:
        wrapper = _make_wrapper()
        wrapper.completion_with_budget(
            model="gpt-4o-mini",
            messages=_make_messages(),
            budget_limit=100.0,
        )
        summary = wrapper.session_summary()
        # The mock returns prompt_tokens=50, completion_tokens=100
        assert summary.total_input_tokens == 50
        assert summary.total_output_tokens == 100


# ---------------------------------------------------------------------------
# BudgetExceededError
# ---------------------------------------------------------------------------


class TestBudgetExceededError:
    def test_inherits_from_exception(self) -> None:
        adapter = _import_adapter()
        assert issubclass(adapter.BudgetExceededError, Exception)

    def test_str_representation(self) -> None:
        adapter = _import_adapter()
        from agent_energy_budget.prediction.predictor import PredictionResult  # noqa: PLC0415

        dummy_prediction = PredictionResult(
            estimated_cost_usd=0.05,
            input_tokens=100,
            estimated_output_tokens=50,
            model="gpt-4o-mini",
            will_exceed_budget=True,
            confidence=0.8,
        )
        err = adapter.BudgetExceededError(
            message="Test error",
            predicted_cost_usd=0.05,
            budget_limit_usd=0.01,
            prediction=dummy_prediction,
        )
        assert "Test error" in str(err)
        assert err.predicted_cost_usd == 0.05
        assert err.budget_limit_usd == 0.01
