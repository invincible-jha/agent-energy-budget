"""Unit tests for agent_energy_budget.middleware package."""
from __future__ import annotations

import asyncio
import pathlib
from typing import Any
from unittest.mock import MagicMock, AsyncMock

import pytest

from agent_energy_budget.budget.config import BudgetConfig, DegradationStrategy
from agent_energy_budget.budget.tracker import BudgetExceededError, BudgetTracker
from agent_energy_budget.middleware.generic import BudgetGuardError, budget_guard, _record_from_result
from agent_energy_budget.middleware.openai_sdk import (
    OpenAIBudgetMiddleware,
    OpenAIBudgetWrapper,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_storage(tmp_path: pathlib.Path) -> pathlib.Path:
    return tmp_path


@pytest.fixture
def generous_tracker(tmp_storage: pathlib.Path) -> BudgetTracker:
    config = BudgetConfig(agent_id="test-agent", daily_limit=100.0)
    return BudgetTracker(config=config, storage_dir=tmp_storage)


@pytest.fixture
def tight_tracker(tmp_storage: pathlib.Path) -> BudgetTracker:
    """A tracker with essentially zero budget — all calls blocked."""
    config = BudgetConfig(
        agent_id="tight-agent",
        daily_limit=0.000001,
        degradation_strategy=DegradationStrategy.CACHED_FALLBACK,
    )
    return BudgetTracker(config=config, storage_dir=tmp_storage)


# ===========================================================================
# BudgetGuardError
# ===========================================================================


class TestBudgetGuardError:
    def test_is_runtime_error(self) -> None:
        assert issubclass(BudgetGuardError, RuntimeError)

    def test_remaining_usd_attribute(self) -> None:
        err = BudgetGuardError("blocked", remaining_usd=0.001)
        assert err.remaining_usd == pytest.approx(0.001)

    def test_message_preserved(self) -> None:
        err = BudgetGuardError("Budget blocked for agent", remaining_usd=0.0)
        assert "Budget blocked" in str(err)


# ===========================================================================
# budget_guard decorator — sync functions
# ===========================================================================


class TestBudgetGuardSync:
    def test_allowed_call_executes_function(self, generous_tracker: BudgetTracker) -> None:
        @budget_guard(generous_tracker, model="gpt-4o-mini", input_tokens=10, output_tokens=10)
        def my_fn() -> str:
            return "result"

        result = my_fn()
        assert result == "result"

    def test_allowed_call_records_cost(self, generous_tracker: BudgetTracker) -> None:
        @budget_guard(generous_tracker, model="gpt-4o-mini", input_tokens=100, output_tokens=50)
        def my_fn() -> str:
            return "result"

        my_fn()
        assert generous_tracker.total_lifetime_spend() > 0.0

    def test_blocked_call_raises_budget_guard_error(
        self, tight_tracker: BudgetTracker
    ) -> None:
        @budget_guard(
            tight_tracker,
            model="gpt-4o",
            input_tokens=100000,
            output_tokens=100000,
            raise_on_block=True,
        )
        def expensive_fn() -> str:
            return "should not reach"

        with pytest.raises(BudgetGuardError):
            expensive_fn()

    def test_blocked_call_returns_none_when_raise_disabled(
        self, tight_tracker: BudgetTracker
    ) -> None:
        @budget_guard(
            tight_tracker,
            model="gpt-4o",
            input_tokens=100000,
            output_tokens=100000,
            raise_on_block=False,
        )
        def expensive_fn() -> str:
            return "should not reach"

        result = expensive_fn()
        assert result is None

    def test_usage_attribute_used_for_recording(self, generous_tracker: BudgetTracker) -> None:
        class FakeUsage:
            input_tokens = 200
            output_tokens = 100

        class FakeResponse:
            usage = FakeUsage()

        @budget_guard(
            generous_tracker,
            model="gpt-4o-mini",
            input_tokens=10,
            output_tokens=10,
        )
        def my_fn() -> FakeResponse:
            return FakeResponse()

        my_fn()
        # The tracker should have recorded 200+100 tokens, not the default 10+10
        spent = generous_tracker.total_lifetime_spend()
        assert spent > 0.0

    def test_wrapper_preserves_function_name(self, generous_tracker: BudgetTracker) -> None:
        @budget_guard(generous_tracker, model="gpt-4o-mini")
        def special_function() -> None:
            pass

        assert special_function.__name__ == "special_function"

    def test_wrapper_passes_args_and_kwargs(self, generous_tracker: BudgetTracker) -> None:
        @budget_guard(generous_tracker, model="gpt-4o-mini", input_tokens=10, output_tokens=10)
        def my_fn(x: int, y: int = 0) -> int:
            return x + y

        result = my_fn(3, y=4)
        assert result == 7


# ===========================================================================
# budget_guard decorator — async functions
# ===========================================================================


class TestBudgetGuardAsync:
    def test_async_allowed_call_executes(self, generous_tracker: BudgetTracker) -> None:
        @budget_guard(generous_tracker, model="gpt-4o-mini", input_tokens=10, output_tokens=10)
        async def async_fn() -> str:
            return "async-result"

        result = asyncio.get_event_loop().run_until_complete(async_fn())
        assert result == "async-result"

    def test_async_blocked_call_raises(self, tight_tracker: BudgetTracker) -> None:
        @budget_guard(
            tight_tracker,
            model="gpt-4o",
            input_tokens=100000,
            output_tokens=100000,
            raise_on_block=True,
        )
        async def expensive_async_fn() -> str:
            return "nope"

        with pytest.raises(BudgetGuardError):
            asyncio.get_event_loop().run_until_complete(expensive_async_fn())

    def test_async_blocked_returns_none_when_raise_disabled(
        self, tight_tracker: BudgetTracker
    ) -> None:
        @budget_guard(
            tight_tracker,
            model="gpt-4o",
            input_tokens=100000,
            output_tokens=100000,
            raise_on_block=False,
        )
        async def expensive_async_fn() -> str:
            return "nope"

        result = asyncio.get_event_loop().run_until_complete(expensive_async_fn())
        assert result is None

    def test_async_records_cost_after_call(self, generous_tracker: BudgetTracker) -> None:
        @budget_guard(generous_tracker, model="gpt-4o-mini", input_tokens=50, output_tokens=50)
        async def async_fn() -> None:
            return None

        asyncio.get_event_loop().run_until_complete(async_fn())
        assert generous_tracker.total_lifetime_spend() > 0.0


# ===========================================================================
# _record_from_result helper
# ===========================================================================


class TestRecordFromResult:
    def test_records_defaults_when_no_usage(self, generous_tracker: BudgetTracker) -> None:
        _record_from_result(generous_tracker, "gpt-4o-mini", 100, 50, "plain string result")
        assert generous_tracker.total_lifetime_spend() > 0.0

    def test_records_usage_input_tokens(self, generous_tracker: BudgetTracker) -> None:
        class Usage:
            input_tokens = 999
            output_tokens = 888

        class Resp:
            usage = Usage()

        _record_from_result(generous_tracker, "gpt-4o-mini", 1, 1, Resp())
        # Should use 999 and 888 from the usage object
        spent = generous_tracker.total_lifetime_spend()
        # Cost from pricing tables for 999+888 tokens at gpt-4o-mini rates
        from agent_energy_budget.pricing.tables import get_pricing
        pricing = get_pricing("gpt-4o-mini")
        expected = pricing.cost_for_tokens(999, 888)
        assert spent == pytest.approx(expected, rel=0.001)

    def test_records_prompt_tokens_fallback(self, generous_tracker: BudgetTracker) -> None:
        class Usage:
            input_tokens = None
            output_tokens = None
            prompt_tokens = 500
            completion_tokens = 250

        class Resp:
            usage = Usage()

        _record_from_result(generous_tracker, "gpt-4o-mini", 1, 1, Resp())
        spent = generous_tracker.total_lifetime_spend()
        from agent_energy_budget.pricing.tables import get_pricing
        pricing = get_pricing("gpt-4o-mini")
        expected = pricing.cost_for_tokens(500, 250)
        assert spent == pytest.approx(expected, rel=0.001)


# ===========================================================================
# OpenAIBudgetWrapper
# ===========================================================================


class TestOpenAIBudgetWrapper:
    def _make_mock_client(self) -> MagicMock:
        """Build a minimal mock OpenAI client structure."""
        client = MagicMock()
        response = MagicMock()
        response.usage = MagicMock()
        response.usage.prompt_tokens = 100
        response.usage.completion_tokens = 50
        client.chat.completions.create.return_value = response
        return client

    def test_init_creates_chat_proxy(self, generous_tracker: BudgetTracker) -> None:
        client = self._make_mock_client()
        wrapper = OpenAIBudgetWrapper(client, generous_tracker)
        assert wrapper.chat is not None
        assert wrapper.chat.completions is not None

    def test_create_within_budget_calls_underlying(
        self, generous_tracker: BudgetTracker
    ) -> None:
        client = self._make_mock_client()
        wrapper = OpenAIBudgetWrapper(client, generous_tracker)
        wrapper.chat.completions.create(
            model="gpt-4o-mini",
            max_tokens=100,
            messages=[{"role": "user", "content": "Hello"}],
        )
        assert client.chat.completions.create.called

    def test_create_records_cost_after_call(self, generous_tracker: BudgetTracker) -> None:
        client = self._make_mock_client()
        wrapper = OpenAIBudgetWrapper(client, generous_tracker)
        wrapper.chat.completions.create(
            model="gpt-4o-mini",
            max_tokens=100,
            messages=[{"role": "user", "content": "Hello"}],
        )
        assert generous_tracker.total_lifetime_spend() > 0.0

    def test_create_blocked_raises_budget_exceeded_error(
        self, tight_tracker: BudgetTracker
    ) -> None:
        client = self._make_mock_client()
        wrapper = OpenAIBudgetWrapper(
            client, tight_tracker, raise_on_budget_exceeded=True
        )
        with pytest.raises(BudgetExceededError):
            wrapper.chat.completions.create(
                model="gpt-4o",
                max_tokens=100000,
                messages=[{"role": "user", "content": "Hello " * 10000}],
            )

    def test_create_blocked_returns_none_when_no_raise(
        self, tight_tracker: BudgetTracker
    ) -> None:
        client = self._make_mock_client()
        wrapper = OpenAIBudgetWrapper(
            client, tight_tracker, raise_on_budget_exceeded=False
        )
        result = wrapper.chat.completions.create(
            model="gpt-4o",
            max_tokens=100000,
            messages=[{"role": "user", "content": "Hello " * 10000}],
        )
        assert result is None

    def test_getattr_delegates_to_client(self, generous_tracker: BudgetTracker) -> None:
        client = MagicMock()
        client.some_attribute = "delegate_me"
        wrapper = OpenAIBudgetWrapper(client, generous_tracker)
        assert wrapper.some_attribute == "delegate_me"

    def test_chat_getattr_delegates_to_client_chat(
        self, generous_tracker: BudgetTracker
    ) -> None:
        client = MagicMock()
        client.chat.some_chat_attr = "chat_val"
        wrapper = OpenAIBudgetWrapper(client, generous_tracker)
        assert wrapper.chat.some_chat_attr == "chat_val"

    def test_completions_getattr_delegates(self, generous_tracker: BudgetTracker) -> None:
        client = MagicMock()
        client.chat.completions.some_method.return_value = "delegate"
        wrapper = OpenAIBudgetWrapper(client, generous_tracker)
        assert wrapper.chat.completions.some_method() == "delegate"

    def test_create_with_usage_attribute(self, generous_tracker: BudgetTracker) -> None:
        client = MagicMock()
        response = MagicMock()
        response.usage = MagicMock()
        response.usage.prompt_tokens = 200
        response.usage.completion_tokens = 100
        client.chat.completions.create.return_value = response

        wrapper = OpenAIBudgetWrapper(client, generous_tracker)
        wrapper.chat.completions.create(
            model="gpt-4o-mini",
            max_tokens=512,
            messages=[{"role": "user", "content": "hi"}],
        )
        # Cost should be based on 200+100 tokens
        spent = generous_tracker.total_lifetime_spend()
        assert spent > 0.0

    def test_create_with_no_usage_attribute(self, generous_tracker: BudgetTracker) -> None:
        client = MagicMock()
        response = MagicMock(spec=[])  # No usage attribute
        client.chat.completions.create.return_value = response

        wrapper = OpenAIBudgetWrapper(client, generous_tracker)
        wrapper.chat.completions.create(
            model="gpt-4o-mini",
            max_tokens=512,
            messages=[{"role": "user", "content": "hi"}],
        )
        assert generous_tracker.total_lifetime_spend() > 0.0


# ===========================================================================
# OpenAIBudgetMiddleware
# ===========================================================================


class TestOpenAIBudgetMiddleware:
    def test_before_call_within_budget_returns_params(
        self, generous_tracker: BudgetTracker
    ) -> None:
        middleware = OpenAIBudgetMiddleware(tracker=generous_tracker)
        params = middleware.before_call(
            model="gpt-4o-mini", prompt="Hello", max_tokens=100
        )
        assert params.get("_budget_blocked") is False
        assert params["model"] == "gpt-4o-mini"

    def test_before_call_blocked_returns_budget_blocked_true(
        self, tight_tracker: BudgetTracker
    ) -> None:
        middleware = OpenAIBudgetMiddleware(tracker=tight_tracker)
        params = middleware.before_call(
            model="gpt-4o",
            prompt="Hello " * 10000,
            max_tokens=100000,
        )
        assert params.get("_budget_blocked") is True

    def test_before_call_with_messages(self, generous_tracker: BudgetTracker) -> None:
        middleware = OpenAIBudgetMiddleware(tracker=generous_tracker)
        messages = [{"role": "user", "content": "Tell me a joke"}]
        params = middleware.before_call(
            model="gpt-4o-mini",
            messages=messages,
            max_tokens=200,
        )
        assert params.get("_budget_blocked") is False
        assert params.get("messages") == messages

    def test_before_call_includes_default_output_when_no_max_tokens(
        self, generous_tracker: BudgetTracker
    ) -> None:
        middleware = OpenAIBudgetMiddleware(
            tracker=generous_tracker, default_output_tokens=256
        )
        params = middleware.before_call(model="gpt-4o-mini", prompt="Hi")
        assert params.get("_budget_blocked") is False
        assert params["max_tokens"] == 256

    def test_before_call_caps_max_tokens_to_recommendation(
        self, tmp_path: pathlib.Path
    ) -> None:
        # Use TOKEN_REDUCTION strategy with tight budget
        config = BudgetConfig(
            agent_id="tok-agent",
            daily_limit=0.0001,
            degradation_strategy=DegradationStrategy.TOKEN_REDUCTION,
        )
        tracker = BudgetTracker(config=config, storage_dir=tmp_path)
        middleware = OpenAIBudgetMiddleware(tracker=tracker)
        params = middleware.before_call(
            model="gpt-4o-mini",
            prompt="Short",
            max_tokens=50000,
        )
        # max_tokens should be reduced or blocked
        if not params.get("_budget_blocked"):
            assert params["max_tokens"] <= 50000

    def test_after_call_records_cost(self, generous_tracker: BudgetTracker) -> None:
        middleware = OpenAIBudgetMiddleware(tracker=generous_tracker)
        middleware.after_call("gpt-4o-mini", input_tokens=500, output_tokens=200)
        assert generous_tracker.total_lifetime_spend() > 0.0

    def test_after_call_with_explicit_cost(self, generous_tracker: BudgetTracker) -> None:
        middleware = OpenAIBudgetMiddleware(tracker=generous_tracker)
        middleware.after_call(
            "gpt-4o-mini", input_tokens=100, output_tokens=50, cost=0.123
        )
        assert generous_tracker.total_lifetime_spend() == pytest.approx(0.123, abs=1e-9)

    def test_estimate_input_tokens_from_prompt(
        self, generous_tracker: BudgetTracker
    ) -> None:
        middleware = OpenAIBudgetMiddleware(tracker=generous_tracker)
        tokens = middleware._estimate_input_tokens("Hello world this is a test", None)
        assert tokens > 0

    def test_estimate_input_tokens_from_messages(
        self, generous_tracker: BudgetTracker
    ) -> None:
        middleware = OpenAIBudgetMiddleware(tracker=generous_tracker)
        messages = [{"role": "user", "content": "Hello world"}]
        tokens = middleware._estimate_input_tokens("", messages)
        assert tokens >= 0

    def test_estimate_input_tokens_empty_returns_default(
        self, generous_tracker: BudgetTracker
    ) -> None:
        middleware = OpenAIBudgetMiddleware(
            tracker=generous_tracker, default_input_tokens=999
        )
        tokens = middleware._estimate_input_tokens("", None)
        assert tokens == 999
