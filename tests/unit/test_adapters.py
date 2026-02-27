"""Unit tests for agent_energy_budget framework adapters."""
from __future__ import annotations

import pytest

from agent_energy_budget.adapters import (
    AnthropicCostTracker,
    CrewAICostTracker,
    LangChainCostTracker,
    MicrosoftCostTracker,
    OpenAICostTracker,
)


# ---------------------------------------------------------------------------
# LangChainCostTracker
# ---------------------------------------------------------------------------


class TestLangChainCostTracker:
    def test_construction_no_args(self) -> None:
        tracker = LangChainCostTracker()
        assert tracker.total_tokens == 0
        assert tracker.total_cost_usd == 0.0

    def test_construction_with_pricing(self) -> None:
        pricing = {"gpt-4o": 0.00001}
        tracker = LangChainCostTracker(model_pricing=pricing)
        assert tracker.model_pricing == pricing

    def test_on_llm_start_returns_none(self) -> None:
        tracker = LangChainCostTracker()
        result = tracker.on_llm_start("gpt-4o")
        assert result is None

    def test_on_llm_end_returns_dict(self) -> None:
        tracker = LangChainCostTracker()
        result = tracker.on_llm_end("gpt-4o", 100)
        assert isinstance(result, dict)
        assert "tokens" in result
        assert "cost_usd" in result

    def test_on_llm_end_accumulates_tokens(self) -> None:
        tracker = LangChainCostTracker()
        tracker.on_llm_end("gpt-4o", 100)
        tracker.on_llm_end("gpt-4o", 50)
        assert tracker.total_tokens == 150

    def test_get_total_cost_returns_float(self) -> None:
        tracker = LangChainCostTracker()
        tracker.on_llm_end("gpt-4o", 500)
        assert isinstance(tracker.get_total_cost(), float)
        assert tracker.get_total_cost() > 0.0

    def test_get_token_usage_returns_dict(self) -> None:
        tracker = LangChainCostTracker()
        tracker.on_llm_end("gpt-4o", 200)
        usage = tracker.get_token_usage()
        assert isinstance(usage, dict)
        assert "total_tokens" in usage
        assert usage["total_tokens"] == 200

    def test_reset_clears_state(self) -> None:
        tracker = LangChainCostTracker()
        tracker.on_llm_end("gpt-4o", 100)
        tracker.reset()
        assert tracker.total_tokens == 0
        assert tracker.total_cost_usd == 0.0
        assert tracker.get_token_usage()["total_tokens"] == 0


# ---------------------------------------------------------------------------
# CrewAICostTracker
# ---------------------------------------------------------------------------


class TestCrewAICostTracker:
    def test_construction_no_args(self) -> None:
        tracker = CrewAICostTracker()
        assert tracker.total_tokens == 0
        assert tracker.total_cost_usd == 0.0

    def test_construction_with_pricing(self) -> None:
        pricing = {"gpt-4o": 0.00001}
        tracker = CrewAICostTracker(model_pricing=pricing)
        assert tracker.model_pricing == pricing

    def test_on_task_start_returns_none(self) -> None:
        tracker = CrewAICostTracker()
        result = tracker.on_task_start("analyse")
        assert result is None

    def test_on_task_end_returns_dict(self) -> None:
        tracker = CrewAICostTracker()
        result = tracker.on_task_end("analyse", 200)
        assert isinstance(result, dict)
        assert result["task_name"] == "analyse"
        assert result["tokens"] == 200

    def test_get_crew_cost_returns_float(self) -> None:
        tracker = CrewAICostTracker()
        tracker.on_task_end("t1", 100)
        assert isinstance(tracker.get_crew_cost(), float)

    def test_get_agent_costs_returns_dict(self) -> None:
        tracker = CrewAICostTracker()
        tracker.on_task_end("t1", 100)
        costs = tracker.get_agent_costs()
        assert isinstance(costs, dict)
        assert "task_costs" in costs

    def test_reset_clears_state(self) -> None:
        tracker = CrewAICostTracker()
        tracker.on_task_end("t1", 300)
        tracker.reset()
        assert tracker.total_tokens == 0
        assert tracker.total_cost_usd == 0.0
        assert tracker.get_crew_cost() == 0.0

    def test_multiple_tasks_accumulate(self) -> None:
        tracker = CrewAICostTracker()
        tracker.on_task_end("t1", 100)
        tracker.on_task_end("t2", 200)
        assert tracker.total_tokens == 300


# ---------------------------------------------------------------------------
# OpenAICostTracker
# ---------------------------------------------------------------------------


class TestOpenAICostTracker:
    def test_construction_no_args(self) -> None:
        tracker = OpenAICostTracker()
        assert tracker.total_tokens == 0
        assert tracker.total_cost_usd == 0.0

    def test_construction_with_pricing(self) -> None:
        pricing = {"gpt-4o": 0.00001}
        tracker = OpenAICostTracker(model_pricing=pricing)
        assert tracker.model_pricing == pricing

    def test_on_completion_returns_dict(self) -> None:
        tracker = OpenAICostTracker()
        result = tracker.on_completion("gpt-4o", 150)
        assert isinstance(result, dict)
        assert result["model"] == "gpt-4o"
        assert result["tokens"] == 150

    def test_on_completion_uses_pricing(self) -> None:
        tracker = OpenAICostTracker(model_pricing={"gpt-4o": 0.000010})
        result = tracker.on_completion("gpt-4o", 100)
        assert abs(result["cost_usd"] - 0.001) < 1e-9

    def test_on_tool_call_returns_none(self) -> None:
        tracker = OpenAICostTracker()
        result = tracker.on_tool_call("web_search")
        assert result is None

    def test_on_tool_call_increments_counter(self) -> None:
        tracker = OpenAICostTracker()
        tracker.on_tool_call("search")
        tracker.on_tool_call("calculator")
        assert tracker._tool_calls == 2

    def test_get_session_cost_returns_float(self) -> None:
        tracker = OpenAICostTracker()
        tracker.on_completion("gpt-4o", 500)
        assert isinstance(tracker.get_session_cost(), float)
        assert tracker.get_session_cost() > 0.0

    def test_reset_clears_state(self) -> None:
        tracker = OpenAICostTracker()
        tracker.on_completion("gpt-4o", 500)
        tracker.on_tool_call("search")
        tracker.reset()
        assert tracker.total_tokens == 0
        assert tracker.total_cost_usd == 0.0
        assert tracker._tool_calls == 0


# ---------------------------------------------------------------------------
# AnthropicCostTracker
# ---------------------------------------------------------------------------


class TestAnthropicCostTracker:
    def test_construction_no_args(self) -> None:
        tracker = AnthropicCostTracker()
        assert tracker.total_tokens == 0
        assert tracker.total_cost_usd == 0.0

    def test_construction_with_pricing(self) -> None:
        pricing = {"claude-sonnet-4-5": 0.000003}
        tracker = AnthropicCostTracker(model_pricing=pricing)
        assert tracker.model_pricing == pricing

    def test_on_message_returns_dict(self) -> None:
        tracker = AnthropicCostTracker()
        result = tracker.on_message("claude-sonnet-4-5", 100, 50)
        assert isinstance(result, dict)
        assert result["input_tokens"] == 100
        assert result["output_tokens"] == 50

    def test_on_message_accumulates_tokens(self) -> None:
        tracker = AnthropicCostTracker()
        tracker.on_message("claude-sonnet-4-5", 100, 50)
        tracker.on_message("claude-sonnet-4-5", 200, 100)
        assert tracker.total_tokens == 450

    def test_on_tool_use_returns_none(self) -> None:
        tracker = AnthropicCostTracker()
        result = tracker.on_tool_use("calculator")
        assert result is None

    def test_get_session_cost_returns_float(self) -> None:
        tracker = AnthropicCostTracker()
        tracker.on_message("claude-sonnet-4-5", 100, 50)
        assert isinstance(tracker.get_session_cost(), float)
        assert tracker.get_session_cost() > 0.0

    def test_reset_clears_state(self) -> None:
        tracker = AnthropicCostTracker()
        tracker.on_message("claude-sonnet-4-5", 100, 50)
        tracker.on_tool_use("search")
        tracker.reset()
        assert tracker.total_tokens == 0
        assert tracker.total_cost_usd == 0.0
        assert tracker._tool_calls == 0


# ---------------------------------------------------------------------------
# MicrosoftCostTracker
# ---------------------------------------------------------------------------


class TestMicrosoftCostTracker:
    def test_construction_no_args(self) -> None:
        tracker = MicrosoftCostTracker()
        assert tracker.total_tokens == 0
        assert tracker.total_cost_usd == 0.0

    def test_construction_with_pricing(self) -> None:
        pricing = {"gpt-4o": 0.000005}
        tracker = MicrosoftCostTracker(model_pricing=pricing)
        assert tracker.model_pricing == pricing

    def test_on_turn_returns_dict(self) -> None:
        tracker = MicrosoftCostTracker()
        result = tracker.on_turn("gpt-4o", 200)
        assert isinstance(result, dict)
        assert result["model"] == "gpt-4o"
        assert result["tokens"] == 200

    def test_on_turn_accumulates_tokens(self) -> None:
        tracker = MicrosoftCostTracker()
        tracker.on_turn("gpt-4o", 100)
        tracker.on_turn("gpt-4o", 50)
        assert tracker.total_tokens == 150

    def test_on_activity_returns_none(self) -> None:
        tracker = MicrosoftCostTracker()
        result = tracker.on_activity("message")
        assert result is None

    def test_on_activity_increments_counter(self) -> None:
        tracker = MicrosoftCostTracker()
        tracker.on_activity("message")
        tracker.on_activity("typing")
        assert tracker._activity_count == 2

    def test_get_conversation_cost_returns_float(self) -> None:
        tracker = MicrosoftCostTracker()
        tracker.on_turn("gpt-4o", 300)
        assert isinstance(tracker.get_conversation_cost(), float)
        assert tracker.get_conversation_cost() > 0.0

    def test_reset_clears_state(self) -> None:
        tracker = MicrosoftCostTracker()
        tracker.on_turn("gpt-4o", 300)
        tracker.on_activity("message")
        tracker.reset()
        assert tracker.total_tokens == 0
        assert tracker.total_cost_usd == 0.0
        assert tracker._activity_count == 0

    def test_all_classes_importable_from_init(self) -> None:
        from agent_energy_budget.adapters import (
            AnthropicCostTracker,
            CrewAICostTracker,
            LangChainCostTracker,
            MicrosoftCostTracker,
            OpenAICostTracker,
        )
        assert LangChainCostTracker is not None
        assert CrewAICostTracker is not None
        assert OpenAICostTracker is not None
        assert AnthropicCostTracker is not None
        assert MicrosoftCostTracker is not None
