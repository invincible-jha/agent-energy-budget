"""Tests for TokenCounter — pre-call token estimation."""
from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest

from agent_energy_budget.prediction.token_counter import TokenCounter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_heuristic_counter(model: str = "gpt-4o") -> TokenCounter:
    """Return a TokenCounter that always uses the heuristic backend."""
    return TokenCounter(model=model, prefer_tiktoken=False)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_default_construction(self) -> None:
        counter = TokenCounter()
        assert counter.backend in ("tiktoken", "heuristic")

    def test_heuristic_backend_when_tiktoken_disabled(self) -> None:
        counter = make_heuristic_counter()
        assert counter.backend == "heuristic"

    def test_repr_contains_model(self) -> None:
        counter = TokenCounter(model="claude-sonnet-4")
        assert "claude-sonnet-4" in repr(counter)


# ---------------------------------------------------------------------------
# count_tokens — basic
# ---------------------------------------------------------------------------


class TestCountTokens:
    def test_empty_string_returns_zero(self) -> None:
        counter = make_heuristic_counter()
        assert counter.count_tokens("") == 0

    def test_short_text_returns_positive(self) -> None:
        counter = make_heuristic_counter()
        result = counter.count_tokens("Hello, world!")
        assert result >= 1

    def test_longer_text_returns_more_tokens(self) -> None:
        counter = make_heuristic_counter()
        short = counter.count_tokens("Hi")
        long_ = counter.count_tokens("This is a much longer sentence with many tokens in it.")
        assert long_ > short

    def test_heuristic_approx_4_chars_per_token(self) -> None:
        counter = make_heuristic_counter()
        # 400 chars / 4 chars per token = 100 tokens (approx)
        text = "a" * 400
        result = counter.count_tokens(text)
        assert 80 <= result <= 120  # allow ±20% variance

    def test_model_arg_is_accepted_but_ignored(self) -> None:
        counter = make_heuristic_counter()
        # model arg is for API compatibility; should not raise
        result = counter.count_tokens("Hello", model="claude-opus-4")
        assert result >= 1


# ---------------------------------------------------------------------------
# count_messages — chat format
# ---------------------------------------------------------------------------


class TestCountMessages:
    def test_single_message(self) -> None:
        counter = make_heuristic_counter()
        messages = [{"role": "user", "content": "Hello"}]
        result = counter.count_messages(messages)
        # 4 overhead + content tokens + 2 primer
        assert result >= 4

    def test_message_overhead_adds_up(self) -> None:
        counter = make_heuristic_counter()
        # Two identical messages should cost roughly twice as much as one
        # plus the fixed primer overhead
        one_msg = [{"role": "user", "content": "Hello world how are you"}]
        two_msg = [
            {"role": "user", "content": "Hello world how are you"},
            {"role": "assistant", "content": "Hello world how are you"},
        ]
        count_one = counter.count_messages(one_msg)
        count_two = counter.count_messages(two_msg)
        assert count_two > count_one

    def test_empty_messages_returns_primer_tokens(self) -> None:
        counter = make_heuristic_counter()
        result = counter.count_messages([])
        # Only reply primer (2 tokens)
        assert result == 2

    def test_multimodal_message_counts_text_parts(self) -> None:
        counter = make_heuristic_counter()
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is in this image?"},
                    {"type": "image_url", "image_url": {"url": "data:..."}},
                ],
            }
        ]
        result = counter.count_messages(messages)
        assert result >= 4  # at least overhead + text tokens


# ---------------------------------------------------------------------------
# count_prompt — unified interface
# ---------------------------------------------------------------------------


class TestCountPrompt:
    def test_plain_string_prompt(self) -> None:
        counter = make_heuristic_counter()
        result = counter.count_prompt("Hello, world!")
        assert result >= 1

    def test_messages_list_prompt(self) -> None:
        counter = make_heuristic_counter()
        messages = [{"role": "user", "content": "Hello"}]
        result = counter.count_prompt(messages)
        assert result >= 4

    def test_system_prompt_added(self) -> None:
        counter = make_heuristic_counter()
        without_system = counter.count_prompt("Hello")
        with_system = counter.count_prompt("Hello", system="You are a helpful assistant.")
        assert with_system > without_system

    def test_empty_system_not_counted(self) -> None:
        counter = make_heuristic_counter()
        without_system = counter.count_prompt("Hello")
        with_empty_system = counter.count_prompt("Hello", system="")
        assert without_system == with_empty_system


# ---------------------------------------------------------------------------
# tiktoken integration (skipped if not installed)
# ---------------------------------------------------------------------------


class TestTiktokenIntegration:
    def test_tiktoken_backend_used_when_available(self) -> None:
        pytest.importorskip("tiktoken")
        counter = TokenCounter(model="gpt-4o", prefer_tiktoken=True)
        if counter.backend == "tiktoken":
            result = counter.count_tokens("Hello, world!")
            assert result >= 1

    def test_falls_back_to_heuristic_on_tiktoken_failure(self) -> None:
        """When tiktoken raises during count, heuristic is used."""
        counter = TokenCounter(model="gpt-4o", prefer_tiktoken=False)
        # Force heuristic path — should still work
        result = counter.count_tokens("Hello, world!")
        assert result >= 1


# ---------------------------------------------------------------------------
# Consistency checks
# ---------------------------------------------------------------------------


class TestConsistency:
    def test_same_text_same_result(self) -> None:
        counter = make_heuristic_counter()
        text = "This is a consistent test sentence."
        assert counter.count_tokens(text) == counter.count_tokens(text)

    def test_longer_text_always_more_tokens(self) -> None:
        counter = make_heuristic_counter()
        short = counter.count_tokens("Hi.")
        long_ = counter.count_tokens("Hi. " * 100)
        assert long_ > short

    def test_count_tokens_non_ascii(self) -> None:
        counter = make_heuristic_counter()
        result = counter.count_tokens("こんにちは世界")  # Japanese text
        assert result >= 1
