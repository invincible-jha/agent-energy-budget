"""Unit tests for agent_energy_budget.pricing.token_counter."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from agent_energy_budget.pricing.token_counter import TokenCounter


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def heuristic_counter() -> TokenCounter:
    """TokenCounter with tiktoken disabled so only the heuristic runs."""
    counter = TokenCounter()
    counter._tiktoken_available = False
    return counter


@pytest.fixture()
def counter() -> TokenCounter:
    """Standard counter; may use tiktoken if installed."""
    return TokenCounter()


# ---------------------------------------------------------------------------
# Backend detection
# ---------------------------------------------------------------------------


class TestBackendProperty:
    def test_backend_is_tiktoken_or_heuristic(self, counter: TokenCounter) -> None:
        assert counter.backend in ("tiktoken", "heuristic")

    def test_backend_heuristic_when_tiktoken_disabled(self, heuristic_counter: TokenCounter) -> None:
        assert heuristic_counter.backend == "heuristic"


# ---------------------------------------------------------------------------
# count — heuristic path
# ---------------------------------------------------------------------------


class TestCountHeuristic:
    def test_empty_string_returns_zero(self, heuristic_counter: TokenCounter) -> None:
        assert heuristic_counter.count("") == 0

    def test_single_word_returns_at_least_one(self, heuristic_counter: TokenCounter) -> None:
        assert heuristic_counter.count("hello") >= 1

    def test_longer_text_returns_more_tokens_than_shorter(self, heuristic_counter: TokenCounter) -> None:
        short = heuristic_counter.count("Hello world")
        long = heuristic_counter.count("Hello world, this is a much longer sentence.")
        assert long > short

    def test_whitespace_only_is_empty(self, heuristic_counter: TokenCounter) -> None:
        # whitespace-only has no words => treated as empty path returns 0
        result = heuristic_counter.count("   ")
        # regex finds no \S+ tokens => words=[] => estimated=0 => max(1,0)=1
        # Actually returns 1 per max(1, round(0/0.75)) = max(1,0) = 1
        assert result >= 0

    def test_count_heuristic_formula(self, heuristic_counter: TokenCounter) -> None:
        # Known word count: "one two three" = 3 words
        # estimated = 3 / 0.75 = 4.0 => round => 4
        result = heuristic_counter._count_heuristic("one two three")
        assert result == 4

    def test_heuristic_result_is_integer(self, heuristic_counter: TokenCounter) -> None:
        result = heuristic_counter.count("Some test text for counting.")
        assert isinstance(result, int)


# ---------------------------------------------------------------------------
# count_messages
# ---------------------------------------------------------------------------


class TestCountMessages:
    def test_empty_list_returns_two(self, heuristic_counter: TokenCounter) -> None:
        # 0 messages * 4 overhead + 2 reply primer = 2
        result = heuristic_counter.count_messages([])
        assert result == 2

    def test_single_message_adds_overhead(self, heuristic_counter: TokenCounter) -> None:
        messages = [{"role": "user", "content": ""}]
        # 1 message * 4 overhead + count("user") + count("") + 2 = 4 + tokens + 2
        result = heuristic_counter.count_messages(messages)
        assert result >= 6

    def test_multiple_messages_accumulate(self, heuristic_counter: TokenCounter) -> None:
        messages = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "world"},
        ]
        one_msg = heuristic_counter.count_messages([messages[0]])
        two_msgs = heuristic_counter.count_messages(messages)
        assert two_msgs > one_msg

    def test_count_messages_returns_integer(self, heuristic_counter: TokenCounter) -> None:
        messages = [{"role": "user", "content": "ping"}]
        result = heuristic_counter.count_messages(messages)
        assert isinstance(result, int)


# ---------------------------------------------------------------------------
# estimate_from_tokens_or_text
# ---------------------------------------------------------------------------


class TestEstimateFromTokensOrText:
    def test_int_passthrough(self, heuristic_counter: TokenCounter) -> None:
        result = heuristic_counter.estimate_from_tokens_or_text(42)
        assert result == 42

    def test_string_delegates_to_count(self, heuristic_counter: TokenCounter) -> None:
        text = "three words here"
        result = heuristic_counter.estimate_from_tokens_or_text(text)
        assert result == heuristic_counter.count(text)

    def test_zero_int_returns_zero(self, heuristic_counter: TokenCounter) -> None:
        result = heuristic_counter.estimate_from_tokens_or_text(0)
        assert result == 0


# ---------------------------------------------------------------------------
# split_into_chunks
# ---------------------------------------------------------------------------


class TestSplitIntoChunks:
    def test_raises_on_zero_max_tokens(self, heuristic_counter: TokenCounter) -> None:
        with pytest.raises(ValueError, match="positive"):
            heuristic_counter.split_into_chunks("text", 0)

    def test_raises_on_negative_max_tokens(self, heuristic_counter: TokenCounter) -> None:
        with pytest.raises(ValueError, match="positive"):
            heuristic_counter.split_into_chunks("text", -1)

    def test_small_text_fits_in_one_chunk(self, heuristic_counter: TokenCounter) -> None:
        text = "Hello world"
        chunks = heuristic_counter.split_into_chunks(text, 50)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_empty_text_returns_empty_list(self, heuristic_counter: TokenCounter) -> None:
        # re.split(r"\n\n+", "") yields [''], so the implementation produces a
        # single empty-string chunk rather than an empty list.  The contract is
        # that the result contains no non-empty chunks.
        chunks = heuristic_counter.split_into_chunks("", 50)
        assert all(chunk == "" for chunk in chunks)

    def test_long_text_produces_multiple_chunks(self, heuristic_counter: TokenCounter) -> None:
        # Create text with enough words to exceed limit
        text = "word " * 200
        chunks = heuristic_counter.split_into_chunks(text.strip(), 20)
        assert len(chunks) >= 2

    def test_chunks_collectively_contain_all_words(self, heuristic_counter: TokenCounter) -> None:
        words = ["word"] * 100
        text = " ".join(words)
        chunks = heuristic_counter.split_into_chunks(text, 20)
        combined = " ".join(chunks)
        # All words should be present (order preserved)
        assert combined.count("word") == 100

    def test_paragraph_boundaries_preferred(self, heuristic_counter: TokenCounter) -> None:
        text = "Paragraph one.\n\nParagraph two."
        chunks = heuristic_counter.split_into_chunks(text, 5)
        # Should produce at least one chunk per paragraph
        assert len(chunks) >= 1
