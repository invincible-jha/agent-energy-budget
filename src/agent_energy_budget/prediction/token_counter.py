"""TokenCounter — pre-call token estimation for cost prediction.

Provides token count estimates before making LLM API calls. Uses tiktoken
when available for accurate counts; falls back to a character-based
heuristic (~4 characters per token for English text) otherwise.

This counter is distinct from ``pricing.token_counter.TokenCounter`` in
that it also handles multi-modal message lists and system prompts in a
single interface tailored to the prediction workflow.
"""
from __future__ import annotations

import re
from typing import Union


# ---------------------------------------------------------------------------
# Heuristic constants
# ---------------------------------------------------------------------------

# English text averages ~4 characters per token (GPT-3/4 tokenizer research)
_CHARS_PER_TOKEN: float = 4.0

# Per-message overhead in the OpenAI chat format (role + separator tokens)
_MESSAGE_OVERHEAD_TOKENS: int = 4

# Reply primer overhead added to every chat completion request
_REPLY_PRIMER_TOKENS: int = 2

# Whitespace token splitter (word-level fallback)
_WORD_RE = re.compile(r"\S+")
# Empirically: 1 word ≈ 1.33 tokens
_WORDS_PER_TOKEN: float = 0.75


class TokenCounter:
    """Estimate token counts without an API call.

    Parameters
    ----------
    model:
        The target model. Used to select the right tiktoken encoding when
        available (e.g. "gpt-4o" → "cl100k_base"). Falls back to heuristic.
    prefer_tiktoken:
        When False, skips tiktoken even if installed. Useful in tests.

    Examples
    --------
    >>> counter = TokenCounter(model="gpt-4o")
    >>> counter.count_tokens("Hello, world!")
    4
    """

    def __init__(
        self,
        model: str = "gpt-4o",
        prefer_tiktoken: bool = True,
    ) -> None:
        self._model = model
        self._tiktoken_encoding: object = None
        self._tiktoken_available: bool = False
        if prefer_tiktoken:
            self._tiktoken_available = self._try_load_tiktoken()

    # ------------------------------------------------------------------
    # Internal: tiktoken
    # ------------------------------------------------------------------

    def _try_load_tiktoken(self) -> bool:
        """Attempt to load tiktoken for the configured model."""
        try:
            import tiktoken  # type: ignore[import-untyped]

            try:
                self._tiktoken_encoding = tiktoken.encoding_for_model(self._model)
            except KeyError:
                # Fall back to cl100k_base for unknown models
                self._tiktoken_encoding = tiktoken.get_encoding("cl100k_base")
            return True
        except Exception:
            return False

    def _count_with_tiktoken(self, text: str) -> int:
        if self._tiktoken_encoding is None:
            return self._count_heuristic(text)
        tokens: list[int] = self._tiktoken_encoding.encode(text)
        return len(tokens)

    # ------------------------------------------------------------------
    # Internal: heuristics
    # ------------------------------------------------------------------

    def _count_heuristic(self, text: str) -> int:
        """Character-ratio heuristic: ~4 chars/token for English."""
        if not text:
            return 0
        return max(1, round(len(text) / _CHARS_PER_TOKEN))

    def _count_heuristic_words(self, text: str) -> int:
        """Word-based fallback: ~0.75 words/token."""
        words = _WORD_RE.findall(text)
        if not words:
            return 0
        return max(1, round(len(words) / _WORDS_PER_TOKEN))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def backend(self) -> str:
        """Return the active token counting backend."""
        return "tiktoken" if self._tiktoken_available else "heuristic"

    def count_tokens(self, text: str, model: str = "") -> int:
        """Estimate token count for *text*.

        Parameters
        ----------
        text:
            Raw string to count. Can be plain text, a stringified message
            list, or any other textual content.
        model:
            Ignored in this implementation (encoding is set at
            construction time). Included for API compatibility.

        Returns
        -------
        int
            Estimated token count (0 for empty input).
        """
        if not text:
            return 0
        if self._tiktoken_available:
            try:
                return self._count_with_tiktoken(text)
            except Exception:
                pass
        return self._count_heuristic(text)

    def count_messages(self, messages: list[dict[str, object]]) -> int:
        """Count tokens for a list of chat-format message dicts.

        Applies per-message overhead following the OpenAI convention:
        - 4 tokens per message (role + separators)
        - 2 tokens reply primer added at the end

        Parameters
        ----------
        messages:
            List of dicts with at least a "content" key. The "role" value
            is also counted when present.

        Returns
        -------
        int
            Total estimated input token count.
        """
        total = 0
        for msg in messages:
            total += _MESSAGE_OVERHEAD_TOKENS
            for key, value in msg.items():
                if isinstance(value, str):
                    total += self.count_tokens(value)
                elif isinstance(value, list):
                    # Multi-modal: count text parts only
                    for part in value:
                        if isinstance(part, dict) and "text" in part:
                            total += self.count_tokens(str(part["text"]))
                else:
                    total += self.count_tokens(str(value))
        total += _REPLY_PRIMER_TOKENS
        return total

    def count_prompt(
        self,
        prompt: Union[str, list[dict[str, object]]],
        system: str = "",
    ) -> int:
        """Count tokens for a prompt that may be a string or messages list.

        Parameters
        ----------
        prompt:
            Either a plain string or a list of chat message dicts.
        system:
            Optional system prompt (counted separately with overhead).

        Returns
        -------
        int
            Total estimated input token count including system prompt.
        """
        system_tokens = self.count_tokens(system) if system else 0

        if isinstance(prompt, str):
            prompt_tokens = self.count_tokens(prompt)
        else:
            prompt_tokens = self.count_messages(prompt)

        return system_tokens + prompt_tokens

    def __repr__(self) -> str:
        return f"TokenCounter(model={self._model!r}, backend={self.backend!r})"
