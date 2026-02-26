"""Token counting utilities.

Uses tiktoken (cl100k_base encoding) when available for accurate counts.
Falls back to a word-based heuristic (~0.75 tokens per word) otherwise.
"""
from __future__ import annotations

import re
from typing import Union


class TokenCounter:
    """Estimate token counts from raw text.

    Parameters
    ----------
    encoding_name:
        tiktoken encoding to use when tiktoken is installed.
        Defaults to "cl100k_base" which covers GPT-4 and Claude models.

    Examples
    --------
    >>> counter = TokenCounter()
    >>> counter.count("Hello, world!")
    4
    """

    _WORD_RE = re.compile(r"\S+")
    # Empirically derived ratio — 1 word ≈ 1.33 tokens (i.e. 0.75 words/token)
    _WORDS_PER_TOKEN: float = 0.75

    def __init__(self, encoding_name: str = "cl100k_base") -> None:
        self._encoding_name = encoding_name
        self._tiktoken_encoding: object | None = None
        self._tiktoken_available: bool = self._try_load_tiktoken()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _try_load_tiktoken(self) -> bool:
        """Attempt to load tiktoken; return True on success."""
        try:
            import tiktoken  # type: ignore[import-untyped]

            self._tiktoken_encoding = tiktoken.get_encoding(self._encoding_name)
            return True
        except Exception:
            return False

    def _count_with_tiktoken(self, text: str) -> int:
        """Count tokens using the loaded tiktoken encoding."""
        import tiktoken  # type: ignore[import-untyped]

        enc = self._tiktoken_encoding
        if enc is None:
            raise RuntimeError("tiktoken encoding not loaded")
        tokens: list[int] = enc.encode(text)
        return len(tokens)

    def _count_heuristic(self, text: str) -> int:
        """Estimate token count using word-based heuristic.

        Splits on whitespace and applies the empirical ratio
        words * (1 / 0.75) ≈ words * 1.333.
        """
        words = self._WORD_RE.findall(text)
        estimated = len(words) / self._WORDS_PER_TOKEN
        return max(1, round(estimated))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def backend(self) -> str:
        """Return the active counting backend ('tiktoken' or 'heuristic')."""
        return "tiktoken" if self._tiktoken_available else "heuristic"

    def count(self, text: str) -> int:
        """Return the estimated token count for *text*.

        Parameters
        ----------
        text:
            Raw text string to count.

        Returns
        -------
        int
            Token count estimate (>= 1 for non-empty strings, 0 for empty).
        """
        if not text:
            return 0
        if self._tiktoken_available:
            try:
                return self._count_with_tiktoken(text)
            except Exception:
                pass
        return self._count_heuristic(text)

    def count_messages(self, messages: list[dict[str, str]]) -> int:
        """Count tokens across a list of role/content message dicts.

        Adds 4 tokens per message for role/separator overhead (OpenAI
        convention) and 2 tokens for the reply primer.

        Parameters
        ----------
        messages:
            List of dicts with at least a "content" key.

        Returns
        -------
        int
            Estimated total token count.
        """
        total = 0
        for message in messages:
            total += 4  # per-message overhead
            for value in message.values():
                total += self.count(str(value))
        total += 2  # reply primer
        return total

    def estimate_from_tokens_or_text(
        self, tokens_or_text: Union[int, str]
    ) -> int:
        """Resolve an input that is either already a token count or raw text.

        Parameters
        ----------
        tokens_or_text:
            When an int, it is returned as-is.
            When a str, it is passed through :meth:`count`.

        Returns
        -------
        int
            Token count.
        """
        if isinstance(tokens_or_text, int):
            return tokens_or_text
        return self.count(tokens_or_text)

    def split_into_chunks(
        self, text: str, max_tokens_per_chunk: int
    ) -> list[str]:
        """Split *text* into chunks that each fit within *max_tokens_per_chunk*.

        Uses sentence/paragraph boundaries where possible.

        Parameters
        ----------
        text:
            Input text to split.
        max_tokens_per_chunk:
            Maximum tokens allowed per chunk.

        Returns
        -------
        list[str]
            List of text chunks.
        """
        if max_tokens_per_chunk <= 0:
            raise ValueError("max_tokens_per_chunk must be positive")

        # Split on paragraph boundaries first
        paragraphs = re.split(r"\n\n+", text)
        chunks: list[str] = []
        current_parts: list[str] = []
        current_tokens = 0

        for para in paragraphs:
            para_tokens = self.count(para)
            if current_tokens + para_tokens > max_tokens_per_chunk:
                if current_parts:
                    chunks.append("\n\n".join(current_parts))
                    current_parts = []
                    current_tokens = 0
                # If single paragraph exceeds limit, force split by words
                if para_tokens > max_tokens_per_chunk:
                    words = para.split()
                    word_chunk: list[str] = []
                    word_tokens = 0
                    for word in words:
                        wt = self.count(word)
                        if word_tokens + wt > max_tokens_per_chunk and word_chunk:
                            chunks.append(" ".join(word_chunk))
                            word_chunk = []
                            word_tokens = 0
                        word_chunk.append(word)
                        word_tokens += wt
                    if word_chunk:
                        current_parts = [" ".join(word_chunk)]
                        current_tokens = word_tokens
                    continue
            current_parts.append(para)
            current_tokens += para_tokens

        if current_parts:
            chunks.append("\n\n".join(current_parts))

        return chunks
