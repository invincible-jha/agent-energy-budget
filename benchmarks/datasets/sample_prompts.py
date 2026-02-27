"""Diverse prompt dataset for cost prediction and token counting benchmarks.

All prompts are generated synthetically — no downloads required.
Each prompt has a known approximate character count so token-counting
accuracy can be evaluated against the heuristic and tiktoken backends.

The dataset covers:
- Short prompts (10-50 tokens)
- Medium prompts (100-500 tokens)
- Long prompts (1K-4K tokens)
- Code-heavy prompts (dense, low chars-per-token ratio)
- Natural-language prompts (typical prose)
"""
from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass
class SamplePrompt:
    """A single prompt sample with metadata for benchmark evaluation.

    Parameters
    ----------
    prompt_id:
        Unique identifier.
    text:
        The prompt text.
    category:
        Prompt category: 'short', 'medium', 'long', 'code', 'prose'.
    approx_char_count:
        Character count (exact, for reference).
    expected_token_range:
        Inclusive (min, max) range for reasonable token counts.
        Used to assess whether the counter is within bounds.
    model:
        Model to use for cost estimation benchmarks.
    """

    prompt_id: str
    text: str
    category: str
    approx_char_count: int
    expected_token_range: tuple[int, int]
    model: str = "gpt-4o-mini"


_CODE_SNIPPET = """\
def merge_sort(arr: list[int]) -> list[int]:
    if len(arr) <= 1:
        return arr
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    result = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    result.extend(left[i:])
    result.extend(right[j:])
    return result
"""

_PROSE_PARAGRAPH = (
    "The quick brown fox jumps over the lazy dog. "
    "This sentence has been used for decades as a pangram — "
    "a sentence that contains every letter of the English alphabet at least once. "
    "It is commonly used to display font samples, test keyboards, and benchmark "
    "natural language processing systems that operate on English text. "
)

_SYSTEM_PROMPT = (
    "You are a helpful, harmless, and honest AI assistant. "
    "Your goal is to provide accurate, thoughtful, and well-reasoned responses. "
    "When you are unsure, say so. When asked to do something harmful, refuse politely. "
    "Always cite your sources when making factual claims about the world. "
)


def _repeat_to_length(text: str, target_chars: int) -> str:
    """Repeat text until it reaches approximately target_chars characters."""
    repetitions = (target_chars // len(text)) + 1
    return (text * repetitions)[:target_chars]


def generate_prompt_dataset(seed: int = 42) -> list[SamplePrompt]:
    """Generate a fixed dataset of diverse prompt samples.

    Parameters
    ----------
    seed:
        Random seed for reproducibility.

    Returns
    -------
    list of SamplePrompt
    """
    rng = random.Random(seed)

    prompts: list[SamplePrompt] = []

    # --- Short prompts (10-50 expected tokens) ---
    short_texts = [
        "What is the capital of France?",
        "Summarise this paragraph in one sentence.",
        "Translate 'hello' to Spanish.",
        "What is 2 + 2?",
        "List three programming languages.",
    ]
    for index, text in enumerate(short_texts):
        prompts.append(
            SamplePrompt(
                prompt_id=f"short_{index:03d}",
                text=text,
                category="short",
                approx_char_count=len(text),
                expected_token_range=(5, 25),
                model="gpt-4o-mini",
            )
        )

    # --- Medium prompts (100-500 expected tokens) ---
    medium_base = _PROSE_PARAGRAPH * 3
    for index in range(5):
        text = medium_base + f" Additional context block {index}: " + _SYSTEM_PROMPT
        prompts.append(
            SamplePrompt(
                prompt_id=f"medium_{index:03d}",
                text=text,
                category="medium",
                approx_char_count=len(text),
                expected_token_range=(80, 400),
                model="claude-haiku-4",
            )
        )

    # --- Long prompts (1K-4K expected tokens) ---
    long_base = _repeat_to_length(_PROSE_PARAGRAPH, 8000)
    for index in range(3):
        text = long_base + f"\n\nQuestion {index}: explain the above."
        prompts.append(
            SamplePrompt(
                prompt_id=f"long_{index:03d}",
                text=text,
                category="long",
                approx_char_count=len(text),
                expected_token_range=(1500, 5000),
                model="claude-sonnet-4",
            )
        )

    # --- Code-heavy prompts (lower tokens-per-char ratio due to identifiers) ---
    for index in range(5):
        repeat_factor = rng.randint(3, 8)
        text = _CODE_SNIPPET * repeat_factor + f"\n# Explain function {index}"
        prompts.append(
            SamplePrompt(
                prompt_id=f"code_{index:03d}",
                text=text,
                category="code",
                approx_char_count=len(text),
                expected_token_range=(len(text) // 8, len(text) // 2),
                model="gpt-4o",
            )
        )

    # --- Cached token simulation prompts (shared prefix) ---
    shared_prefix = _SYSTEM_PROMPT * 10
    for index in range(4):
        text = shared_prefix + f"User question {index}: " + short_texts[index % len(short_texts)]
        prompts.append(
            SamplePrompt(
                prompt_id=f"cached_{index:03d}",
                text=text,
                category="cached_simulation",
                approx_char_count=len(text),
                expected_token_range=(len(text) // 6, len(text) // 3),
                model="claude-haiku-4",
            )
        )

    return prompts


__all__ = ["SamplePrompt", "generate_prompt_dataset"]
