"""Task complexity classifier for adaptive model routing (E10.2).

TaskClassifier classifies task prompts into complexity buckets:
- "simple"  — fewer than 100 tokens (approximate) and no complexity keywords
- "medium"  — 100–500 tokens (approximate) or contains moderate-complexity signals
- "complex" — more than 500 tokens (approximate) or contains high-complexity keywords

Classification uses token count approximation (word-based, ~1.3 words per token)
plus keyword heuristics. This is deliberately a commodity algorithm — it avoids
any proprietary ML models or learned weights.

Example
-------
::

    classifier = TaskClassifier()
    result = classifier.classify("What is 2+2?")
    assert result.task_type == "simple"

    result = classifier.classify("Analyse the geopolitical implications of...")
    assert result.task_type == "complex"
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

TaskType = Literal["simple", "medium", "complex"]

# ---------------------------------------------------------------------------
# Token count approximation constants
# ---------------------------------------------------------------------------

# Words-per-token approximation for English text (rough estimate)
_WORDS_PER_TOKEN: float = 0.75  # ~1 token per 0.75 words

_SIMPLE_TOKEN_THRESHOLD: int = 100    # < 100 tokens → candidate for simple
_COMPLEX_TOKEN_THRESHOLD: int = 500   # >= 500 tokens → always complex

# ---------------------------------------------------------------------------
# Keyword tables
# ---------------------------------------------------------------------------

# Keywords that strongly indicate HIGH complexity regardless of length
_COMPLEX_KEYWORDS: frozenset[str] = frozenset(
    [
        "analyse",
        "analyze",
        "analysis",
        "elaborate",
        "comprehensive",
        "in-depth",
        "in depth",
        "compare and contrast",
        "compare",
        "contrast",
        "evaluate",
        "evaluation",
        "assess",
        "assessment",
        "synthesise",
        "synthesize",
        "research",
        "detailed",
        "explain in detail",
        "step by step",
        "step-by-step",
        "walk me through",
        "deep dive",
        "critique",
        "critically",
        "implications",
        "implications of",
        "geopolitical",
        "architecture",
        "design document",
        "technical specification",
        "long-form",
        "essay",
        "dissertation",
    ]
)

# Keywords indicating MEDIUM complexity (upgrade from simple if token count allows)
_MEDIUM_KEYWORDS: frozenset[str] = frozenset(
    [
        "explain",
        "describe",
        "how does",
        "how do",
        "summarise",
        "summarize",
        "summary",
        "overview",
        "difference between",
        "pros and cons",
        "advantages",
        "disadvantages",
        "list",
        "enumerate",
        "outline",
        "review",
        "discuss",
    ]
)

# Patterns indicating simple factual queries (override keywords if present)
_SIMPLE_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"^\s*what is\s+", re.IGNORECASE),
    re.compile(r"^\s*who is\s+", re.IGNORECASE),
    re.compile(r"^\s*when is\s+", re.IGNORECASE),
    re.compile(r"^\s*where is\s+", re.IGNORECASE),
    re.compile(r"^\s*define\s+", re.IGNORECASE),
    re.compile(r"^\s*translate\s+", re.IGNORECASE),
]


# ---------------------------------------------------------------------------
# ClassificationResult
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ClassificationResult:
    """Result of a task classification.

    Attributes
    ----------
    task_type:
        The classified complexity tier: ``"simple"``, ``"medium"``, or ``"complex"``.
    estimated_tokens:
        Approximate token count (based on word count approximation).
    detected_keywords:
        Keywords detected in the prompt that influenced classification.
    is_factual_query:
        True if the prompt was detected as a simple factual query.
    confidence:
        Informal confidence label: ``"high"`` when classification is clear,
        ``"medium"`` when it relies on keyword signals over length signals.
    """

    task_type: TaskType
    estimated_tokens: int
    detected_keywords: tuple[str, ...]
    is_factual_query: bool
    confidence: Literal["high", "medium", "low"]


# ---------------------------------------------------------------------------
# TaskClassifier
# ---------------------------------------------------------------------------


class TaskClassifier:
    """Classifies task prompts into complexity buckets for routing.

    Parameters
    ----------
    simple_token_threshold:
        Maximum approximate token count for a "simple" task.
        Prompts at or below this threshold are simple unless keyword signals
        indicate higher complexity.
    complex_token_threshold:
        Minimum approximate token count for a "complex" task.
        Prompts at or above this threshold are always complex.
    words_per_token:
        Words-per-token ratio for token estimation. Default ~0.75.

    Examples
    --------
    ::

        classifier = TaskClassifier()
        result = classifier.classify("What is the capital of France?")
        assert result.task_type == "simple"
        assert result.is_factual_query is True
    """

    def __init__(
        self,
        simple_token_threshold: int = _SIMPLE_TOKEN_THRESHOLD,
        complex_token_threshold: int = _COMPLEX_TOKEN_THRESHOLD,
        words_per_token: float = _WORDS_PER_TOKEN,
    ) -> None:
        if simple_token_threshold <= 0:
            raise ValueError(
                f"simple_token_threshold must be > 0; got {simple_token_threshold!r}."
            )
        if complex_token_threshold <= simple_token_threshold:
            raise ValueError(
                f"complex_token_threshold ({complex_token_threshold}) must be > "
                f"simple_token_threshold ({simple_token_threshold})."
            )
        if words_per_token <= 0.0:
            raise ValueError(
                f"words_per_token must be > 0; got {words_per_token!r}."
            )
        self._simple_threshold = simple_token_threshold
        self._complex_threshold = complex_token_threshold
        self._words_per_token = words_per_token

    def estimate_tokens(self, text: str) -> int:
        """Estimate token count from a text string.

        Uses a word-count approximation divided by the words_per_token ratio.
        This is not an exact tokenizer count — it is a fast heuristic suitable
        for routing decisions.

        Parameters
        ----------
        text:
            The input text to estimate tokens for.

        Returns
        -------
        int
            Estimated token count (always >= 0).
        """
        word_count = len(text.split())
        return max(0, round(word_count / self._words_per_token))

    def classify(self, prompt: str) -> ClassificationResult:
        """Classify a prompt into a complexity task type.

        Parameters
        ----------
        prompt:
            The raw prompt text to classify.

        Returns
        -------
        ClassificationResult
            Contains the task_type and supporting metadata.
        """
        stripped = prompt.strip()
        lower_prompt = stripped.lower()
        estimated_tokens = self.estimate_tokens(stripped)

        # Detect keywords
        detected_complex = tuple(kw for kw in _COMPLEX_KEYWORDS if kw in lower_prompt)
        detected_medium = tuple(kw for kw in _MEDIUM_KEYWORDS if kw in lower_prompt)
        detected_keywords = detected_complex + detected_medium

        # Detect simple factual query patterns
        is_factual = any(p.match(stripped) for p in _SIMPLE_PATTERNS)

        # Classification logic (priority order):
        # 1. Token count >= complex_threshold → always complex
        if estimated_tokens >= self._complex_threshold:
            return ClassificationResult(
                task_type="complex",
                estimated_tokens=estimated_tokens,
                detected_keywords=detected_keywords,
                is_factual_query=False,
                confidence="high",
            )

        # 2. High-complexity keyword detected → complex (overrides token count)
        if detected_complex:
            confidence = "high" if estimated_tokens > self._simple_threshold else "medium"
            return ClassificationResult(
                task_type="complex",
                estimated_tokens=estimated_tokens,
                detected_keywords=detected_keywords,
                is_factual_query=is_factual,
                confidence=confidence,
            )

        # 3. Token count <= simple_threshold AND no medium keywords AND factual → simple
        if estimated_tokens <= self._simple_threshold and is_factual and not detected_medium:
            return ClassificationResult(
                task_type="simple",
                estimated_tokens=estimated_tokens,
                detected_keywords=detected_keywords,
                is_factual_query=True,
                confidence="high",
            )

        # 4. Token count <= simple_threshold AND no keywords → simple
        if estimated_tokens <= self._simple_threshold and not detected_medium:
            return ClassificationResult(
                task_type="simple",
                estimated_tokens=estimated_tokens,
                detected_keywords=detected_keywords,
                is_factual_query=is_factual,
                confidence="high",
            )

        # 5. Medium keyword detected OR token count is in medium range → medium
        if detected_medium or self._simple_threshold < estimated_tokens < self._complex_threshold:
            confidence = "high" if estimated_tokens > self._simple_threshold else "medium"
            return ClassificationResult(
                task_type="medium",
                estimated_tokens=estimated_tokens,
                detected_keywords=detected_keywords,
                is_factual_query=is_factual,
                confidence=confidence,
            )

        # Default to simple for very short prompts with no signals
        return ClassificationResult(
            task_type="simple",
            estimated_tokens=estimated_tokens,
            detected_keywords=detected_keywords,
            is_factual_query=is_factual,
            confidence="low",
        )

    def classify_many(self, prompts: list[str]) -> list[ClassificationResult]:
        """Classify a batch of prompts.

        Parameters
        ----------
        prompts:
            List of prompt strings to classify.

        Returns
        -------
        list[ClassificationResult]
            One result per prompt, in the same order.
        """
        return [self.classify(p) for p in prompts]
