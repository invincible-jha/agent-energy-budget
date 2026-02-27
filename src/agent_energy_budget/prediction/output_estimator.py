"""OutputEstimator — heuristic estimation of output token count.

Estimating output tokens before an LLM call completes is inherently
uncertain. This module provides task-type heuristics that give a
reasonable central estimate plus a confidence score, enabling the
CostPredictor to compute a realistic cost range.

Task types
----------
- ``chat``      : Short conversational replies (~200 tokens)
- ``code_gen``  : Code generation tasks (~500 tokens)
- ``summary``   : Summarisation (1/4 of input length)
- ``qa``        : Question-answering (~150 tokens)
- ``analysis``  : Document analysis (~400 tokens)
- ``extraction``  : Entity / data extraction (~250 tokens)
- ``translation`` : Translation (same length as input)
- ``unknown``   : Conservative fallback (~300 tokens)
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class TaskType(str, Enum):
    """Known task categories for output estimation."""

    CHAT = "chat"
    CODE_GEN = "code_gen"
    SUMMARY = "summary"
    QA = "qa"
    ANALYSIS = "analysis"
    EXTRACTION = "extraction"
    TRANSLATION = "translation"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class OutputEstimate:
    """Result of an output token estimation.

    Parameters
    ----------
    estimated_tokens:
        Central estimate of expected output tokens.
    confidence:
        Confidence score in [0.0, 1.0]. Higher means more reliable.
        Task-type-specific heuristics score 0.5–0.8; unknown falls back
        to 0.3.
    method:
        Human-readable description of the estimation method used.
    task_type:
        The task type resolved for this estimate.
    low_estimate:
        Conservative (10th-percentile) token count estimate.
    high_estimate:
        Generous (90th-percentile) token count estimate.
    """

    estimated_tokens: int
    confidence: float
    method: str
    task_type: TaskType
    low_estimate: int
    high_estimate: int

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence must be in [0, 1], got {self.confidence}")


# ---------------------------------------------------------------------------
# Task-type heuristic configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _TaskConfig:
    base_tokens: int
    low_multiplier: float
    high_multiplier: float
    confidence: float
    method: str


# Proportion of input to use for input-dependent tasks
_INPUT_FRACTION: dict[TaskType, float] = {
    TaskType.SUMMARY: 0.25,
    TaskType.TRANSLATION: 1.0,
}

_TASK_CONFIGS: dict[TaskType, _TaskConfig] = {
    TaskType.CHAT: _TaskConfig(
        base_tokens=200,
        low_multiplier=0.4,
        high_multiplier=2.5,
        confidence=0.65,
        method="chat-heuristic",
    ),
    TaskType.CODE_GEN: _TaskConfig(
        base_tokens=500,
        low_multiplier=0.3,
        high_multiplier=2.0,
        confidence=0.55,
        method="code-gen-heuristic",
    ),
    TaskType.QA: _TaskConfig(
        base_tokens=150,
        low_multiplier=0.3,
        high_multiplier=2.0,
        confidence=0.70,
        method="qa-heuristic",
    ),
    TaskType.ANALYSIS: _TaskConfig(
        base_tokens=400,
        low_multiplier=0.5,
        high_multiplier=2.0,
        confidence=0.55,
        method="analysis-heuristic",
    ),
    TaskType.EXTRACTION: _TaskConfig(
        base_tokens=250,
        low_multiplier=0.3,
        high_multiplier=2.0,
        confidence=0.60,
        method="extraction-heuristic",
    ),
    TaskType.UNKNOWN: _TaskConfig(
        base_tokens=300,
        low_multiplier=0.3,
        high_multiplier=3.0,
        confidence=0.30,
        method="unknown-fallback",
    ),
}


class OutputEstimator:
    """Estimate output token counts using task-type heuristics.

    Parameters
    ----------
    default_task_type:
        Fallback task type when none is specified. Defaults to UNKNOWN.

    Examples
    --------
    >>> estimator = OutputEstimator()
    >>> estimate = estimator.estimate(task_type="chat", input_tokens=500)
    >>> estimate.estimated_tokens
    200
    """

    def __init__(
        self,
        default_task_type: TaskType = TaskType.UNKNOWN,
    ) -> None:
        self._default_task_type = default_task_type

    # ------------------------------------------------------------------
    # Core estimation
    # ------------------------------------------------------------------

    def estimate(
        self,
        task_type: str | TaskType = TaskType.UNKNOWN,
        input_tokens: int = 0,
        max_tokens: Optional[int] = None,
    ) -> OutputEstimate:
        """Estimate output token count for the given task.

        Parameters
        ----------
        task_type:
            Task type string or TaskType enum value.
        input_tokens:
            Number of input tokens (used for input-proportional tasks
            like summary and translation).
        max_tokens:
            Hard cap provided by the caller (e.g. from API parameter).
            When provided, estimates are capped to this value.

        Returns
        -------
        OutputEstimate
        """
        resolved_type = self._resolve_task_type(task_type)

        estimated = self._base_estimate(resolved_type, input_tokens)
        config = _TASK_CONFIGS.get(resolved_type, _TASK_CONFIGS[TaskType.UNKNOWN])

        low = max(1, int(estimated * config.low_multiplier))
        high = int(estimated * config.high_multiplier)

        if max_tokens is not None and max_tokens > 0:
            estimated = min(estimated, max_tokens)
            low = min(low, max_tokens)
            high = min(high, max_tokens)

        return OutputEstimate(
            estimated_tokens=max(1, estimated),
            confidence=config.confidence,
            method=config.method,
            task_type=resolved_type,
            low_estimate=max(1, low),
            high_estimate=max(1, high),
        )

    def estimate_from_hint(self, hint: str, input_tokens: int = 0) -> OutputEstimate:
        """Estimate output tokens from a free-text task description.

        Performs keyword matching to classify the task type, then falls
        back to UNKNOWN.

        Parameters
        ----------
        hint:
            Free-text description of the task (e.g. "write a Python
            function", "summarize the document").
        input_tokens:
            Input token count for proportional tasks.

        Returns
        -------
        OutputEstimate
        """
        lower = hint.lower()

        if any(kw in lower for kw in ("code", "function", "class", "script", "program", "implement")):
            return self.estimate(TaskType.CODE_GEN, input_tokens)
        if any(kw in lower for kw in ("summar", "tldr", "brief", "condense", "shorten")):
            return self.estimate(TaskType.SUMMARY, input_tokens)
        if any(kw in lower for kw in ("translat", "convert language", "in french", "in spanish")):
            return self.estimate(TaskType.TRANSLATION, input_tokens)
        if any(kw in lower for kw in ("extract", "parse", "identify", "find all", "list all")):
            return self.estimate(TaskType.EXTRACTION, input_tokens)
        if any(kw in lower for kw in ("analyz", "analyse", "evaluate", "assess", "review")):
            return self.estimate(TaskType.ANALYSIS, input_tokens)
        if any(kw in lower for kw in ("answer", "explain", "what is", "how does", "why")):
            return self.estimate(TaskType.QA, input_tokens)
        if any(kw in lower for kw in ("chat", "hello", "hi,", "hi ", "convers")):
            return self.estimate(TaskType.CHAT, input_tokens)

        return self.estimate(TaskType.UNKNOWN, input_tokens)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _resolve_task_type(self, task_type: str | TaskType) -> TaskType:
        if isinstance(task_type, TaskType):
            return task_type
        try:
            return TaskType(task_type.lower())
        except ValueError:
            return self._default_task_type

    def _base_estimate(self, task_type: TaskType, input_tokens: int) -> int:
        """Compute the base token estimate for *task_type*."""
        # Input-proportional tasks
        fraction = _INPUT_FRACTION.get(task_type)
        if fraction is not None and input_tokens > 0:
            return max(1, int(input_tokens * fraction))

        config = _TASK_CONFIGS.get(task_type, _TASK_CONFIGS[TaskType.UNKNOWN])
        return config.base_tokens

    def __repr__(self) -> str:
        return f"OutputEstimator(default_task_type={self._default_task_type.value!r})"
