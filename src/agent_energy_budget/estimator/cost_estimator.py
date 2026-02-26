"""Pre-execution LLM cost estimator.

Estimates the cost of an LLM call before it is made, using the pricing
tables and character-based token approximation. When tiktoken is
available the token counts are more accurate and the confidence score
is higher.

Confidence levels
-----------------
char_counting  — 0.6  (rough chars/4 approximation)
heuristic      — 0.7  (word-based heuristic via TokenCounter)
tiktoken       — 0.9  (byte-pair-encoding token count)
"""
from __future__ import annotations

from dataclasses import dataclass

from agent_energy_budget.pricing.tables import (
    PROVIDER_PRICING,
    ModelPricing,
    get_pricing,
)
from agent_energy_budget.pricing.token_counter import TokenCounter


@dataclass(frozen=True)
class CostEstimate:
    """Result of a single pre-execution cost estimation.

    Parameters
    ----------
    model:
        Model identifier used for the estimate.
    estimated_input_tokens:
        Estimated number of input (prompt) tokens.
    estimated_output_tokens:
        Estimated number of output (completion) tokens.
    estimated_cost_usd:
        Estimated total cost in USD.
    confidence:
        Confidence score for the estimate in [0.0, 1.0].
        Higher means more accurate token counting was used.
    """

    model: str
    estimated_input_tokens: int
    estimated_output_tokens: int
    estimated_cost_usd: float
    confidence: float


class CostEstimator:
    """Standalone pre-call LLM cost estimator.

    Uses pricing tables from the :mod:`agent_energy_budget.pricing` module
    to produce cost estimates before any API call is made. This allows
    agents to gate decisions on projected cost without issuing network
    requests.

    Parameters
    ----------
    token_counter:
        Optional custom :class:`~agent_energy_budget.pricing.token_counter.TokenCounter`.
        A default instance is created when not provided.
    custom_pricing:
        Optional extra ``{model_id: ModelPricing}`` entries to include
        alongside the built-in ``PROVIDER_PRICING`` table.

    Examples
    --------
    >>> estimator = CostEstimator()
    >>> estimate = estimator.estimate_llm_call("gpt-4o-mini", "Summarise this document.", 512)
    >>> estimate.estimated_cost_usd
    0.000...
    """

    # Confidence values per counting backend
    _CONFIDENCE_CHAR_COUNT: float = 0.6
    _CONFIDENCE_HEURISTIC: float = 0.7
    _CONFIDENCE_TIKTOKEN: float = 0.9

    def __init__(
        self,
        token_counter: TokenCounter | None = None,
        custom_pricing: dict[str, ModelPricing] | None = None,
    ) -> None:
        self._counter = token_counter or TokenCounter()
        self._custom_pricing: dict[str, ModelPricing] = custom_pricing or {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_pricing(self, model: str) -> ModelPricing:
        """Resolve model to pricing, checking custom table first."""
        if model in self._custom_pricing:
            return self._custom_pricing[model]
        return get_pricing(model)

    def _confidence_for_backend(self) -> float:
        """Return the confidence score for the active counting backend."""
        backend = self._counter.backend
        if backend == "tiktoken":
            return self._CONFIDENCE_TIKTOKEN
        return self._CONFIDENCE_HEURISTIC

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def estimate_tokens(self, text: str) -> int:
        """Estimate the number of tokens in *text*.

        Uses character-division (``len(text) // 4``) as a fast fallback
        when the TokenCounter's heuristic is unavailable for very short
        strings. For normal text the :class:`TokenCounter` is preferred.

        Parameters
        ----------
        text:
            Raw UTF-8 text to estimate.

        Returns
        -------
        int
            Estimated token count (>= 1 for non-empty strings, 0 otherwise).
        """
        if not text:
            return 0
        return self._counter.count(text)

    def estimate_llm_call(
        self,
        model: str,
        prompt_text: str,
        max_output_tokens: int = 4096,
    ) -> CostEstimate:
        """Estimate the cost of a single LLM call before it is made.

        The input token count is derived from *prompt_text* using the
        active :class:`TokenCounter` backend. The output token count
        uses *max_output_tokens* as the worst-case upper bound, making
        this a conservative (upper-bound) estimate.

        Parameters
        ----------
        model:
            Model identifier (e.g. ``"claude-haiku-4"``, ``"gpt-4o-mini"``).
        prompt_text:
            The full prompt string that will be sent as input.
        max_output_tokens:
            Maximum output tokens that may be generated (default 4096).

        Returns
        -------
        CostEstimate
            Estimated cost and associated metadata.

        Raises
        ------
        KeyError
            If *model* cannot be resolved to a pricing record.
        """
        pricing = self._resolve_pricing(model)
        input_tokens = self.estimate_tokens(prompt_text)
        cost = pricing.cost_for_tokens(input_tokens, max_output_tokens)
        confidence = self._confidence_for_backend()

        return CostEstimate(
            model=model,
            estimated_input_tokens=input_tokens,
            estimated_output_tokens=max_output_tokens,
            estimated_cost_usd=cost,
            confidence=confidence,
        )

    def estimate_workflow(self, steps: list[dict[str, str]]) -> float:
        """Estimate the total cost of a multi-step workflow.

        Each step dict must contain at minimum ``"model"`` and
        ``"prompt"`` keys. An optional ``"max_output_tokens"`` key
        (as a string-encoded integer) sets the output token cap for
        that step; it defaults to ``4096`` when absent.

        Parameters
        ----------
        steps:
            List of step dicts, each with keys:
            - ``"model"`` (str) — model identifier
            - ``"prompt"`` (str) — prompt text for this step
            - ``"max_output_tokens"`` (str, optional) — output token cap

        Returns
        -------
        float
            Summed estimated cost in USD across all steps.

        Raises
        ------
        KeyError
            If a step's ``"model"`` key is missing or cannot be resolved.
        ValueError
            If ``"max_output_tokens"`` cannot be parsed as an integer.
        """
        total_cost = 0.0
        for step in steps:
            model = step["model"]
            prompt = step.get("prompt", "")
            max_output_raw = step.get("max_output_tokens", "4096")
            max_output_tokens = int(max_output_raw)
            estimate = self.estimate_llm_call(model, prompt, max_output_tokens)
            total_cost += estimate.estimated_cost_usd
        return round(total_cost, 8)

    def compare_models(
        self,
        models: list[str],
        prompt_text: str,
        max_output_tokens: int = 4096,
    ) -> list[CostEstimate]:
        """Compare costs for *prompt_text* across multiple models.

        Models that cannot be resolved are silently skipped.

        Parameters
        ----------
        models:
            List of model identifiers to compare.
        prompt_text:
            Prompt text to estimate for each model.
        max_output_tokens:
            Output token cap used for each estimate.

        Returns
        -------
        list[CostEstimate]
            Estimates sorted cheapest first.
        """
        estimates: list[CostEstimate] = []
        for model in models:
            try:
                estimates.append(
                    self.estimate_llm_call(model, prompt_text, max_output_tokens)
                )
            except KeyError:
                continue
        return sorted(estimates, key=lambda e: e.estimated_cost_usd)

    def cheapest_model_for_budget(
        self,
        budget_usd: float,
        prompt_text: str,
        max_output_tokens: int = 4096,
    ) -> CostEstimate | None:
        """Return the most capable model whose estimated cost fits *budget_usd*.

        Searches all known models (``PROVIDER_PRICING`` plus any custom
        pricing supplied at construction). Returns the model with the
        highest input-token price that still fits — i.e., the best
        quality that is affordable.

        Parameters
        ----------
        budget_usd:
            Available budget in USD.
        prompt_text:
            Prompt to estimate.
        max_output_tokens:
            Output token cap for the estimate.

        Returns
        -------
        CostEstimate | None
            Best affordable estimate, or ``None`` if nothing fits.
        """
        combined: dict[str, ModelPricing] = {
            **PROVIDER_PRICING,
            **self._custom_pricing,
        }
        candidates: list[CostEstimate] = []
        for model_id in combined:
            try:
                estimate = self.estimate_llm_call(
                    model_id, prompt_text, max_output_tokens
                )
                if estimate.estimated_cost_usd <= budget_usd:
                    candidates.append(estimate)
            except KeyError:
                continue

        if not candidates:
            return None

        # Return the most capable (highest input rate) that still fits
        pricing_map: dict[str, ModelPricing] = {**PROVIDER_PRICING, **self._custom_pricing}
        return max(
            candidates,
            key=lambda e: pricing_map.get(e.model, combined[e.model]).input_per_million,
        )
