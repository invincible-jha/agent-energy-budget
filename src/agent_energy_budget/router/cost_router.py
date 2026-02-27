"""Cost-aware model router for agent energy budget management.

CostAwareRouter selects the most appropriate LLM model for each request
based on cost constraints, quality requirements, and configurable routing
strategies. It is the primary entry point for the router subsystem.

Usage
-----
::

    from agent_energy_budget.router.cost_router import CostAwareRouter
    from agent_energy_budget.router.models import DEFAULT_MODEL_PROFILES, RouterBudgetConfig

    budget = RouterBudgetConfig(total_budget_usd=10.0, min_quality_score=0.6)
    router = CostAwareRouter(models=DEFAULT_MODEL_PROFILES, budget=budget)

    decision = router.route("Summarise the following article...", max_cost=0.05)
    print(decision.selected_model.name, decision.estimated_cost)
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from agent_energy_budget.router.models import (
    DEFAULT_MODEL_PROFILES,
    ModelProfile,
    RoutingDecision,
    RoutingStrategy as RoutingStrategyLiteral,
    RouterBudgetConfig,
    TaskComplexity,
)
from agent_energy_budget.router.strategies import (
    BalancedStrategy,
    BudgetAwareStrategy,
    CheapestFirstStrategy,
    NoAffordableModelError,
    QualityFirstStrategy,
    RoutingStrategy,
)

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Nominal token counts used for cost estimation inside the router.
# Real token counts are unknown at routing time; these are conservative
# approximations for selection purposes.
# ---------------------------------------------------------------------------

_NOMINAL_INPUT_TOKENS: int = 1_000
_NOMINAL_OUTPUT_TOKENS: int = 500

# Complexity → rough token multiplier applied to nominal cost estimate.
_COMPLEXITY_TOKEN_MULTIPLIER: dict[TaskComplexity, float] = {
    "low": 0.5,
    "medium": 1.0,
    "high": 2.5,
}


def _infer_complexity(prompt: str) -> TaskComplexity:
    """Infer task complexity from prompt length and keyword signals.

    Keyword detection takes priority over length thresholds so that
    short prompts containing analysis keywords are still classified as
    ``"high"`` complexity.

    Parameters
    ----------
    prompt:
        The raw prompt text submitted by the caller.

    Returns
    -------
    TaskComplexity
        ``"high"`` when the prompt contains reasoning/analysis keywords
        or is >= 500 chars. ``"low"`` for very short prompts (<= 100
        chars) with no keywords. ``"medium"`` otherwise.
    """
    stripped = prompt.strip()
    length = len(stripped)

    high_complexity_keywords = {
        "analyse",
        "analyze",
        "explain",
        "compare",
        "contrast",
        "evaluate",
        "synthesise",
        "synthesize",
        "summarise",
        "summarize",
        "research",
        "elaborate",
        "comprehensive",
        "detailed",
        "in-depth",
        "step by step",
        "step-by-step",
    }
    lower_prompt = stripped.lower()
    has_keyword = any(kw in lower_prompt for kw in high_complexity_keywords)

    # Keyword match always implies high complexity regardless of length.
    if has_keyword or length >= 500:
        return "high"

    if length <= 100:
        return "low"

    return "medium"


class CostAwareRouter:
    """Routes LLM calls to the most cost-effective model for the request.

    The router holds a catalogue of ``ModelProfile`` instances and applies
    a configurable ``RoutingStrategy`` to select a model for each call.
    It tracks remaining budget internally so successive calls deplete the
    session budget.

    Parameters
    ----------
    models:
        List of candidate model profiles. Must not be empty.
    budget:
        Router-level budget configuration.
    strategy:
        Routing strategy name. One of ``"cheapest_first"``,
        ``"quality_first"``, ``"balanced"``, ``"budget_aware"``.
        Defaults to ``"balanced"``.

    Raises
    ------
    ValueError
        If ``models`` is empty or ``strategy`` is not recognised.

    Examples
    --------
    >>> from agent_energy_budget.router.models import DEFAULT_MODEL_PROFILES, RouterBudgetConfig
    >>> budget = RouterBudgetConfig(total_budget_usd=5.0)
    >>> router = CostAwareRouter(models=DEFAULT_MODEL_PROFILES, budget=budget)
    >>> decision = router.route("Hello, world!")
    >>> print(decision.selected_model.name)
    """

    _STRATEGY_MAP: dict[str, RoutingStrategy] = {
        "cheapest_first": CheapestFirstStrategy(),
        "quality_first": QualityFirstStrategy(),
        "balanced": BalancedStrategy(),
        "budget_aware": BudgetAwareStrategy(),
    }

    def __init__(
        self,
        models: list[ModelProfile] | None = None,
        budget: RouterBudgetConfig | None = None,
        strategy: RoutingStrategyLiteral = "balanced",
    ) -> None:
        resolved_models = models if models is not None else DEFAULT_MODEL_PROFILES
        if not resolved_models:
            raise ValueError(
                "CostAwareRouter requires at least one ModelProfile in models."
            )
        if strategy not in self._STRATEGY_MAP:
            raise ValueError(
                f"Unknown routing strategy {strategy!r}. "
                f"Valid options: {sorted(self._STRATEGY_MAP.keys())}."
            )

        self._models: list[ModelProfile] = list(resolved_models)
        self._budget: RouterBudgetConfig = budget or RouterBudgetConfig(
            total_budget_usd=float("inf")
        )
        self._strategy_name: RoutingStrategyLiteral = strategy
        self._strategy: RoutingStrategy = self._STRATEGY_MAP[strategy]
        self._remaining_budget: float = self._budget.total_budget_usd

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def models(self) -> list[ModelProfile]:
        """Read-only view of the model catalogue."""
        return list(self._models)

    @property
    def budget(self) -> RouterBudgetConfig:
        """Router-level budget configuration."""
        return self._budget

    @property
    def strategy_name(self) -> RoutingStrategyLiteral:
        """Name of the active routing strategy."""
        return self._strategy_name

    @property
    def remaining_budget(self) -> float:
        """Remaining budget in USD after previous routing decisions."""
        return self._remaining_budget

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def select_model(
        self,
        task_complexity: TaskComplexity = "medium",
        remaining_budget: float | None = None,
    ) -> ModelProfile:
        """Select a model without recording a routing decision.

        Use ``route()`` for the full routing workflow that tracks budget
        consumption. This method is useful for previewing selections.

        Parameters
        ----------
        task_complexity:
            Hint about the complexity of the task.
        remaining_budget:
            Override the internal remaining budget for this selection.
            Defaults to the router's current remaining budget.

        Returns
        -------
        ModelProfile
            The selected model profile.

        Raises
        ------
        NoAffordableModelError
            When no model fits within the budget.
        """
        budget_to_use = (
            remaining_budget if remaining_budget is not None else self._remaining_budget
        )
        selected, _ = self._strategy.select(
            self._models,
            budget_to_use,
            self._budget,
            task_complexity,
        )
        return selected

    def route(
        self,
        prompt: str,
        max_cost: float | None = None,
    ) -> RoutingDecision:
        """Route a prompt to the best model and record the budget consumption.

        The method:
        1. Infers task complexity from the prompt.
        2. Applies the configured routing strategy to select a model.
        3. Computes an estimated cost using a nominal token count scaled
           by task complexity.
        4. Deducts the estimated cost from the internal remaining budget.
        5. Returns a ``RoutingDecision`` describing the outcome.

        Parameters
        ----------
        prompt:
            The prompt text to route. Used to infer task complexity.
        max_cost:
            Optional hard cap on estimated cost for this request. If the
            selected model's estimated cost exceeds ``max_cost``, the
            cheapest affordable model is selected instead.

        Returns
        -------
        RoutingDecision
            Routing outcome including selected model, reason, and costs.

        Raises
        ------
        NoAffordableModelError
            When no model can be selected within the budget constraints.
        """
        task_complexity = _infer_complexity(prompt)
        effective_budget = self._remaining_budget
        if max_cost is not None:
            effective_budget = min(effective_budget, max_cost)

        selected_model, reason = self._strategy.select(
            self._models,
            effective_budget,
            self._budget,
            task_complexity,
        )

        # If max_cost is set and the primary selection exceeds it, fall back
        # to cheapest-first within max_cost.
        multiplier = _COMPLEXITY_TOKEN_MULTIPLIER.get(task_complexity, 1.0)
        estimated_cost = selected_model.cost_for_tokens(
            int(_NOMINAL_INPUT_TOKENS * multiplier),
            int(_NOMINAL_OUTPUT_TOKENS * multiplier),
        )

        if max_cost is not None and estimated_cost > max_cost:
            fallback_strategy = CheapestFirstStrategy()
            try:
                selected_model, reason = fallback_strategy.select(
                    self._models,
                    max_cost,
                    self._budget,
                    task_complexity,
                )
                estimated_cost = selected_model.cost_for_tokens(
                    int(_NOMINAL_INPUT_TOKENS * multiplier),
                    int(_NOMINAL_OUTPUT_TOKENS * multiplier),
                )
                reason = f"max_cost override (${max_cost:.4f}): {reason}"
            except NoAffordableModelError:
                # Re-raise with context
                raise NoAffordableModelError(
                    remaining_budget=max_cost,
                    min_quality_score=self._budget.min_quality_score,
                )

        self._remaining_budget = max(0.0, self._remaining_budget - estimated_cost)

        logger.debug(
            "Router selected '%s' via %s strategy. "
            "estimated_cost=%.6f remaining_budget=%.6f",
            selected_model.name,
            self._strategy_name,
            estimated_cost,
            self._remaining_budget,
        )

        return RoutingDecision(
            selected_model=selected_model,
            reason=reason,
            estimated_cost=estimated_cost,
            remaining_budget=self._remaining_budget,
        )

    def reset_budget(self, new_budget: float | None = None) -> None:
        """Reset the internal remaining budget.

        Parameters
        ----------
        new_budget:
            New budget value in USD. Defaults to
            ``self._budget.total_budget_usd``.
        """
        self._remaining_budget = (
            new_budget if new_budget is not None else self._budget.total_budget_usd
        )

    def swap_strategy(self, strategy: RoutingStrategyLiteral) -> None:
        """Replace the active routing strategy at runtime.

        Parameters
        ----------
        strategy:
            Name of the new strategy to activate.

        Raises
        ------
        ValueError
            If ``strategy`` is not a known strategy name.
        """
        if strategy not in self._STRATEGY_MAP:
            raise ValueError(
                f"Unknown routing strategy {strategy!r}. "
                f"Valid options: {sorted(self._STRATEGY_MAP.keys())}."
            )
        self._strategy_name = strategy
        self._strategy = self._STRATEGY_MAP[strategy]

    def summary(self) -> dict[str, object]:
        """Return a summary of the router state as a plain dict.

        Returns
        -------
        dict[str, object]
            Keys: ``strategy``, ``model_count``, ``total_budget_usd``,
            ``remaining_budget_usd``, ``models``.
        """
        return {
            "strategy": self._strategy_name,
            "model_count": len(self._models),
            "total_budget_usd": self._budget.total_budget_usd,
            "remaining_budget_usd": self._remaining_budget,
            "models": [m.name for m in self._models],
        }
