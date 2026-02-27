"""Routing strategy implementations for the cost-aware model router.

Each strategy implements the ``RoutingStrategy`` Protocol, which defines a
single ``select`` method. Strategies are composable and stateless — they
receive the full model list and current budget at call time and return the
single best ``ModelProfile`` for the request.

Strategies
----------
CheapestFirstStrategy   — always picks the lowest-cost model.
QualityFirstStrategy    — picks the highest quality score within budget.
BalancedStrategy        — optimises quality-per-cost (efficiency ratio).
BudgetAwareStrategy     — starts quality-first, degrades to cheap as budget depletes.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

from agent_energy_budget.router.models import ModelProfile, RouterBudgetConfig, TaskComplexity


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class NoAffordableModelError(ValueError):
    """Raised when no model in the catalogue fits within the remaining budget.

    Parameters
    ----------
    remaining_budget:
        The available budget at the time of selection.
    min_quality_score:
        The minimum quality threshold that was applied.
    """

    def __init__(self, remaining_budget: float, min_quality_score: float = 0.0) -> None:
        self.remaining_budget = remaining_budget
        self.min_quality_score = min_quality_score
        super().__init__(
            f"No affordable model found: remaining_budget=${remaining_budget:.6f}, "
            f"min_quality_score={min_quality_score:.2f}."
        )


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class RoutingStrategy(Protocol):
    """Strategy interface for model selection.

    Implementations must be stateless with respect to the models list and
    budget — all state required for selection must be passed as arguments.
    """

    def select(
        self,
        models: list[ModelProfile],
        remaining_budget: float,
        budget_config: RouterBudgetConfig,
        task_complexity: TaskComplexity = "medium",
    ) -> tuple[ModelProfile, str]:
        """Select the best model given the current context.

        Parameters
        ----------
        models:
            Non-empty catalogue of candidate models.
        remaining_budget:
            Available budget in USD for this call.
        budget_config:
            Router-level budget configuration (quality floor, alert threshold).
        task_complexity:
            Hint about the task complexity: ``"low"``, ``"medium"``, or ``"high"``.

        Returns
        -------
        tuple[ModelProfile, str]
            ``(selected_model, reason)`` — the chosen profile and a
            human-readable explanation.

        Raises
        ------
        NoAffordableModelError
            When no model meets the budget + quality constraints.
        ValueError
            When ``models`` is empty.
        """
        ...


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _nominal_cost(model: ModelProfile) -> float:
    """Estimate cost for a nominal 1 000 input + 500 output token call."""
    return model.cost_for_tokens(1_000, 500)


def _filter_affordable(
    models: list[ModelProfile],
    remaining_budget: float,
    min_quality_score: float,
) -> list[ModelProfile]:
    """Return models that are affordable and meet the quality floor.

    A model is considered affordable when its nominal cost (1 000 input +
    500 output tokens) does not exceed *remaining_budget*.

    Parameters
    ----------
    models:
        Full candidate list.
    remaining_budget:
        Available budget in USD.
    min_quality_score:
        Minimum quality score (inclusive).

    Returns
    -------
    list[ModelProfile]
        Filtered list; may be empty.
    """
    return [
        m
        for m in models
        if _nominal_cost(m) <= remaining_budget and m.quality_score >= min_quality_score
    ]


def _require_non_empty(models: list[ModelProfile]) -> None:
    """Raise ValueError when the models list is empty."""
    if not models:
        raise ValueError("The models list must not be empty.")


def _raise_if_no_affordable(
    affordable: list[ModelProfile],
    remaining_budget: float,
    min_quality_score: float,
) -> None:
    """Raise NoAffordableModelError when the affordable list is empty."""
    if not affordable:
        raise NoAffordableModelError(
            remaining_budget=remaining_budget,
            min_quality_score=min_quality_score,
        )


# ---------------------------------------------------------------------------
# CheapestFirstStrategy
# ---------------------------------------------------------------------------


class CheapestFirstStrategy:
    """Always selects the model with the lowest nominal cost.

    This strategy ignores quality scores entirely. Use it when strict cost
    minimisation is the only concern (e.g. high-volume classification tasks).

    The nominal cost used for comparison is calculated with a 1 000 input +
    500 output token profile so that models with different input/output cost
    ratios are ranked fairly.

    Examples
    --------
    >>> strategy = CheapestFirstStrategy()
    >>> model, reason = strategy.select(models, remaining_budget=5.0, budget_config=cfg)
    """

    def select(
        self,
        models: list[ModelProfile],
        remaining_budget: float,
        budget_config: RouterBudgetConfig,
        task_complexity: TaskComplexity = "medium",
    ) -> tuple[ModelProfile, str]:
        """Select the lowest-cost affordable model.

        Parameters
        ----------
        models:
            Candidate model profiles.
        remaining_budget:
            Available budget in USD.
        budget_config:
            Router-level budget config.
        task_complexity:
            Not used by this strategy (cheapest is always cheapest).

        Returns
        -------
        tuple[ModelProfile, str]
            Selected model and selection reason.

        Raises
        ------
        ValueError
            If ``models`` is empty.
        NoAffordableModelError
            If no model fits within ``remaining_budget``.
        """
        _require_non_empty(models)
        affordable = _filter_affordable(
            models, remaining_budget, budget_config.min_quality_score
        )
        _raise_if_no_affordable(affordable, remaining_budget, budget_config.min_quality_score)

        selected = min(affordable, key=_nominal_cost)
        cost = _nominal_cost(selected)
        reason = (
            f"CheapestFirst: selected '{selected.name}' "
            f"(nominal cost ${cost:.6f}/call, quality={selected.quality_score:.2f})."
        )
        return selected, reason


# ---------------------------------------------------------------------------
# QualityFirstStrategy
# ---------------------------------------------------------------------------


class QualityFirstStrategy:
    """Selects the highest quality model that fits within the remaining budget.

    Within models sharing the same (maximum) quality score, the cheaper one
    is preferred as a tiebreaker.

    Examples
    --------
    >>> strategy = QualityFirstStrategy()
    >>> model, reason = strategy.select(models, remaining_budget=5.0, budget_config=cfg)
    """

    def select(
        self,
        models: list[ModelProfile],
        remaining_budget: float,
        budget_config: RouterBudgetConfig,
        task_complexity: TaskComplexity = "medium",
    ) -> tuple[ModelProfile, str]:
        """Select the highest-quality affordable model.

        Parameters
        ----------
        models:
            Candidate model profiles.
        remaining_budget:
            Available budget in USD.
        budget_config:
            Router-level budget config.
        task_complexity:
            Not used by this strategy.

        Returns
        -------
        tuple[ModelProfile, str]
            Selected model and selection reason.

        Raises
        ------
        ValueError
            If ``models`` is empty.
        NoAffordableModelError
            If no model fits within ``remaining_budget``.
        """
        _require_non_empty(models)
        affordable = _filter_affordable(
            models, remaining_budget, budget_config.min_quality_score
        )
        _raise_if_no_affordable(affordable, remaining_budget, budget_config.min_quality_score)

        # Primary: highest quality; tiebreaker: lowest cost
        selected = max(affordable, key=lambda m: (m.quality_score, -_nominal_cost(m)))
        cost = _nominal_cost(selected)
        reason = (
            f"QualityFirst: selected '{selected.name}' "
            f"(quality={selected.quality_score:.2f}, nominal cost ${cost:.6f}/call)."
        )
        return selected, reason


# ---------------------------------------------------------------------------
# BalancedStrategy
# ---------------------------------------------------------------------------


class BalancedStrategy:
    """Optimises the quality-per-cost efficiency ratio.

    Uses ``ModelProfile.cost_efficiency_ratio()`` (quality divided by
    nominal cost) as the selection criterion. Models with the same ratio
    are broken by higher quality score.

    When task complexity is ``"high"``, a 20 % quality bonus is applied to
    encourage selection of better models. When complexity is ``"low"``, a
    20 % cost bonus is applied to nudge selection toward cheaper options.

    Examples
    --------
    >>> strategy = BalancedStrategy()
    >>> model, reason = strategy.select(models, remaining_budget=5.0, budget_config=cfg)
    """

    # Complexity adjustment multipliers applied to effective quality score.
    _COMPLEXITY_QUALITY_BONUS: dict[TaskComplexity, float] = {
        "low": 0.80,
        "medium": 1.00,
        "high": 1.20,
    }

    def select(
        self,
        models: list[ModelProfile],
        remaining_budget: float,
        budget_config: RouterBudgetConfig,
        task_complexity: TaskComplexity = "medium",
    ) -> tuple[ModelProfile, str]:
        """Select the model with the best quality-per-cost ratio.

        Parameters
        ----------
        models:
            Candidate model profiles.
        remaining_budget:
            Available budget in USD.
        budget_config:
            Router-level budget config.
        task_complexity:
            Adjusts the quality weighting. ``"high"`` favours quality;
            ``"low"`` favours cost.

        Returns
        -------
        tuple[ModelProfile, str]
            Selected model and selection reason.

        Raises
        ------
        ValueError
            If ``models`` is empty.
        NoAffordableModelError
            If no model fits within ``remaining_budget``.
        """
        _require_non_empty(models)
        affordable = _filter_affordable(
            models, remaining_budget, budget_config.min_quality_score
        )
        _raise_if_no_affordable(affordable, remaining_budget, budget_config.min_quality_score)

        quality_multiplier = self._COMPLEXITY_QUALITY_BONUS.get(task_complexity, 1.0)

        def effective_ratio(model: ModelProfile) -> float:
            nominal_cost = _nominal_cost(model)
            if nominal_cost == 0.0:
                return float("inf")
            effective_quality = model.quality_score * quality_multiplier
            return effective_quality / nominal_cost

        selected = max(affordable, key=lambda m: (effective_ratio(m), m.quality_score))
        cost = _nominal_cost(selected)
        ratio = effective_ratio(selected)
        reason = (
            f"Balanced: selected '{selected.name}' "
            f"(efficiency_ratio={ratio:.4f}, "
            f"quality={selected.quality_score:.2f}, "
            f"nominal cost ${cost:.6f}/call, "
            f"complexity='{task_complexity}')."
        )
        return selected, reason


# ---------------------------------------------------------------------------
# BudgetAwareStrategy
# ---------------------------------------------------------------------------


class BudgetAwareStrategy:
    """Dynamic strategy that degrades quality selection as budget depletes.

    Behaviour at three budget utilisation levels:

    +-----------------------+----------------------------------------------+
    | Budget remaining      | Behaviour                                    |
    +=======================+==============================================+
    | > alert_threshold_pct | Quality-first: highest quality within budget |
    +-----------------------+----------------------------------------------+
    | 20–alert_threshold    | Balanced: best quality-per-cost ratio        |
    +-----------------------+----------------------------------------------+
    | < 20 %                | Cheapest-first: pure cost minimisation       |
    +-----------------------+----------------------------------------------+

    ``alert_threshold_pct`` is read from ``budget_config.alert_threshold_pct``.
    The lower 20 % threshold is fixed.

    Examples
    --------
    >>> strategy = BudgetAwareStrategy()
    >>> model, reason = strategy.select(
    ...     models, remaining_budget=2.0, budget_config=cfg
    ... )
    """

    # Below this fraction of total budget, switch to cheapest-first.
    _CRITICAL_THRESHOLD_FRACTION: float = 0.20

    def __init__(self) -> None:
        self._quality_strategy = QualityFirstStrategy()
        self._balanced_strategy = BalancedStrategy()
        self._cheapest_strategy = CheapestFirstStrategy()

    def select(
        self,
        models: list[ModelProfile],
        remaining_budget: float,
        budget_config: RouterBudgetConfig,
        task_complexity: TaskComplexity = "medium",
    ) -> tuple[ModelProfile, str]:
        """Select a model based on the current budget utilisation level.

        The active sub-strategy is determined by the fraction of
        ``budget_config.total_budget_usd`` that remains:

        - >= alert_threshold_fraction remaining: QualityFirst
        - >= critical threshold (20 %) remaining: Balanced
        - < critical threshold: CheapestFirst

        Parameters
        ----------
        models:
            Candidate model profiles.
        remaining_budget:
            Available budget in USD.
        budget_config:
            Router-level budget config (must contain total_budget_usd).
        task_complexity:
            Passed through to the active sub-strategy.

        Returns
        -------
        tuple[ModelProfile, str]
            Selected model and selection reason.

        Raises
        ------
        ValueError
            If ``models`` is empty.
        NoAffordableModelError
            If no model fits within ``remaining_budget``.
        """
        _require_non_empty(models)

        total = budget_config.total_budget_usd
        if total > 0.0:
            remaining_fraction = remaining_budget / total
        else:
            # Zero total budget — treat as fully depleted
            remaining_fraction = 0.0

        alert_fraction = budget_config.alert_threshold_fraction

        if remaining_fraction >= alert_fraction:
            sub_strategy: RoutingStrategy = self._quality_strategy
            tier_label = "quality-first"
        elif remaining_fraction >= self._CRITICAL_THRESHOLD_FRACTION:
            sub_strategy = self._balanced_strategy
            tier_label = "balanced"
        else:
            sub_strategy = self._cheapest_strategy
            tier_label = "cheapest-first"

        selected, sub_reason = sub_strategy.select(
            models, remaining_budget, budget_config, task_complexity
        )
        reason = (
            f"BudgetAware [{tier_label}] "
            f"(remaining={remaining_fraction:.1%} of ${total:.2f}): {sub_reason}"
        )
        return selected, reason
