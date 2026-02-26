"""Strategy registry for degradation strategies.

Allows runtime registration and retrieval of DegradationStrategyBase
implementations by name. Pre-registers all built-in strategies.
"""
from __future__ import annotations

import logging

from agent_energy_budget.degradation.base import DegradationStrategyBase

logger = logging.getLogger(__name__)


class StrategyNotFoundError(KeyError):
    """Raised when a requested strategy name is not in the registry."""

    def __init__(self, name: str, available: list[str]) -> None:
        super().__init__(
            f"Degradation strategy {name!r} not registered. "
            f"Available strategies: {available}"
        )


class StrategyAlreadyRegisteredError(ValueError):
    """Raised when registering a name that is already taken."""

    def __init__(self, name: str) -> None:
        super().__init__(
            f"Degradation strategy {name!r} is already registered. "
            "Deregister it first or use a different name."
        )


class StrategyRegistry:
    """Register and retrieve degradation strategies by name.

    The registry stores both class references and optional singleton
    instances. If a singleton is stored, :meth:`get_instance` returns it
    directly; otherwise a new instance is created each call.

    Built-in strategies are registered automatically when the registry
    is first instantiated.

    Examples
    --------
    >>> registry = StrategyRegistry()
    >>> strategy = registry.get_instance("token_reduction")
    """

    def __init__(self, *, register_builtins: bool = True) -> None:
        self._classes: dict[str, type[DegradationStrategyBase]] = {}
        self._instances: dict[str, DegradationStrategyBase] = {}

        if register_builtins:
            self._register_builtins()

    # ------------------------------------------------------------------
    # Internal bootstrap
    # ------------------------------------------------------------------

    def _register_builtins(self) -> None:
        """Register all built-in strategies."""
        from agent_energy_budget.degradation.block_with_error import BlockStrategy
        from agent_energy_budget.degradation.cached_fallback import CachedFallbackStrategy
        from agent_energy_budget.degradation.model_downgrade import ModelDowngradeStrategy
        from agent_energy_budget.degradation.token_reduction import TokenReductionStrategy

        builtins: dict[str, type[DegradationStrategyBase]] = {
            "model_downgrade": ModelDowngradeStrategy,
            "token_reduction": TokenReductionStrategy,
            "block_with_error": BlockStrategy,
            "cached_fallback": CachedFallbackStrategy,
        }
        for name, cls in builtins.items():
            self._classes[name] = cls

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(
        self,
        name: str,
        strategy_class: type[DegradationStrategyBase],
        *,
        instance: DegradationStrategyBase | None = None,
        overwrite: bool = False,
    ) -> None:
        """Register a strategy class (and optionally a singleton instance).

        Parameters
        ----------
        name:
            Unique string key for this strategy.
        strategy_class:
            The concrete strategy class.
        instance:
            Optional pre-constructed singleton instance.
        overwrite:
            When True, silently replaces an existing registration.
            Defaults to False.

        Raises
        ------
        StrategyAlreadyRegisteredError
            If *name* is already registered and *overwrite* is False.
        TypeError
            If *strategy_class* is not a subclass of DegradationStrategyBase.
        """
        if not (isinstance(strategy_class, type) and issubclass(strategy_class, DegradationStrategyBase)):
            raise TypeError(
                f"{strategy_class!r} must be a subclass of DegradationStrategyBase"
            )
        if name in self._classes and not overwrite:
            raise StrategyAlreadyRegisteredError(name)

        self._classes[name] = strategy_class
        if instance is not None:
            self._instances[name] = instance
        logger.debug("Registered degradation strategy %r -> %s", name, strategy_class.__name__)

    def deregister(self, name: str) -> None:
        """Remove a strategy registration.

        Parameters
        ----------
        name:
            The key to remove.

        Raises
        ------
        StrategyNotFoundError
            If *name* is not registered.
        """
        if name not in self._classes:
            raise StrategyNotFoundError(name, self.list_strategies())
        del self._classes[name]
        self._instances.pop(name, None)
        logger.debug("Deregistered degradation strategy %r", name)

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def get_class(self, name: str) -> type[DegradationStrategyBase]:
        """Return the strategy class registered under *name*.

        Parameters
        ----------
        name:
            Registered strategy name.

        Returns
        -------
        type[DegradationStrategyBase]
            The strategy class.

        Raises
        ------
        StrategyNotFoundError
            If *name* is not registered.
        """
        if name not in self._classes:
            raise StrategyNotFoundError(name, self.list_strategies())
        return self._classes[name]

    def get_instance(self, name: str) -> DegradationStrategyBase:
        """Return a strategy instance, creating one if needed.

        If a singleton instance was registered, it is returned directly.
        Otherwise a new instance is created with no arguments.

        Parameters
        ----------
        name:
            Registered strategy name.

        Returns
        -------
        DegradationStrategyBase
            Strategy instance.

        Raises
        ------
        StrategyNotFoundError
            If *name* is not registered.
        """
        if name not in self._classes:
            raise StrategyNotFoundError(name, self.list_strategies())
        if name not in self._instances:
            self._instances[name] = self._classes[name]()
        return self._instances[name]

    def list_strategies(self) -> list[str]:
        """Return sorted list of all registered strategy names."""
        return sorted(self._classes.keys())

    def __contains__(self, name: object) -> bool:
        """Support ``"token_reduction" in registry``."""
        return name in self._classes

    def __len__(self) -> int:
        """Return number of registered strategies."""
        return len(self._classes)
