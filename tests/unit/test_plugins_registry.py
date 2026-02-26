"""Unit tests for agent_energy_budget.plugins.registry."""
from __future__ import annotations

from abc import ABC, abstractmethod

import pytest

from agent_energy_budget.plugins.registry import (
    PluginAlreadyRegisteredError,
    PluginNotFoundError,
    PluginRegistry,
)


# ---------------------------------------------------------------------------
# Base class for test plugins
# ---------------------------------------------------------------------------


class BaseWidget(ABC):
    @abstractmethod
    def render(self) -> str: ...


class SquareWidget(BaseWidget):
    def render(self) -> str:
        return "[]"


class CircleWidget(BaseWidget):
    def render(self) -> str:
        return "()"


# ---------------------------------------------------------------------------
# PluginNotFoundError / PluginAlreadyRegisteredError
# ---------------------------------------------------------------------------


class TestPluginErrors:
    def test_plugin_not_found_error_message(self) -> None:
        error = PluginNotFoundError("my-plugin", "widgets")
        assert "my-plugin" in str(error)
        assert "widgets" in str(error)

    def test_plugin_not_found_attributes(self) -> None:
        error = PluginNotFoundError("p", "r")
        assert error.plugin_name == "p"
        assert error.registry_name == "r"

    def test_plugin_already_registered_error_message(self) -> None:
        error = PluginAlreadyRegisteredError("my-plugin", "widgets")
        assert "my-plugin" in str(error)

    def test_plugin_already_registered_attributes(self) -> None:
        error = PluginAlreadyRegisteredError("p", "r")
        assert error.plugin_name == "p"
        assert error.registry_name == "r"


# ---------------------------------------------------------------------------
# PluginRegistry — construction and basic properties
# ---------------------------------------------------------------------------


@pytest.fixture
def registry() -> PluginRegistry[BaseWidget]:
    return PluginRegistry(BaseWidget, "widgets")


class TestPluginRegistryInit:
    def test_empty_registry(self, registry: PluginRegistry[BaseWidget]) -> None:
        assert len(registry) == 0

    def test_list_plugins_empty(self, registry: PluginRegistry[BaseWidget]) -> None:
        assert registry.list_plugins() == []

    def test_repr_contains_name(self, registry: PluginRegistry[BaseWidget]) -> None:
        assert "widgets" in repr(registry)
        assert "BaseWidget" in repr(registry)


# ---------------------------------------------------------------------------
# PluginRegistry.register (decorator)
# ---------------------------------------------------------------------------


class TestRegisterDecorator:
    def test_decorator_registers_class(self, registry: PluginRegistry[BaseWidget]) -> None:
        @registry.register("square")
        class MySquare(BaseWidget):
            def render(self) -> str:
                return "[]"

        assert "square" in registry

    def test_decorator_returns_class_unchanged(self, registry: PluginRegistry[BaseWidget]) -> None:
        @registry.register("circle")
        class MyCircle(BaseWidget):
            def render(self) -> str:
                return "()"

        assert MyCircle().render() == "()"

    def test_duplicate_name_raises(self, registry: PluginRegistry[BaseWidget]) -> None:
        @registry.register("widget-a")
        class W1(BaseWidget):
            def render(self) -> str:
                return "w1"

        with pytest.raises(PluginAlreadyRegisteredError):
            @registry.register("widget-a")
            class W2(BaseWidget):
                def render(self) -> str:
                    return "w2"

    def test_non_subclass_raises_type_error(self, registry: PluginRegistry[BaseWidget]) -> None:
        with pytest.raises(TypeError, match="subclass"):
            @registry.register("bad")
            class NotAWidget:
                pass


# ---------------------------------------------------------------------------
# PluginRegistry.register_class (direct)
# ---------------------------------------------------------------------------


class TestRegisterClass:
    def test_register_class_directly(self, registry: PluginRegistry[BaseWidget]) -> None:
        registry.register_class("square", SquareWidget)
        assert "square" in registry

    def test_register_class_duplicate_raises(self, registry: PluginRegistry[BaseWidget]) -> None:
        registry.register_class("square", SquareWidget)
        with pytest.raises(PluginAlreadyRegisteredError):
            registry.register_class("square", CircleWidget)

    def test_register_class_non_subclass_raises(self, registry: PluginRegistry[BaseWidget]) -> None:
        with pytest.raises(TypeError):
            registry.register_class("bad", str)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# PluginRegistry.deregister
# ---------------------------------------------------------------------------


class TestDeregister:
    def test_deregister_removes_plugin(self, registry: PluginRegistry[BaseWidget]) -> None:
        registry.register_class("square", SquareWidget)
        registry.deregister("square")
        assert "square" not in registry

    def test_deregister_unknown_raises(self, registry: PluginRegistry[BaseWidget]) -> None:
        with pytest.raises(PluginNotFoundError):
            registry.deregister("nonexistent")

    def test_deregister_then_re_register(self, registry: PluginRegistry[BaseWidget]) -> None:
        registry.register_class("square", SquareWidget)
        registry.deregister("square")
        registry.register_class("square", CircleWidget)
        cls = registry.get("square")
        assert cls is CircleWidget


# ---------------------------------------------------------------------------
# PluginRegistry.get
# ---------------------------------------------------------------------------


class TestGet:
    def test_get_returns_class(self, registry: PluginRegistry[BaseWidget]) -> None:
        registry.register_class("square", SquareWidget)
        cls = registry.get("square")
        assert cls is SquareWidget

    def test_get_unknown_raises(self, registry: PluginRegistry[BaseWidget]) -> None:
        with pytest.raises(PluginNotFoundError):
            registry.get("nonexistent")

    def test_get_class_is_instantiable(self, registry: PluginRegistry[BaseWidget]) -> None:
        registry.register_class("square", SquareWidget)
        cls = registry.get("square")
        instance = cls()
        assert instance.render() == "[]"


# ---------------------------------------------------------------------------
# PluginRegistry.list_plugins
# ---------------------------------------------------------------------------


class TestListPlugins:
    def test_sorted_alphabetically(self, registry: PluginRegistry[BaseWidget]) -> None:
        registry.register_class("zebra", SquareWidget)
        registry.register_class("apple", CircleWidget)
        names = registry.list_plugins()
        assert names == sorted(names)

    def test_all_registered_names_present(self, registry: PluginRegistry[BaseWidget]) -> None:
        registry.register_class("p1", SquareWidget)
        registry.register_class("p2", CircleWidget)
        names = registry.list_plugins()
        assert "p1" in names
        assert "p2" in names


# ---------------------------------------------------------------------------
# PluginRegistry operators
# ---------------------------------------------------------------------------


class TestOperators:
    def test_contains_true(self, registry: PluginRegistry[BaseWidget]) -> None:
        registry.register_class("square", SquareWidget)
        assert "square" in registry

    def test_contains_false(self, registry: PluginRegistry[BaseWidget]) -> None:
        assert "nonexistent" not in registry

    def test_len_empty(self, registry: PluginRegistry[BaseWidget]) -> None:
        assert len(registry) == 0

    def test_len_after_registration(self, registry: PluginRegistry[BaseWidget]) -> None:
        registry.register_class("a", SquareWidget)
        registry.register_class("b", CircleWidget)
        assert len(registry) == 2

    def test_len_after_deregistration(self, registry: PluginRegistry[BaseWidget]) -> None:
        registry.register_class("a", SquareWidget)
        registry.deregister("a")
        assert len(registry) == 0
