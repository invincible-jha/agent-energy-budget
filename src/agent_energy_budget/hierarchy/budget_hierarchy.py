"""Hierarchical budget tree: org → team → agent.

:class:`HierarchicalBudget` manages a tree of budget nodes where each
node has a spend limit and tracks cumulative spend.  When an agent
spends, the cost rolls up through every ancestor so that enforcement
can happen at any level of the hierarchy.

Design
------
- Each node is an independent :class:`BudgetNode` with a limit, spent
  counter, and list of children.
- All mutations are protected by a single root-level lock.  This is safe
  for typical org-scale trees (hundreds of nodes, not millions).
- ``record_spend`` walks the path from the target node up to the root
  and deducts from each node.
- ``check_spend`` verifies that no node on the path would be exceeded.

Usage
-----
::

    from agent_energy_budget.hierarchy import HierarchicalBudget

    budget = HierarchicalBudget(root_id="org", root_limit=1000.0)
    budget.add_node("eng_team", parent_id="org", limit=300.0)
    budget.add_node("agent_1", parent_id="eng_team", limit=50.0)

    ok, reason = budget.check_spend("agent_1", 10.0)
    if ok:
        budget.record_spend("agent_1", 10.0)

    print(budget.node_status("org").utilisation_pct)
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class HierarchyConfig:
    """Configuration options for the hierarchy.

    Parameters
    ----------
    allow_child_to_exceed_parent:
        If ``True``, child limits are not constrained to be <= parent
        limits (but spend still rolls up).  Default ``False``.
    rollup_mode:
        ``"strict"`` — enforce parent limits; ``"advisory"`` — roll up
        but do not block when parent is exceeded.
    """

    allow_child_to_exceed_parent: bool = False
    rollup_mode: str = "strict"  # "strict" | "advisory"

    def __post_init__(self) -> None:
        valid_modes = {"strict", "advisory"}
        if self.rollup_mode not in valid_modes:
            raise ValueError(
                f"rollup_mode must be one of {sorted(valid_modes)}, "
                f"got {self.rollup_mode!r}."
            )


# ---------------------------------------------------------------------------
# BudgetNode
# ---------------------------------------------------------------------------


@dataclass
class BudgetNode:
    """A single node in the budget hierarchy.

    Parameters
    ----------
    node_id:
        Unique identifier for this node.
    limit_usd:
        Spending cap for this node.  Spending is checked against this
        limit during ``check_spend``.
    label:
        Human-readable label (e.g. department name).
    """

    node_id: str
    limit_usd: float
    label: str = ""
    _spent_usd: float = field(default=0.0, init=False, repr=False)
    _children: list[str] = field(default_factory=list, init=False, repr=False)
    _parent_id: str | None = field(default=None, init=False, repr=False)

    @property
    def spent_usd(self) -> float:
        """Confirmed spend for this node (including all child spend)."""
        return self._spent_usd

    @property
    def remaining_usd(self) -> float:
        """Remaining budget for this node."""
        return max(0.0, self.limit_usd - self._spent_usd)

    @property
    def utilisation_pct(self) -> float:
        """Percentage of budget consumed (0–100+)."""
        if self.limit_usd <= 0:
            return 0.0
        return round(self._spent_usd / self.limit_usd * 100.0, 4)

    @property
    def is_exhausted(self) -> bool:
        """Return True when remaining budget is zero or negative."""
        return self._spent_usd >= self.limit_usd


@dataclass(frozen=True)
class NodeStatus:
    """Immutable point-in-time snapshot of a BudgetNode.

    Parameters
    ----------
    node_id:
        Node identifier.
    label:
        Human-readable label.
    limit_usd:
        Budget cap.
    spent_usd:
        Confirmed spend.
    remaining_usd:
        Remaining budget.
    utilisation_pct:
        Percentage consumed.
    is_exhausted:
        True when fully consumed.
    child_count:
        Number of direct children.
    parent_id:
        Parent node id (``None`` for root).
    """

    node_id: str
    label: str
    limit_usd: float
    spent_usd: float
    remaining_usd: float
    utilisation_pct: float
    is_exhausted: bool
    child_count: int
    parent_id: str | None


# ---------------------------------------------------------------------------
# HierarchicalBudget
# ---------------------------------------------------------------------------


class HierarchicalBudget:
    """Org → team → agent budget tree with roll-up enforcement.

    All public methods are thread-safe (protected by a single lock).

    Parameters
    ----------
    root_id:
        Identifier for the root (organisation) node.
    root_limit:
        Total spending cap for the organisation.
    root_label:
        Human-readable label for the root node.
    config:
        Optional :class:`HierarchyConfig`.

    Example
    -------
    ::

        budget = HierarchicalBudget(root_id="acme", root_limit=10_000.0)
        budget.add_node("eng", parent_id="acme", limit=4_000.0)
        budget.add_node("agent_a", parent_id="eng", limit=500.0)
        ok, _ = budget.check_spend("agent_a", 100.0)
        budget.record_spend("agent_a", 100.0)
    """

    def __init__(
        self,
        root_id: str,
        root_limit: float,
        root_label: str = "",
        config: HierarchyConfig | None = None,
    ) -> None:
        if root_limit <= 0:
            raise ValueError(f"root_limit must be positive, got {root_limit!r}.")
        self._config = config or HierarchyConfig()
        self._lock = threading.Lock()
        self._nodes: dict[str, BudgetNode] = {}
        root = BudgetNode(node_id=root_id, limit_usd=root_limit, label=root_label or root_id)
        self._nodes[root_id] = root
        self._root_id = root_id

    # ------------------------------------------------------------------
    # Tree construction
    # ------------------------------------------------------------------

    def add_node(
        self,
        node_id: str,
        parent_id: str,
        limit: float = 0.0,
        label: str = "",
    ) -> BudgetNode:
        """Add a child budget node under *parent_id*.

        Parameters
        ----------
        node_id:
            Unique identifier for the new node.
        parent_id:
            The parent node identifier (must already exist).
        limit:
            Budget cap for the new node in USD.
        label:
            Human-readable name.

        Returns
        -------
        BudgetNode
            The newly created node.

        Raises
        ------
        KeyError
            If *parent_id* does not exist.
        ValueError
            If *node_id* already exists, or *limit* exceeds the
            parent limit and ``allow_child_to_exceed_parent`` is False.
        """
        if limit <= 0:
            raise ValueError(f"limit_usd must be positive, got {limit!r}.")
        with self._lock:
            if parent_id not in self._nodes:
                raise KeyError(f"Parent node '{parent_id}' does not exist.")
            if node_id in self._nodes:
                raise ValueError(f"Node '{node_id}' already exists.")

            parent = self._nodes[parent_id]
            if (
                not self._config.allow_child_to_exceed_parent
                and limit > parent.limit_usd
            ):
                raise ValueError(
                    f"Child limit ${limit} exceeds parent '{parent_id}' "
                    f"limit ${parent.limit_usd}. Set allow_child_to_exceed_parent=True "
                    f"to override."
                )

            node = BudgetNode(
                node_id=node_id,
                limit_usd=limit,
                label=label or node_id,
            )
            node._parent_id = parent_id
            parent._children.append(node_id)
            self._nodes[node_id] = node

        return node

    # ------------------------------------------------------------------
    # Budget operations
    # ------------------------------------------------------------------

    def check_spend(
        self, node_id: str, amount_usd: float
    ) -> tuple[bool, str]:
        """Check whether *amount_usd* can be spent at *node_id*.

        Verifies that *amount_usd* does not exceed the remaining budget
        of *node_id* or any of its ancestors.

        Parameters
        ----------
        node_id:
            Target node for the spend check.
        amount_usd:
            Amount to check in USD.

        Returns
        -------
        tuple[bool, str]
            ``(allowed, reason)`` — reason is empty when allowed.

        Raises
        ------
        KeyError
            If *node_id* does not exist.
        ValueError
            If *amount_usd* is negative.
        """
        if amount_usd < 0:
            raise ValueError(f"amount_usd must be non-negative, got {amount_usd!r}.")
        with self._lock:
            if node_id not in self._nodes:
                raise KeyError(f"Node '{node_id}' does not exist.")

            if self._config.rollup_mode == "advisory":
                return True, ""

            path = self._ancestor_path(node_id)
            for ancestor_id in path:
                ancestor = self._nodes[ancestor_id]
                if ancestor.remaining_usd < amount_usd:
                    return (
                        False,
                        f"Budget exceeded at node '{ancestor_id}': "
                        f"remaining ${ancestor.remaining_usd:.6f} < "
                        f"requested ${amount_usd:.6f}.",
                    )
        return True, ""

    def record_spend(self, node_id: str, amount_usd: float) -> None:
        """Record *amount_usd* of spend at *node_id* and roll up to all ancestors.

        Parameters
        ----------
        node_id:
            Target node for the spend.
        amount_usd:
            Amount in USD.  Must be non-negative.

        Raises
        ------
        KeyError
            If *node_id* does not exist.
        ValueError
            If *amount_usd* is negative.
        """
        if amount_usd < 0:
            raise ValueError(f"amount_usd must be non-negative, got {amount_usd!r}.")
        with self._lock:
            if node_id not in self._nodes:
                raise KeyError(f"Node '{node_id}' does not exist.")

            path = self._ancestor_path(node_id)
            for ancestor_id in path:
                self._nodes[ancestor_id]._spent_usd += amount_usd

    def reset_node(self, node_id: str) -> None:
        """Reset the spend counter for a single node (does not affect ancestors/children).

        Parameters
        ----------
        node_id:
            Node to reset.

        Raises
        ------
        KeyError
            If *node_id* does not exist.
        """
        with self._lock:
            if node_id not in self._nodes:
                raise KeyError(f"Node '{node_id}' does not exist.")
            self._nodes[node_id]._spent_usd = 0.0

    def reset_all(self) -> None:
        """Reset spend counters for all nodes in the hierarchy."""
        with self._lock:
            for node in self._nodes.values():
                node._spent_usd = 0.0

    # ------------------------------------------------------------------
    # Query API
    # ------------------------------------------------------------------

    def node_status(self, node_id: str) -> NodeStatus:
        """Return a snapshot of the current state of *node_id*.

        Parameters
        ----------
        node_id:
            Target node.

        Returns
        -------
        NodeStatus

        Raises
        ------
        KeyError
            If *node_id* does not exist.
        """
        with self._lock:
            if node_id not in self._nodes:
                raise KeyError(f"Node '{node_id}' does not exist.")
            node = self._nodes[node_id]
            return NodeStatus(
                node_id=node.node_id,
                label=node.label,
                limit_usd=node.limit_usd,
                spent_usd=node.spent_usd,
                remaining_usd=node.remaining_usd,
                utilisation_pct=node.utilisation_pct,
                is_exhausted=node.is_exhausted,
                child_count=len(node._children),
                parent_id=node._parent_id,
            )

    def list_nodes(self) -> list[NodeStatus]:
        """Return status snapshots for all nodes, sorted by node_id."""
        with self._lock:
            return [
                NodeStatus(
                    node_id=node.node_id,
                    label=node.label,
                    limit_usd=node.limit_usd,
                    spent_usd=node.spent_usd,
                    remaining_usd=node.remaining_usd,
                    utilisation_pct=node.utilisation_pct,
                    is_exhausted=node.is_exhausted,
                    child_count=len(node._children),
                    parent_id=node._parent_id,
                )
                for node in sorted(self._nodes.values(), key=lambda n: n.node_id)
            ]

    def children(self, node_id: str) -> list[str]:
        """Return the direct child IDs of *node_id* in insertion order.

        Raises
        ------
        KeyError
            If *node_id* does not exist.
        """
        with self._lock:
            if node_id not in self._nodes:
                raise KeyError(f"Node '{node_id}' does not exist.")
            return list(self._nodes[node_id]._children)

    def root_id(self) -> str:
        """Return the root node identifier."""
        return self._root_id

    def __contains__(self, node_id: object) -> bool:
        """Return True if *node_id* is registered in the hierarchy."""
        return node_id in self._nodes

    def __len__(self) -> int:
        """Return the total number of nodes including the root."""
        return len(self._nodes)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _ancestor_path(self, node_id: str) -> list[str]:
        """Return list of node IDs from *node_id* up to (and including) the root."""
        path: list[str] = []
        current_id: str | None = node_id
        while current_id is not None:
            path.append(current_id)
            node = self._nodes[current_id]
            current_id = node._parent_id
        return path
