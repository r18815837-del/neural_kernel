from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Set, Tuple


@dataclass
class GraphNode:
    """A lightweight autograd graph node.

    Stores a tensor-like value and links to its parent nodes.
    This layer is intentionally minimal so it can be introduced
    without breaking the current Tensor.backward() implementation.
    """

    value: Any
    parents: Tuple[Any, ...] = field(default_factory=tuple)

    @property
    def is_leaf(self) -> bool:
        return len(self.parents) == 0


def iter_parents(node: Any) -> Tuple[Any, ...]:
    """Return parent tensors/nodes for a graph node-like object."""
    ctx = getattr(node, "_ctx", None)
    if ctx is None:
        return tuple()
    return tuple(ctx.parents)


def topological_sort(root: Any) -> List[Any]:
    """Build reverse-autodiff topological order from a root tensor.

    Returns nodes in forward order, so callers can iterate with
    reversed(order) for backward propagation.
    """
    topo: List[Any] = []
    visited: Set[int] = set()

    def build(node: Any) -> None:
        node_id = id(node)
        if node_id in visited:
            return
        visited.add(node_id)

        for parent in iter_parents(node):
            build(parent)

        topo.append(node)

    build(root)
    return topo