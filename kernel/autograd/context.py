from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Tuple

@dataclass
class Context:
    """Execution context for a differentiable operation.

    Stores the parent tensors participating in the forward pass, tensors saved
    for backward, and arbitrary metadata needed by the backward function.
    """

    parents: Tuple[Any, ...] = field(default_factory=tuple)
    saved_tensors: Tuple[Any, ...] = field(default_factory=tuple)
    meta: Dict[str, Any] = field(default_factory=dict)

    def save_for_backward(self, *tensors: Any) -> None:
        self.saved_tensors = tuple(tensors)

    def clear_saved_tensors(self) -> None:
        self.saved_tensors = tuple()

    @property
    def has_saved_tensors(self) -> bool:
        return len(self.saved_tensors) > 0