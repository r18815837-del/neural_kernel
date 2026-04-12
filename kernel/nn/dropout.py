from __future__ import annotations

from kernel.core.tensor import Tensor
from kernel.nn.module import Module


class Dropout(Module):
    def __init__(self, p: float = 0.5):
        super().__init__()
        if not 0.0 <= p < 1.0:
            raise ValueError(f"dropout probability must be in [0, 1), got {p}")
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        return x.dropout(p=self.p, training=self.training)