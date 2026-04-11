from __future__ import annotations

from kernel.core.tensor import Tensor
from kernel.nn.module import Module


class MSELoss(Module):
    def forward(self, prediction: Tensor, target: Tensor) -> Tensor:
        diff = prediction - target
        return (diff * diff).mean()
