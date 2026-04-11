from __future__ import annotations

import numpy as np

from kernel.core.tensor import Tensor
from kernel.nn.module import Module


class Linear(Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        limit = np.sqrt(6.0 / (in_features + out_features))
        weight = np.random.uniform(-limit, limit, size=(in_features, out_features))
        bias = np.zeros((1, out_features))
        self.weight = Tensor(weight, requires_grad=True)
        self.bias = Tensor(bias, requires_grad=True)

    def forward(self, x: Tensor) -> Tensor:
        return (x @ self.weight) + self.bias
