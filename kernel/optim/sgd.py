from __future__ import annotations

from kernel.optim.optimizer import Optimizer


class SGD(Optimizer):
    def step(self) -> None:
        for param in self.params:
            if param.grad is None:
                continue
            param.data -= self.lr * param.grad