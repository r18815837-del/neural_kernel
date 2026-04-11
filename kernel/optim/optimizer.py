from __future__ import annotations


class Optimizer:
    def __init__(self, params, lr: float = 1e-3):
        self.params = list(params)
        self.lr = lr

    def zero_grad(self) -> None:
        for param in self.params:
            param.grad = None

    def step(self) -> None:
        raise NotImplementedError
