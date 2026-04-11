from __future__ import annotations

import numpy as np

from kernel.optim.optimizer import Optimizer


class Adam(Optimizer):
    def __init__(self, params, lr: float = 1e-3, betas=(0.9, 0.999), eps: float = 1e-8):
        super().__init__(params, lr=lr)
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.t = 0

        self.m = [np.zeros_like(p.data) for p in self.params]
        self.v = [np.zeros_like(p.data) for p in self.params]

    def step(self) -> None:
        self.t += 1

        for i, param in enumerate(self.params):
            if param.grad is None:
                continue

            grad = param.grad

            self.m[i] = self.beta1 * self.m[i] + (1.0 - self.beta1) * grad
            self.v[i] = self.beta2 * self.v[i] + (1.0 - self.beta2) * (grad ** 2)

            m_hat = self.m[i] / (1.0 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1.0 - self.beta2 ** self.t)

            param.data = param.data - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)