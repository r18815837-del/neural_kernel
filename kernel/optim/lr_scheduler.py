from __future__ import annotations


class StepLR:
    def __init__(self, optimizer, step_size: int, gamma: float = 0.1):
        if step_size <= 0:
            raise ValueError(f"step_size must be > 0, got {step_size}")
        if gamma <= 0.0:
            raise ValueError(f"gamma must be > 0, got {gamma}")

        self.optimizer = optimizer
        self.step_size = step_size
        self.gamma = gamma
        self.last_epoch = 0

    def get_lr(self) -> float:
        return self.optimizer.lr

    def step(self) -> None:
        self.last_epoch += 1

        if self.last_epoch % self.step_size == 0:
            self.optimizer.lr *= self.gamma