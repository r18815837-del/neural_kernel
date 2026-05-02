from __future__ import annotations

from kernel.nn.module import Module


class Flatten(Module):
    def __init__(self, start_dim: int = 1):
        super().__init__()
        self.start_dim = start_dim

    def forward(self, x):
        if self.start_dim != 1:
            raise NotImplementedError("Only start_dim=1 is supported for now")

        batch_size = x.data.shape[0]
        return x.reshape(batch_size, -1)