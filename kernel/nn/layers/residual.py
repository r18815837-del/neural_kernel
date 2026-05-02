from __future__ import annotations

from kernel.nn.module import Module
from kernel.nn.containers import Sequential
from kernel.nn.activations import ReLU, Identity
from kernel.nn.layers.conv import Conv2d
from kernel.nn.normalization import BatchNorm2d


class ResidualBlock(Module):
    def __init__(self, channels: int):
        super().__init__()

        self.main = Sequential(
            Conv2d(channels, channels, 3, padding=1),
            BatchNorm2d(channels),
            ReLU(),
            Conv2d(channels, channels, 3, padding=1),
            BatchNorm2d(channels),
        )
        self.skip = Identity()
        self.act = ReLU()

    def forward(self, x):
        out = self.main(x)
        out = out + self.skip(x)
        return self.act(out)