from __future__ import annotations

from kernel.nn.module import Module
from kernel.autograd.ops.pool_ops import (
    maxpool2d,
    avgpool2d,
    adaptive_avgpool2d,
    adaptive_maxpool2d,
)


class MaxPool2d(Module):
    def __init__(self, kernel_size, stride=None):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, x):
        return maxpool2d(x, self.kernel_size, self.stride)


class AvgPool2d(Module):
    def __init__(self, kernel_size, stride=None):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, x):
        return avgpool2d(x, self.kernel_size, self.stride)


class AdaptiveAvgPool2d(Module):
    def __init__(self, output_size):
        super().__init__()
        self.output_size = output_size

    def forward(self, x):
        return adaptive_avgpool2d(x, self.output_size)


class AdaptiveMaxPool2d(Module):
    def __init__(self, output_size):
        super().__init__()
        self.output_size = output_size

    def forward(self, x):
        return adaptive_maxpool2d(x, self.output_size)