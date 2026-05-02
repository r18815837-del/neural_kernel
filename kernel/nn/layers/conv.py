from __future__ import annotations

import numpy as np

from kernel.core.tensor import Tensor
from kernel.nn.module import Module
from kernel.autograd.ops.conv_ops import conv2d


def _pair(v: int | tuple[int, int]) -> tuple[int, int]:
    if isinstance(v, int):
        return (v, v)
    if isinstance(v, tuple) and len(v) == 2:
        return v
    raise ValueError(f"Expected int or tuple of length 2, got {v}")


class Conv2d(Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int],
        stride: int | tuple[int, int] = 1,
        padding: int | tuple[int, int] = 0,
        bias: bool = True,
    ):
        super().__init__()

        kh, kw = _pair(kernel_size)
        sh, sw = _pair(stride)
        ph, pw = _pair(padding)

        limit = np.sqrt(6.0 / (in_channels * kh * kw + out_channels * kh * kw))

        self.weight = Tensor(
            np.random.uniform(
                -limit,
                limit,
                size=(out_channels, in_channels, kh, kw),
            ),
            requires_grad=True,
        )

        if bias:
            self.bias = Tensor(
                np.zeros(out_channels, dtype=np.float32),
                requires_grad=True,
            )
        else:
            self.bias = None

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kh, kw)
        self.stride = (sh, sw)
        self.padding = (ph, pw)

    def forward(self, x: Tensor) -> Tensor:
        return conv2d(
            x,
            self.weight,
            self.bias,
            stride=self.stride,
            padding=self.padding,
        )