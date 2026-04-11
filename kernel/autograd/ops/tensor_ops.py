from __future__ import annotations

import numpy as np

from kernel.autograd.function import Function


class Reshape(Function):
    @staticmethod
    def forward(ctx, x, shape):
        ctx.meta["input_shape"] = x.shape
        return x.reshape(shape)

    @staticmethod
    def backward(ctx, grad_output):
        input_shape = ctx.meta["input_shape"]
        return grad_output.reshape(input_shape), None


def reshape(x, shape):
    return Reshape.apply(x, shape)