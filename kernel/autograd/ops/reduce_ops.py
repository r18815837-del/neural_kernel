from __future__ import annotations

import numpy as np

from kernel.autograd.function import Function


class Sum(Function):
    @staticmethod
    def forward(ctx, a):
        ctx.meta["shape"] = a.shape
        return np.array(a.sum())

    @staticmethod
    def backward(ctx, grad_output):
        return np.ones(ctx.meta["shape"]) * grad_output


class Mean(Function):
    @staticmethod
    def forward(ctx, a):
        ctx.meta["shape"] = a.shape
        ctx.meta["size"] = a.size
        return np.array(a.mean())

    @staticmethod
    def backward(ctx, grad_output):
        return np.ones(ctx.meta["shape"]) * (grad_output / ctx.meta["size"])


def sum(a):
    return Sum.apply(a)


def mean(a):
    return Mean.apply(a)
