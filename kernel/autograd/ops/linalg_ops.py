from __future__ import annotations

from kernel.autograd.function import Function


class MatMul(Function):
    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)
        return a @ b

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        return grad_output @ b.T, a.T @ grad_output


def matmul(a, b):
    return MatMul.apply(a, b)
