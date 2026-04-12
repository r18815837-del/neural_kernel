from __future__ import annotations

from kernel.autograd.function import Function


def _swap_last_two_axes(x):
    if x.ndim < 2:
        raise ValueError("matmul backward requires tensors with ndim >= 2")

    axes = list(range(x.ndim))
    axes[-1], axes[-2] = axes[-2], axes[-1]
    return x.transpose(tuple(axes))


def _sum_to_shape(grad, shape):
    if grad.shape == shape:
        return grad

    # убираем лишние ведущие batch-оси
    while len(grad.shape) > len(shape):
        grad = grad.sum(axis=0)

    # сводим broadcasted оси
    for axis, size in enumerate(shape):
        if size == 1 and grad.shape[axis] != 1:
            grad = grad.sum(axis=axis, keepdims=True)

    # если shape был без batch-осей, а grad их получил после batched matmul,
    # то после цикла выше shape уже должен совпасть
    if grad.shape != shape:
        # это как раз важный случай типа:
        # grad: (B, Din, Dout), shape: (Din, Dout)
        extra = len(grad.shape) - len(shape)
        if extra > 0:
            axes = tuple(range(extra))
            grad = grad.sum(axis=axes)

    if grad.shape != shape:
        grad = grad.reshape(shape)

    return grad


class MatMul(Function):
    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)
        ctx.meta["a_shape"] = a.shape
        ctx.meta["b_shape"] = b.shape
        return a @ b

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors

        b_t = _swap_last_two_axes(b)
        a_t = _swap_last_two_axes(a)

        grad_a = grad_output @ b_t
        grad_b = a_t @ grad_output

        grad_a = _sum_to_shape(grad_a, ctx.meta["a_shape"])
        grad_b = _sum_to_shape(grad_b, ctx.meta["b_shape"])

        return grad_a, grad_b


def matmul(a, b):
    return MatMul.apply(a, b)