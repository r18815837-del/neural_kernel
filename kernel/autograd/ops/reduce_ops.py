from __future__ import annotations

import numpy as np

from kernel.autograd.function import Function

try:
    import cupy as cp
except Exception:
    cp = None


def _get_xp(x):
    if cp is not None and isinstance(x, cp.ndarray):
        return cp
    return np


def _normalize_axes(axis, ndim):
    if axis is None:
        return None

    if isinstance(axis, int):
        axis = (axis,)
    else:
        axis = tuple(axis)

    normalized = []
    for ax in axis:
        if ax < 0:
            ax += ndim
        if ax < 0 or ax >= ndim:
            raise ValueError(f"Axis out of range: axis={ax}, ndim={ndim}")
        normalized.append(ax)

    # убрать дубли, сохранить порядок
    seen = set()
    result = []
    for ax in normalized:
        if ax not in seen:
            seen.add(ax)
            result.append(ax)

    return tuple(result)


def _expand_grad_to_input_shape(grad_output, input_shape, axes, keepdims):
    xp = _get_xp(grad_output)

    if axes is None:
        return xp.ones(input_shape, dtype=grad_output.dtype) * grad_output

    if not keepdims:
        shape = list(grad_output.shape)
        for ax in sorted(axes):
            shape.insert(ax, 1)
        grad_output = grad_output.reshape(shape)

    return xp.ones(input_shape, dtype=grad_output.dtype) * grad_output


class Sum(Function):
    @staticmethod
    def forward(ctx, a, axis=None, keepdims=False):
        xp = _get_xp(a)

        axes = _normalize_axes(axis, a.ndim)
        ctx.meta["shape"] = a.shape
        ctx.meta["axes"] = axes
        ctx.meta["keepdims"] = keepdims

        return xp.asarray(a.sum(axis=axes, keepdims=keepdims))

    @staticmethod
    def backward(ctx, grad_output):
        input_shape = ctx.meta["shape"]
        axes = ctx.meta["axes"]
        keepdims = ctx.meta["keepdims"]

        grad = _expand_grad_to_input_shape(
            grad_output,
            input_shape,
            axes,
            keepdims,
        )
        return grad, None, None


class Mean(Function):
    @staticmethod
    def forward(ctx, a, axis=None, keepdims=False):
        xp = _get_xp(a)

        axes = _normalize_axes(axis, a.ndim)
        ctx.meta["shape"] = a.shape
        ctx.meta["axes"] = axes
        ctx.meta["keepdims"] = keepdims

        if axes is None:
            count = a.size
        else:
            count = 1
            for ax in axes:
                count *= a.shape[ax]
        ctx.meta["count"] = count

        return xp.asarray(a.mean(axis=axes, keepdims=keepdims))

    @staticmethod
    def backward(ctx, grad_output):
        input_shape = ctx.meta["shape"]
        axes = ctx.meta["axes"]
        keepdims = ctx.meta["keepdims"]
        count = ctx.meta["count"]

        grad = _expand_grad_to_input_shape(
            grad_output,
            input_shape,
            axes,
            keepdims,
        )
        return grad / count, None, None


def sum(a, axis=None, keepdims=False):
    return Sum.apply(a, axis, keepdims)


def mean(a, axis=None, keepdims=False):
    return Mean.apply(a, axis, keepdims)