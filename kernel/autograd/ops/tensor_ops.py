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


class Reshape(Function):
    @staticmethod
    def forward(ctx, x, shape):
        ctx.meta["input_shape"] = x.shape
        return x.reshape(shape)

    @staticmethod
    def backward(ctx, grad_output):
        input_shape = ctx.meta["input_shape"]
        return grad_output.reshape(input_shape), None


class Transpose(Function):
    @staticmethod
    def forward(ctx, x, axes):
        if isinstance(axes, list):
            axes = tuple(axes)
        elif not isinstance(axes, tuple):
            axes = tuple(axes)

        ctx.meta["axes"] = axes
        return x.transpose(axes)

    @staticmethod
    def backward(ctx, grad_output):
        axes = ctx.meta["axes"]

        inv_axes = [0] * len(axes)
        for i, ax in enumerate(axes):
            inv_axes[ax] = i
        inv_axes = tuple(inv_axes)

        return grad_output.transpose(inv_axes), None


class Unsqueeze(Function):
    @staticmethod
    def forward(ctx, x, axis):
        input_shape = x.shape

        if axis < 0:
            axis += len(input_shape) + 1

        if axis < 0 or axis > len(input_shape):
            raise ValueError(
                f"Axis out of range for unsqueeze: axis={axis}, ndim={len(input_shape)}"
            )

        new_shape = input_shape[:axis] + (1,) + input_shape[axis:]
        ctx.meta["input_shape"] = input_shape
        return x.reshape(new_shape)

    @staticmethod
    def backward(ctx, grad_output):
        input_shape = ctx.meta["input_shape"]
        return grad_output.reshape(input_shape), None


class Squeeze(Function):
    @staticmethod
    def forward(ctx, x, axis=None):
        input_shape = x.shape

        if axis is None:
            new_shape = tuple(dim for dim in input_shape if dim != 1)
            if len(new_shape) == 0:
                new_shape = ()
        else:
            if axis < 0:
                axis += len(input_shape)

            if axis < 0 or axis >= len(input_shape):
                raise ValueError(
                    f"Axis out of range for squeeze: axis={axis}, ndim={len(input_shape)}"
                )

            if input_shape[axis] != 1:
                raise ValueError(
                    f"Cannot squeeze axis {axis} with size {input_shape[axis]}"
                )

            new_shape = input_shape[:axis] + input_shape[axis + 1:]

        ctx.meta["input_shape"] = input_shape
        return x.reshape(new_shape)

    @staticmethod
    def backward(ctx, grad_output):
        input_shape = ctx.meta["input_shape"]
        return grad_output.reshape(input_shape), None


class GetItem(Function):
    @staticmethod
    def forward(ctx, x, index):
        ctx.meta["input_shape"] = x.shape
        ctx.meta["index"] = index
        return x[index]

    @staticmethod
    def backward(ctx, grad_output):
        input_shape = ctx.meta["input_shape"]
        index = ctx.meta["index"]

        xp = _get_xp(grad_output)
        grad = xp.zeros(input_shape, dtype=grad_output.dtype)
        grad[index] = grad_output
        return grad, None


class EmbeddingLookup(Function):
    @staticmethod
    def forward(ctx, weight, indices):
        ctx.meta["weight_shape"] = weight.shape
        ctx.meta["indices"] = indices
        return weight[indices]

    @staticmethod
    def backward(ctx, grad_output):
        input_shape = ctx.meta["weight_shape"]
        indices = ctx.meta["indices"]

        xp = _get_xp(grad_output)
        grad_weight = xp.zeros(input_shape, dtype=grad_output.dtype)
        xp.add.at(grad_weight, indices, grad_output)

        return grad_weight, None


class Concat(Function):
    @staticmethod
    def forward(ctx, *args, axis=0):
        if len(args) == 0:
            raise ValueError("concat expects at least one tensor")

        arrays = args

        rank = arrays[0].ndim
        if axis < 0:
            axis += rank

        if axis < 0 or axis >= rank:
            raise ValueError(
                f"Axis out of range for concat: axis={axis}, ndim={rank}"
            )

        xp = _get_xp(arrays[0])

        base_shape = list(arrays[0].shape)
        sizes = []

        for arr in arrays:
            if arr.ndim != rank:
                raise ValueError(
                    f"All tensors for concat must have same ndim, got {rank} and {arr.ndim}"
                )

            for dim in range(rank):
                if dim == axis:
                    continue
                if arr.shape[dim] != base_shape[dim]:
                    raise ValueError(
                        f"Concat shape mismatch on dim {dim}: "
                        f"expected {base_shape[dim]}, got {arr.shape[dim]}"
                    )

            sizes.append(arr.shape[axis])

        ctx.meta["axis"] = axis
        ctx.meta["sizes"] = tuple(sizes)

        return xp.concatenate(arrays, axis=axis)

    @staticmethod
    def backward(ctx, grad_output):
        axis = ctx.meta["axis"]
        sizes = ctx.meta["sizes"]

        grads = []
        start = 0

        for size in sizes:
            end = start + size
            index = [slice(None)] * grad_output.ndim
            index[axis] = slice(start, end)
            grads.append(grad_output[tuple(index)])
            start = end

        return tuple(grads)


def reshape(x, shape):
    return Reshape.apply(x, shape)


def transpose(x, axes):
    return Transpose.apply(x, axes)


def unsqueeze(x, axis):
    return Unsqueeze.apply(x, axis)


def squeeze(x, axis=None):
    return Squeeze.apply(x, axis)


def getitem(x, index):
    return GetItem.apply(x, index)


def embedding_lookup(weight, indices):
    return EmbeddingLookup.apply(weight, indices)

def concat(tensors, axis=0):
    if len(tensors) == 0:
        raise ValueError("concat expects at least one tensor")
    return Concat.apply(*tensors, axis=axis)