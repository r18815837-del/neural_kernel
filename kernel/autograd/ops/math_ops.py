from __future__ import annotations

import numpy as np

from kernel.autograd.function import Function


def _unbroadcast(grad, shape):
    if grad.shape == shape:
        return grad

    while len(grad.shape) > len(shape):
        grad = grad.sum(axis=0)

    for axis, size in enumerate(shape):
        if size == 1 and grad.shape[axis] != 1:
            grad = grad.sum(axis=axis, keepdims=True)

    return grad.reshape(shape)


class Add(Function):
    @staticmethod
    def forward(ctx, a, b):
        ctx.meta["a_shape"] = np.shape(a)
        ctx.meta["b_shape"] = np.shape(b)
        return a + b

    @staticmethod
    def backward(ctx, grad_output):
        ga = _unbroadcast(grad_output, ctx.meta["a_shape"])
        gb = _unbroadcast(grad_output, ctx.meta["b_shape"])
        return ga, gb


class Sub(Function):
    @staticmethod
    def forward(ctx, a, b):
        ctx.meta["a_shape"] = np.shape(a)
        ctx.meta["b_shape"] = np.shape(b)
        return a - b

    @staticmethod
    def backward(ctx, grad_output):
        ga = _unbroadcast(grad_output, ctx.meta["a_shape"])
        gb = _unbroadcast(-grad_output, ctx.meta["b_shape"])
        return ga, gb


class Mul(Function):
    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)
        ctx.meta["a_shape"] = np.shape(a)
        ctx.meta["b_shape"] = np.shape(b)
        return a * b

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        ga = _unbroadcast(grad_output * b, ctx.meta["a_shape"])
        gb = _unbroadcast(grad_output * a, ctx.meta["b_shape"])
        return ga, gb


class Div(Function):
    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)
        ctx.meta["a_shape"] = np.shape(a)
        ctx.meta["b_shape"] = np.shape(b)
        return a / b

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        ga = _unbroadcast(grad_output / b, ctx.meta["a_shape"])
        gb = _unbroadcast(grad_output * (-a / (b ** 2)), ctx.meta["b_shape"])
        return ga, gb


class ReLU(Function):
    @staticmethod
    def forward(ctx, a):
        ctx.save_for_backward(a)
        return np.maximum(a, 0)

    @staticmethod
    def backward(ctx, grad_output):
        (a,) = ctx.saved_tensors
        return grad_output * (a > 0).astype(a.dtype)


class Sigmoid(Function):
    @staticmethod
    def forward(ctx, x):
        out = 1.0 / (1.0 + np.exp(-x))
        ctx.save_for_backward(out)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        (out,) = ctx.saved_tensors
        return grad_output * out * (1.0 - out)

class Sqrt(Function):
    @staticmethod
    def forward(ctx, x):
        out = np.sqrt(x)
        ctx.save_for_backward(out)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        (out,) = ctx.saved_tensors
        return grad_output * (0.5 / out)
class LeakyReLU(Function):
    @staticmethod
    def forward(ctx, a, negative_slope):
        ctx.save_for_backward(a)
        ctx.meta["negative_slope"] = negative_slope
        return np.where(a > 0, a, negative_slope * a)

    @staticmethod
    def backward(ctx, grad_output):
        (a,) = ctx.saved_tensors
        negative_slope = ctx.meta["negative_slope"]
        grad = np.where(a > 0, 1.0, negative_slope).astype(a.dtype)
        return grad_output * grad, None

def sqrt(x):
    return Sqrt.apply(x)

def add(a, b):
    return Add.apply(a, b)


def sub(a, b):
    return Sub.apply(a, b)


def mul(a, b):
    return Mul.apply(a, b)


def div(a, b):
    return Div.apply(a, b)


def relu(a):
    return ReLU.apply(a)


def sigmoid(x):
    return Sigmoid.apply(x)

class Dropout(Function):
    @staticmethod
    def forward(ctx, x, mask, scale):
        ctx.save_for_backward(mask)
        ctx.meta["scale"] = scale
        return x * mask * scale

    @staticmethod
    def backward(ctx, grad_output):
        (mask,) = ctx.saved_tensors
        scale = ctx.meta["scale"]
        return grad_output * mask * scale, None, None
class Tanh(Function):
    @staticmethod
    def forward(ctx, x):
        out = np.tanh(x)
        ctx.save_for_backward(out)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        (out,) = ctx.saved_tensors
        return grad_output * (1.0 - out ** 2)


def tanh(x):
    return Tanh.apply(x)

def dropout(x, p: float, training: bool = True):
    if not training or p == 0.0:
        return x

    if not 0.0 <= p < 1.0:
        raise ValueError(f"dropout probability must be in [0, 1), got {p}")

    keep_prob = 1.0 - p
    mask = (np.random.rand(*x.data.shape) < keep_prob).astype(x.data.dtype)
    scale = 1.0 / keep_prob

    return Dropout.apply(x, mask, scale)

def leaky_relu(a, negative_slope: float = 0.01):
    return LeakyReLU.apply(a, negative_slope)

def layer_norm(x, normalized_shape, eps=1e-5, weight=None, bias=None):
    return LayerNormFunction.apply(x, normalized_shape, eps, weight, bias)

class LayerNormFunction(Function):
    @staticmethod
    def forward(ctx, x, normalized_shape, eps, weight, bias):
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)

        normalized_shape = tuple(normalized_shape)
        axes = tuple(range(x.ndim - len(normalized_shape), x.ndim))

        mean = x.mean(axis=axes, keepdims=True)
        var = x.var(axis=axes, keepdims=True)
        inv_std = 1.0 / np.sqrt(var + eps)
        x_hat = (x - mean) * inv_std

        out = x_hat
        if weight is not None:
            out = out * weight
        if bias is not None:
            out = out + bias

        ctx.save_for_backward(x_hat, inv_std, weight)
        ctx.meta["axes"] = axes
        ctx.meta["normalized_shape"] = normalized_shape
        ctx.meta["affine"] = weight is not None and bias is not None

        return out

    @staticmethod
    def backward(ctx, grad_output):
        x_hat, inv_std, weight = ctx.saved_tensors
        axes = ctx.meta["axes"]
        normalized_shape = ctx.meta["normalized_shape"]
        affine = ctx.meta["affine"]

        dxhat = grad_output
        dgamma = None
        dbeta = None

        if affine:
            reduce_axes = tuple(range(grad_output.ndim - len(normalized_shape)))
            dgamma = np.sum(grad_output * x_hat, axis=reduce_axes)
            dbeta = np.sum(grad_output, axis=reduce_axes)

            dxhat = grad_output * weight

        m = 1
        for ax in axes:
            m *= grad_output.shape[ax]

        sum_dxhat = np.sum(dxhat, axis=axes, keepdims=True)
        sum_dxhat_xhat = np.sum(dxhat * x_hat, axis=axes, keepdims=True)

        dx = (1.0 / m) * inv_std * (
                m * dxhat - sum_dxhat - x_hat * sum_dxhat_xhat
        )

        if affine:
            return dx, dgamma, dbeta
        return dx

class Softmax(Function):
    @staticmethod
    def forward(ctx, x, axis):
        shifted = x - np.max(x, axis=axis, keepdims=True)
        exp = np.exp(shifted)
        out = exp / np.sum(exp, axis=axis, keepdims=True)
        ctx.save_for_backward(out)
        ctx.meta["axis"] = axis
        return out

    @staticmethod
    def backward(ctx, grad_output):
        (out,) = ctx.saved_tensors
        axis = ctx.meta["axis"]

        dot = np.sum(grad_output * out, axis=axis, keepdims=True)
        grad_x = out * (grad_output - dot)
        return grad_x, None


def softmax(x, axis: int = -1):
    return Softmax.apply(x, axis)