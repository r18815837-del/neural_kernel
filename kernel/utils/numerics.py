from __future__ import annotations

from typing import Any

import numpy as np


def _to_backend_array(x: Any, xp=None):
    if xp is None:
        return x
    return xp.asarray(x)


def safe_div(numerator, denominator, eps: float = 1e-12, xp=None):
    """Numerically safe division."""
    if eps <= 0:
        raise ValueError("eps must be positive")

    if xp is None:
        xp = np

    numerator = xp.asarray(numerator)
    denominator = xp.asarray(denominator)

    safe_denominator = xp.where(xp.abs(denominator) < eps, eps, denominator)
    return numerator / safe_denominator


def safe_log(x, eps: float = 1e-12, xp=None):
    """Numerically safe logarithm."""
    if eps <= 0:
        raise ValueError("eps must be positive")

    if xp is None:
        xp = np

    x = xp.asarray(x)
    return xp.log(xp.maximum(x, eps))


def stable_softmax(x, axis: int = -1, xp=None):
    """Numerically stable softmax."""
    if xp is None:
        xp = np

    x = xp.asarray(x)
    x_max = xp.max(x, axis=axis, keepdims=True)
    shifted = x - x_max
    exp_x = xp.exp(shifted)
    denom = xp.sum(exp_x, axis=axis, keepdims=True)
    return exp_x / denom


def stable_log_softmax(x, axis: int = -1, xp=None):
    """Numerically stable log-softmax."""
    if xp is None:
        xp = np

    x = xp.asarray(x)
    x_max = xp.max(x, axis=axis, keepdims=True)
    shifted = x - x_max
    logsumexp = xp.log(xp.sum(xp.exp(shifted), axis=axis, keepdims=True))
    return shifted - logsumexp


def has_nan(x, xp=None) -> bool:
    if xp is None:
        xp = np
    x = xp.asarray(x)
    return bool(xp.any(xp.isnan(x)))


def has_inf(x, xp=None) -> bool:
    if xp is None:
        xp = np
    x = xp.asarray(x)
    return bool(xp.any(xp.isinf(x)))


def has_nan_or_inf(x, xp=None) -> bool:
    if xp is None:
        xp = np
    x = xp.asarray(x)
    return bool(xp.any(xp.isnan(x)) or xp.any(xp.isinf(x)))


def clamp(x, min_value=None, max_value=None, xp=None):
    """Clamp values into [min_value, max_value]."""
    if xp is None:
        xp = np

    x = xp.asarray(x)

    if min_value is not None and max_value is not None and min_value > max_value:
        raise ValueError("min_value cannot be greater than max_value")

    if min_value is not None:
        x = xp.maximum(x, min_value)
    if max_value is not None:
        x = xp.minimum(x, max_value)

    return x