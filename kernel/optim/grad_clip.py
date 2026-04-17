from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np


def _iter_params(parameters) -> Iterable:
    for p in parameters:
        if p is None:
            continue
        if not getattr(p, "requires_grad", False):
            continue
        yield p


def clip_grad_value_(parameters, clip_value: float) -> None:
    """In-place clip of gradients by value."""
    if clip_value < 0:
        raise ValueError("clip_value must be non-negative")

    for p in _iter_params(parameters):
        if getattr(p, "grad", None) is None:
            continue

        xp = p.xp
        p.grad = xp.clip(p.grad, -clip_value, clip_value)


def clip_grad_norm_(
    parameters,
    max_norm: float,
    norm_type: float = 2.0,
    eps: float = 1e-12,
) -> float:
    """Clip gradients so that their global norm does not exceed max_norm.

    Returns the total norm before clipping.
    """
    if max_norm <= 0:
        raise ValueError("max_norm must be positive")
    if norm_type <= 0:
        raise ValueError("norm_type must be positive")

    params = list(_iter_params(parameters))
    grads = [p.grad for p in params if getattr(p, "grad", None) is not None]

    if len(grads) == 0:
        return 0.0

    # Compute norm on CPU for backend-agnostic stability/readability
    if norm_type == float("inf"):
        total_norm = max(float(np.max(np.abs(p._backend.to_cpu(p.grad)))) for p in params if p.grad is not None)
    else:
        total = 0.0
        for p in params:
            if p.grad is None:
                continue
            grad_cpu = p._backend.to_cpu(p.grad)
            total += float(np.sum(np.abs(grad_cpu) ** norm_type))
        total_norm = total ** (1.0 / norm_type)

    clip_coef = max_norm / (total_norm + eps)

    if clip_coef < 1.0:
        for p in params:
            if p.grad is None:
                continue
            p.grad = p.grad * clip_coef

    return float(total_norm)