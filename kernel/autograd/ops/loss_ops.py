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


class CrossEntropy(Function):
    @staticmethod
    def forward(ctx, logits, targets):
        xp = _get_xp(logits)

        if logits.ndim != 2:
            raise ValueError(
                f"CrossEntropy expects logits of shape (B, C), got {logits.shape}"
            )

        if targets.ndim != 1:
            raise ValueError(
                f"CrossEntropy expects targets of shape (B,), got {targets.shape}"
            )

        batch_size, num_classes = logits.shape

        if targets.shape[0] != batch_size:
            raise ValueError(
                f"Batch mismatch: logits batch={batch_size}, targets batch={targets.shape[0]}"
            )

        if targets.dtype.kind not in {"i", "u"}:
            targets = targets.astype(np.int64 if xp is np else cp.int64)

        if targets.min() < 0 or targets.max() >= num_classes:
            raise ValueError(
                f"Target indices out of range [0, {num_classes}), "
                f"got min={targets.min()}, max={targets.max()}"
            )

        shifted = logits - xp.max(logits, axis=1, keepdims=True)
        exp = xp.exp(shifted)
        probs = exp / xp.sum(exp, axis=1, keepdims=True)

        batch_indices = xp.arange(batch_size)
        correct_probs = probs[batch_indices, targets]
        loss = -xp.log(correct_probs + 1e-12).mean()

        ctx.save_for_backward(probs, targets)
        ctx.meta["batch_size"] = batch_size

        return xp.asarray(loss)

    @staticmethod
    def backward(ctx, grad_output):
        xp = _get_xp(grad_output)

        probs, targets = ctx.saved_tensors
        batch_size = ctx.meta["batch_size"]

        grad_logits = probs.copy()
        batch_indices = xp.arange(batch_size)
        grad_logits[batch_indices, targets] -= 1.0
        grad_logits = grad_logits / batch_size

        grad_logits = grad_logits * grad_output
        return grad_logits, None


def cross_entropy(logits, targets):
    return CrossEntropy.apply(logits, targets)