from __future__ import annotations

import numpy as np

from kernel.autograd.function import Function
from kernel.core.tensor import Tensor
from kernel.nn.module import Module


class CrossEntropy(Function):
    @staticmethod
    def forward(ctx, logits, targets):
        from kernel.autograd.ops.loss_ops import _get_xp
        xp = _get_xp(logits)

        targets = xp.asarray(targets).astype(np.int64).reshape(-1)

        shifted = logits - xp.max(logits, axis=1, keepdims=True)
        exp_scores = xp.exp(shifted)
        probs = exp_scores / exp_scores.sum(axis=1, keepdims=True)

        batch_size = logits.shape[0]
        loss = -xp.log(probs[xp.arange(batch_size), targets] + 1e-12).mean()

        ctx.save_for_backward(probs, targets)
        ctx.meta["batch_size"] = batch_size
        return xp.array(loss)

    @staticmethod
    def backward(ctx, grad_output):
        from kernel.autograd.ops.loss_ops import _get_xp
        probs, targets = ctx.saved_tensors
        xp = _get_xp(probs)
        batch_size = ctx.meta["batch_size"]

        grad_logits = probs.copy()
        grad_logits[xp.arange(batch_size), targets] -= 1.0
        grad_logits = grad_logits / batch_size
        grad_logits = grad_logits * grad_output

        return grad_logits, None


def cross_entropy(logits: Tensor, targets: Tensor | np.ndarray) -> Tensor:
    if isinstance(targets, Tensor):
        targets = targets.data
    return CrossEntropy.apply(logits, targets)


class CrossEntropyLoss(Module):
    def forward(self, logits: Tensor, targets: Tensor | np.ndarray) -> Tensor:
        return cross_entropy(logits, targets)