from __future__ import annotations

from typing import Any, Iterable

import numpy as np

from .graph import topological_sort


def _accumulate_grad(tensor: Any, grad: Any) -> None:
    """Accumulate gradient into tensor.grad."""
    if grad is None:
        return

    if getattr(tensor, "grad", None) is None:
        tensor.grad = grad
    else:
        tensor.grad = tensor.grad + grad


def backward(root: Any, grad: Any = None) -> None:
    """Run reverse-mode autodiff starting from root.

    Parameters
    ----------
    root:
        Tensor-like object with fields:
        - requires_grad
        - grad
        - _backward
        - _ctx
        - data
    grad:
        Optional upstream gradient. If omitted, a tensor-like scalar gradient
        of ones is created for scalar outputs.
    """
    if not getattr(root, "requires_grad", False):
        return

    if grad is None:
        if getattr(root, "data", None) is None:
            raise RuntimeError("Cannot infer gradient: root has no data.")
        xp = root._backend.xp if hasattr(root, "_backend") else np
        grad = xp.ones_like(root.data)

    # Ensure grad is a raw array, not a Tensor
    if hasattr(grad, "data") and not isinstance(grad, np.ndarray):
        grad = grad.data

    root.grad = grad
    topo = topological_sort(root)

    for node in reversed(topo):
        node_grad = getattr(node, "grad", None)
        if node_grad is None:
            continue

        backward_fn = getattr(node, "_backward", None)
        ctx = getattr(node, "_ctx", None)

        if backward_fn is None or ctx is None:
            continue

        grads = backward_fn(node_grad)
        if grads is None:
            continue

        if not isinstance(grads, tuple):
            grads = (grads,)

        parents: Iterable[Any] = getattr(ctx, "parents", ())
        tensor_arg_indices = ctx.meta.get("tensor_arg_indices", None)

        if tensor_arg_indices is not None and len(tensor_arg_indices) > 0:
            # Map gradients back to tensor arguments using their original positions.
            # backward() returns one gradient per *original* argument (tensors and
            # non-tensors alike), so we pick only those at tensor positions.
            for parent, arg_idx in zip(parents, tensor_arg_indices):
                if arg_idx < len(grads):
                    g = grads[arg_idx]
                    if g is not None:
                        _accumulate_grad(parent, g)
        else:
            # Fallback: pair grads with parents in order
            for parent, g in zip(parents, grads):
                if g is not None:
                    _accumulate_grad(parent, g)