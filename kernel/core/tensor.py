from __future__ import annotations

from typing import Any, Iterable, List, Optional, Set

import numpy as np

from kernel.autograd.ops.linalg_ops import matmul as _matmul
from kernel.autograd.ops.math_ops import add as _add
from kernel.autograd.ops.math_ops import div as _div
from kernel.autograd.ops.math_ops import mul as _mul
from kernel.autograd.ops.math_ops import relu as _relu
from kernel.autograd.ops.math_ops import sigmoid as _sigmoid
from kernel.autograd.ops.math_ops import sub as _sub
from kernel.autograd.ops.reduce_ops import mean as _mean
from kernel.autograd.ops.reduce_ops import sum as _sum


class Tensor:
    __array_priority__ = 1000

    def __init__(self, data: Any, requires_grad: bool = False, dtype=None):
        if isinstance(data, Tensor):
            data = data.data
        self.data = np.array(data, dtype=dtype if dtype is not None else np.float64)
        self.requires_grad = requires_grad
        self.grad: Optional[np.ndarray] = None
        self._ctx = None
        self._backward = None

    @property
    def shape(self):
        return self.data.shape

    @property
    def dtype(self):
        return self.data.dtype

    @property
    def T(self) -> "Tensor":
        return Tensor(self.data.T, requires_grad=self.requires_grad)

    def numpy(self) -> np.ndarray:
        return self.data.copy()

    def reshape(self, *shape):
        from kernel.autograd.ops.tensor_ops import reshape

        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])

        return reshape(self, shape)

    def sqrt(self):
        from kernel.autograd.ops.math_ops import sqrt
        return sqrt(self)

    def dropout(self, p: float = 0.5, training: bool = True):
        from kernel.autograd.ops.math_ops import dropout
        return dropout(self, p=p, training=training)

    def zero_grad(self) -> None:
        self.grad = np.zeros_like(self.data)

    def backward(self, grad: Any = None) -> None:
        if not self.requires_grad:
            raise RuntimeError("Cannot call backward on tensor with requires_grad=False")

        if grad is None:
            if self.data.size != 1:
                raise RuntimeError("grad must be specified for non-scalar tensor")
            grad = np.ones_like(self.data)
        else:
            grad = np.array(grad, dtype=self.data.dtype)

        topo: List[Tensor] = []
        visited: Set[int] = set()

        def build(node: Tensor) -> None:
            if id(node) in visited:
                return
            visited.add(id(node))
            if node._ctx is not None:
                for parent in node._ctx.parents:
                    build(parent)
            topo.append(node)

        build(self)
        self.grad = grad if self.grad is None else self.grad + grad

        for node in reversed(topo):
            if node._backward is None or node.grad is None:
                continue
            grads = node._backward(node.grad)
            if not isinstance(grads, tuple):
                grads = (grads,)
            for parent, parent_grad in zip(node._ctx.parents, grads):
                if parent_grad is None or not parent.requires_grad:
                    continue
                if parent.grad is None:
                    parent.grad = np.array(parent_grad, dtype=parent.data.dtype)
                else:
                    parent.grad = parent.grad + np.array(parent_grad, dtype=parent.data.dtype)

    def __add__(self, other: Any) -> "Tensor":
        return _add(self, ensure_tensor(other))

    def __radd__(self, other: Any) -> "Tensor":
        return ensure_tensor(other).__add__(self)

    def __sub__(self, other: Any) -> "Tensor":
        return _sub(self, ensure_tensor(other))

    def __rsub__(self, other: Any) -> "Tensor":
        return ensure_tensor(other).__sub__(self)

    def __mul__(self, other: Any) -> "Tensor":
        return _mul(self, ensure_tensor(other))

    def __rmul__(self, other: Any) -> "Tensor":
        return ensure_tensor(other).__mul__(self)

    def __truediv__(self, other: Any) -> "Tensor":
        return _div(self, ensure_tensor(other))

    def __rtruediv__(self, other: Any) -> "Tensor":
        return ensure_tensor(other).__truediv__(self)

    def __matmul__(self, other: Any) -> "Tensor":
        return _matmul(self, ensure_tensor(other))

    def sum(self) -> "Tensor":
        return _sum(self)

    def mean(self) -> "Tensor":
        return _mean(self)

    def relu(self) -> "Tensor":
        return _relu(self)

    def sigmoid(self) -> "Tensor":
        return _sigmoid(self)

    def __repr__(self) -> str:
        return f"Tensor(data={self.data!r}, requires_grad={self.requires_grad})"


def ensure_tensor(value: Any) -> Tensor:
    if isinstance(value, Tensor):
        return value
    return Tensor(value, requires_grad=False)