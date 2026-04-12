from __future__ import annotations

from typing import Any, List, Set

import numpy as np
from ..backend import get_backend, normalize_device
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

    def __init__(
        self,
        data: Any,
        requires_grad: bool = False,
        dtype=None,
        device: str = "cpu",
    ):
        self.device = normalize_device(device)
        self._backend = get_backend(self.device)

        if isinstance(data, Tensor):
            data = data.data

        target_dtype = dtype if dtype is not None else np.float64
        self.data = self._backend.asarray(data, dtype=target_dtype)

        self.requires_grad = requires_grad
        self.grad = None
        self._ctx = None
        self._backward = None

    @property
    def xp(self):
        return self._backend.xp

    @property
    def shape(self):
        return self.data.shape

    @property
    def dtype(self):
        return self.data.dtype

    @property
    def T(self) -> "Tensor":
        if self.data.ndim < 2:
            return self
        axes = list(range(self.data.ndim))
        axes[-1], axes[-2] = axes[-2], axes[-1]
        return self.transpose(tuple(axes))

    def numpy(self) -> np.ndarray:
        return self._backend.to_cpu(self.data).copy()

    def to(self, device: str) -> "Tensor":
        target_device = normalize_device(device)

        if target_device == self.device:
            return self

        if self._ctx is not None:
            raise RuntimeError(
                "Moving non-leaf Tensor across devices is not supported yet"
            )

        target_backend = get_backend(target_device)

        cpu_data = self._backend.to_cpu(self.data)
        new_data = target_backend.from_cpu(cpu_data)

        out = Tensor(
            new_data,
            requires_grad=self.requires_grad,
            dtype=self.data.dtype,
            device=target_device,
        )

        if self.grad is not None:
            cpu_grad = self._backend.to_cpu(self.grad)
            out.grad = target_backend.from_cpu(cpu_grad)

        return out



    def cpu(self) -> "Tensor":
        return self.to("cpu")

    def cuda(self) -> "Tensor":
        return self.to("cuda")

    def detach(self) -> "Tensor":
        out = Tensor(
            self._backend.copy(self.data),
            requires_grad=False,
            dtype=self.data.dtype,
            device=self.device,
        )
        return out

    def masked_fill(self, mask, value):
        mask_t = ensure_tensor(mask, like=self)

        mask_data = self.xp.asarray(mask_t.data != 0, dtype=self.data.dtype)
        mask_t = Tensor(mask_data, requires_grad=False, device=self.device)

        value_t = ensure_tensor(value, like=self)
        return self + mask_t * (value_t - self)

    def concat(self, others, axis=0):
        from kernel.autograd.ops.tensor_ops import concat

        tensors = [self]
        for other in others:
            tensors.append(ensure_tensor(other, like=self))

        return concat(tensors, axis=axis)

    def reshape(self, *shape):
        from kernel.autograd.ops.tensor_ops import reshape

        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])

        return reshape(self, shape)

    def transpose(self, *axes):
        from kernel.autograd.ops.tensor_ops import transpose

        if len(axes) == 1 and isinstance(axes[0], (tuple, list)):
            axes = tuple(axes[0])

        return transpose(self, axes)

    def unsqueeze(self, axis):
        from kernel.autograd.ops.tensor_ops import unsqueeze
        return unsqueeze(self, axis)

    def squeeze(self, axis=None):
        from kernel.autograd.ops.tensor_ops import squeeze
        return squeeze(self, axis)

    @staticmethod
    def cat(tensors, axis=0):
        from kernel.autograd.ops.tensor_ops import concat

        if len(tensors) == 0:
            raise ValueError("Tensor.cat expects at least one tensor")

        first = tensors[0]
        if not isinstance(first, Tensor):
            raise TypeError("Tensor.cat expects tensors")

        prepared = [first]
        for t in tensors[1:]:
            prepared.append(ensure_tensor(t, like=first))

        return concat(prepared, axis=axis)
    def permute(self, *axes):
        return self.transpose(*axes)

    def sqrt(self):
        from kernel.autograd.ops.math_ops import sqrt
        return sqrt(self)

    def dropout(self, p: float = 0.5, training: bool = True):
        from kernel.autograd.ops.math_ops import dropout
        return dropout(self, p=p, training=training)

    def zero_grad(self) -> None:
        self.grad = self.xp.zeros_like(self.data)

    def backward(self, grad: Any = None) -> None:
        if not self.requires_grad:
            raise RuntimeError("Cannot call backward on tensor with requires_grad=False")

        if grad is None:
            if self.data.size != 1:
                raise RuntimeError("grad must be specified for non-scalar tensor")
            grad = self.xp.ones_like(self.data)
        else:
            grad = self.xp.asarray(grad, dtype=self.data.dtype)

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

            tensor_arg_indices = node._ctx.meta.get("tensor_arg_indices")

            if tensor_arg_indices is None:
                parent_grads = grads
            else:
                parent_grads = tuple(grads[i] for i in tensor_arg_indices)

            for parent, parent_grad in zip(node._ctx.parents, parent_grads):
                if parent_grad is None or not parent.requires_grad:
                    continue
                parent_grad = parent.xp.asarray(parent_grad, dtype=parent.data.dtype)
                if parent.grad is None:
                    parent.grad = parent_grad
                else:
                    parent.grad = parent.grad + parent_grad

    def __add__(self, other: Any) -> "Tensor":
        return _add(self, ensure_tensor(other, like=self))

    def __radd__(self, other: Any) -> "Tensor":
        return _add(ensure_tensor(other, like=self), self)

    def __sub__(self, other: Any) -> "Tensor":
        return _sub(self, ensure_tensor(other, like=self))

    def __rsub__(self, other: Any) -> "Tensor":
        return _sub(ensure_tensor(other, like=self), self)

    def __mul__(self, other: Any) -> "Tensor":
        return _mul(self, ensure_tensor(other, like=self))

    def __rmul__(self, other: Any) -> "Tensor":
        return _mul(ensure_tensor(other, like=self), self)

    def __truediv__(self, other: Any) -> "Tensor":
        return _div(self, ensure_tensor(other, like=self))

    def __rtruediv__(self, other: Any) -> "Tensor":
        return _div(ensure_tensor(other, like=self), self)

    def __matmul__(self, other: Any) -> "Tensor":
        return _matmul(self, ensure_tensor(other, like=self))

    def sum(self, axis=None, keepdims=False) -> "Tensor":
        return _sum(self, axis=axis, keepdims=keepdims)

    def mean(self, axis=None, keepdims=False) -> "Tensor":
        return _mean(self, axis=axis, keepdims=keepdims)

    def relu(self) -> "Tensor":
        return _relu(self)

    def sigmoid(self) -> "Tensor":
        return _sigmoid(self)

    def gelu(self) -> "Tensor":
        from kernel.autograd.ops.math_ops import gelu
        return gelu(self)

    def __getitem__(self, index):
        from kernel.autograd.ops.tensor_ops import getitem
        return getitem(self, index)

    def __repr__(self) -> str:
        return (
            f"Tensor(data={self.data!r}, "
            f"requires_grad={self.requires_grad}, "
            f"device='{self.device}')"
        )


def ensure_tensor(value: Any, like: Tensor | None = None) -> Tensor:
    if isinstance(value, Tensor):
        return value

    device = like.device if like is not None else "cpu"
    return Tensor(value, requires_grad=False, device=device)