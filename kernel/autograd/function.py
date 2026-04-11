from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Tuple


@dataclass
class Context:
    parents: Tuple[Any, ...] = field(default_factory=tuple)
    saved_tensors: Tuple[Any, ...] = field(default_factory=tuple)
    meta: dict = field(default_factory=dict)

    def save_for_backward(self, *tensors: Any) -> None:
        self.saved_tensors = tuple(tensors)


class Function:
    @classmethod
    def apply(cls, *args: Any, **kwargs: Any):
        from kernel.core.tensor import Tensor

        raw_args: List[Any] = []
        parents: List[Tensor] = []
        requires_grad = False

        for arg in args:
            if isinstance(arg, Tensor):
                raw_args.append(arg.data)
                parents.append(arg)
                requires_grad = requires_grad or arg.requires_grad
            else:
                raw_args.append(arg)

        ctx = Context(parents=tuple(parents))
        result = cls.forward(ctx, *raw_args, **kwargs)
        out = Tensor(result, requires_grad=requires_grad)
        if requires_grad:
            out._ctx = ctx
            out._backward = lambda grad: cls.backward(ctx, grad)
        return out

    @staticmethod
    def forward(ctx: Context, *args: Any, **kwargs: Any):
        raise NotImplementedError

    @staticmethod
    def backward(ctx: Context, grad_output):
        raise NotImplementedError
