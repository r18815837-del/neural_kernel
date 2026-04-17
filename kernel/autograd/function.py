from __future__ import annotations
from typing import Any, List
from .context import Context



class Function:
    @classmethod
    def apply(cls, *args: Any, **kwargs: Any):
        from kernel.core.tensor import Tensor

        raw_args: List[Any] = []
        parents: List[Tensor] = []
        tensor_arg_indices: List[int] = []
        requires_grad = False
        device = "cpu"

        for idx, arg in enumerate(args):
            if isinstance(arg, Tensor):
                raw_args.append(arg.data)
                parents.append(arg)
                tensor_arg_indices.append(idx)
                requires_grad = requires_grad or arg.requires_grad
                if len(parents) == 1:
                    device = arg.device
            else:
                raw_args.append(arg)

        tensor_devices = {parent.device for parent in parents}
        if len(tensor_devices) > 1:
            raise RuntimeError(
                f"Device mismatch in {cls.__name__}: {tensor_devices}"
            )

        ctx = Context(parents=tuple(parents))
        ctx.meta["tensor_arg_indices"] = tuple(tensor_arg_indices)

        result = cls.forward(ctx, *raw_args, **kwargs)
        out = Tensor(result, requires_grad=requires_grad, device=device)

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
