from __future__ import annotations

import math
import numpy as np

from kernel.core.tensor import Tensor


def _calculate_fan_in_and_fan_out(tensor: Tensor) -> tuple[int, int]:
    shape = tensor.data.shape

    if len(shape) < 2:
        raise ValueError(
            f"Fan in and fan out require tensor with at least 2 dims, got shape {shape}"
        )

    if len(shape) == 2:
        fan_in = shape[0]
        fan_out = shape[1]
    else:
        receptive_field_size = int(np.prod(shape[2:])) if len(shape) > 2 else 1
        fan_in = shape[1] * receptive_field_size
        fan_out = shape[0] * receptive_field_size

    return fan_in, fan_out


def xavier_uniform_(tensor: Tensor, gain: float = 1.0) -> Tensor:
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    limit = gain * math.sqrt(6.0 / (fan_in + fan_out))
    xp = tensor.xp
    tensor.data = xp.random.uniform(-limit, limit, size=tensor.data.shape).astype(tensor.data.dtype)
    return tensor


def xavier_normal_(tensor: Tensor, gain: float = 1.0) -> Tensor:
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    std = gain * math.sqrt(2.0 / (fan_in + fan_out))
    xp = tensor.xp
    tensor.data = xp.random.normal(0.0, std, size=tensor.data.shape).astype(tensor.data.dtype)
    return tensor


def kaiming_uniform_(tensor: Tensor, a: float = 0.0) -> Tensor:
    fan_in, _ = _calculate_fan_in_and_fan_out(tensor)
    gain = math.sqrt(2.0 / (1.0 + a ** 2))
    std = gain / math.sqrt(fan_in)
    bound = math.sqrt(3.0) * std
    xp = tensor.xp
    tensor.data = xp.random.uniform(-bound, bound, size=tensor.data.shape).astype(tensor.data.dtype)
    return tensor


def kaiming_normal_(tensor: Tensor, a: float = 0.0) -> Tensor:
    fan_in, _ = _calculate_fan_in_and_fan_out(tensor)
    gain = math.sqrt(2.0 / (1.0 + a ** 2))
    std = gain / math.sqrt(fan_in)
    xp = tensor.xp
    tensor.data = xp.random.normal(0.0, std, size=tensor.data.shape).astype(tensor.data.dtype)
    return tensor