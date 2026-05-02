from __future__ import annotations

import numpy as np

from kernel.core.tensor import Tensor


def default_collate(batch):
    first = batch[0]

    if isinstance(first, Tensor):
        data = np.stack([item.data for item in batch], axis=0)
        requires_grad = any(item.requires_grad for item in batch)
        return Tensor(data, requires_grad=requires_grad)

    if isinstance(first, np.ndarray):
        return np.stack(batch, axis=0)

    if isinstance(first, (int, float, np.integer, np.floating)):
        return np.array(batch)

    if isinstance(first, tuple):
        transposed = list(zip(*batch))
        return tuple(default_collate(list(items)) for items in transposed)

    if isinstance(first, list):
        transposed = list(zip(*batch))
        return [default_collate(list(items)) for items in transposed]

    raise TypeError(f"Unsupported batch element type: {type(first)}")