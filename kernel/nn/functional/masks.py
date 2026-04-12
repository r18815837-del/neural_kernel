from __future__ import annotations

import numpy as np

from kernel.core.tensor import Tensor


def make_causal_mask(
    seq_len: int,
    device: str = "cpu",
    mask_value: float = -1e9,
) -> Tensor:
    if seq_len <= 0:
        raise ValueError(f"seq_len must be > 0, got {seq_len}")

    mask = np.triu(
        np.full((seq_len, seq_len), mask_value, dtype=np.float64),
        k=1,
    )
    mask = mask.reshape(1, 1, seq_len, seq_len)

    return Tensor(mask, requires_grad=False, device=device)
def make_padding_mask(
    lengths,
    max_len: int | None = None,
    device: str = "cpu",
    mask_value: float = -1e9,
) -> Tensor:
    lengths = np.asarray(lengths, dtype=np.int64)

    if lengths.ndim != 1:
        raise ValueError(
            f"lengths must be 1D, got shape {lengths.shape}"
        )

    if lengths.size == 0:
        raise ValueError("lengths must not be empty")

    if np.any(lengths < 0):
        raise ValueError(f"lengths must be non-negative, got {lengths}")

    if max_len is None:
        max_len = int(lengths.max())

    if max_len <= 0:
        raise ValueError(f"max_len must be > 0, got {max_len}")

    if np.any(lengths > max_len):
        raise ValueError(
            f"All lengths must be <= max_len={max_len}, got {lengths}"
        )

    positions = np.arange(max_len, dtype=np.int64).reshape(1, max_len)
    valid = positions < lengths.reshape(-1, 1)         # (B, T)

    mask = np.where(valid, 0.0, mask_value).astype(np.float64)
    mask = mask.reshape(lengths.shape[0], 1, 1, max_len)

    return Tensor(mask, requires_grad=False, device=device)