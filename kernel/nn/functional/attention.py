from __future__ import annotations

import numpy as np

from kernel.core.tensor import Tensor
from kernel.autograd.ops.math_ops import softmax


def scaled_dot_product_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    mask: Tensor | None = None,
    dropout_p: float = 0.0,
    training: bool = True,
):
    if q.data.ndim != 4:
        raise ValueError(f"q must be 4D (B, H, Tq, D), got shape {q.shape}")
    if k.data.ndim != 4:
        raise ValueError(f"k must be 4D (B, H, Tk, D), got shape {k.shape}")
    if v.data.ndim != 4:
        raise ValueError(f"v must be 4D (B, H, Tk, D), got shape {v.shape}")

    bq, hq, tq, dq = q.shape
    bk, hk, tk, dk = k.shape
    bv, hv, tv, dv = v.shape

    if bq != bk or bq != bv:
        raise ValueError(
            f"Batch mismatch: q={q.shape}, k={k.shape}, v={v.shape}"
        )

    if hq != hk or hq != hv:
        raise ValueError(
            f"Head mismatch: q={q.shape}, k={k.shape}, v={v.shape}"
        )

    if dk != dq:
        raise ValueError(
            f"Key/query depth mismatch: q last dim={dq}, k last dim={dk}"
        )

    if tv != tk:
        raise ValueError(
            f"Value/key sequence mismatch: k shape={k.shape}, v shape={v.shape}"
        )

    scale = np.sqrt(dq)

    k_t = k.transpose(0, 1, 3, 2)
    scores = (q @ k_t) / scale

    if mask is not None:
        if not isinstance(mask, Tensor):
            mask = Tensor(mask, requires_grad=False, device=q.device)
        elif mask.device != q.device:
            mask = mask.to(q.device)

        scores = scores + mask

    attn = softmax(scores, axis=-1)

    if dropout_p > 0.0:
        attn = attn.dropout(p=dropout_p, training=training)

    out = attn @ v

    return out, attn