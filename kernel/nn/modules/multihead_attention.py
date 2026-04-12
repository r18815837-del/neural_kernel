from __future__ import annotations

from kernel.core.tensor import Tensor
from kernel.nn.module import Module
from kernel.nn.layers.linear import Linear
from kernel.nn.modules.attention import ScaledDotProductAttention


class MultiHeadAttention(Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout_p: float = 0.0,
    ):
        super().__init__()

        if d_model % num_heads != 0:
            raise ValueError(
                f"d_model must be divisible by num_heads, got d_model={d_model}, num_heads={num_heads}"
            )

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_proj = Linear(d_model, d_model)
        self.k_proj = Linear(d_model, d_model)
        self.v_proj = Linear(d_model, d_model)
        self.out_proj = Linear(d_model, d_model)

        self.attention = ScaledDotProductAttention(dropout_p=dropout_p)

    def _split_heads(self, x: Tensor) -> Tensor:
        # x: (B, T, D) -> (B, H, T, Hd)
        b, t, d = x.shape
        x = x.reshape(b, t, self.num_heads, self.head_dim)
        x = x.transpose(0, 2, 1, 3)
        return x

    def _merge_heads(self, x: Tensor) -> Tensor:
        # x: (B, H, T, Hd) -> (B, T, D)
        b, h, t, hd = x.shape
        x = x.transpose(0, 2, 1, 3)
        x = x.reshape(b, t, h * hd)
        return x

    def forward(self, x: Tensor, mask: Tensor | None = None):
        if x.data.ndim != 3:
            raise ValueError(
                f"MultiHeadAttention expects 3D input (B, T, D), got shape {x.shape}"
            )

        if x.shape[-1] != self.d_model:
            raise ValueError(
                f"Expected last dim {self.d_model}, got {x.shape[-1]}"
            )

        q = self.q_proj(x)   # (B, T, D)
        k = self.k_proj(x)   # (B, T, D)
        v = self.v_proj(x)   # (B, T, D)

        q = self._split_heads(q)   # (B, H, T, Hd)
        k = self._split_heads(k)   # (B, H, T, Hd)
        v = self._split_heads(v)   # (B, H, T, Hd)

        attn_out, attn_weights = self.attention(q, k, v, mask=mask)
        out = self._merge_heads(attn_out)      # (B, T, D)
        out = self.out_proj(out)               # (B, T, D)

        return out, attn_weights