from __future__ import annotations

from kernel.nn.module import Module
from kernel.nn.functional.attention import scaled_dot_product_attention
from kernel.core.tensor import Tensor


class ScaledDotProductAttention(Module):
    def __init__(self, dropout_p: float = 0.0):
        super().__init__()
        self.dropout_p = dropout_p

    def forward(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        mask: Tensor | None = None,
    ):
        return scaled_dot_product_attention(
            q,
            k,
            v,
            mask=mask,
            dropout_p=self.dropout_p,
            training=self.training,
        )