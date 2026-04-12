from __future__ import annotations
import numpy as np
from kernel.core.tensor import Tensor
from kernel.nn.module import Module
from kernel.nn.layers.linear import Linear
from kernel.nn.normalization import LayerNorm
from kernel.nn.modules.multihead_attention import MultiHeadAttention


class FeedForward(Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout_p: float = 0.0,
        activation: str = "relu",
    ):
        super().__init__()
        self.fc1 = Linear(d_model, d_ff)
        self.fc2 = Linear(d_ff, d_model)
        self.dropout_p = dropout_p
        self.activation = activation

        if activation not in {"relu", "gelu"}:
            raise ValueError(
                f"Unsupported activation: {activation}. Expected 'relu' or 'gelu'."
            )

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)

        if self.activation == "relu":
            x = x.relu()
        elif self.activation == "gelu":
            x = x.gelu()

        if self.dropout_p > 0.0:
            x = x.dropout(p=self.dropout_p, training=self.training)

        x = self.fc2(x)
        return x
class TransformerBlock(Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout_p: float = 0.0,
        activation: str = "relu",
    ):
        super().__init__()

        self.ln1 = LayerNorm(d_model)
        self.ln2 = LayerNorm(d_model)

        self.attn = MultiHeadAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout_p=dropout_p,
        )

        self.ffn = FeedForward(
            d_model=d_model,
            d_ff=d_ff,
            dropout_p=dropout_p,
            activation=activation,
        )

        self.dropout_p = dropout_p
        self.activation = activation

    def forward(self, x: Tensor, mask: Tensor | None = None):
        attn_in = self.ln1(x)
        attn_out, attn_weights = self.attn(attn_in, mask=mask)

        if self.dropout_p > 0.0:
            attn_out = attn_out.dropout(p=self.dropout_p, training=self.training)

        x = x + attn_out

        ffn_in = self.ln2(x)
        ffn_out = self.ffn(ffn_in)

        if self.dropout_p > 0.0:
            ffn_out = ffn_out.dropout(p=self.dropout_p, training=self.training)

        x = x + ffn_out
        return x, attn_weights
class PositionalEncoding(Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()

        if d_model <= 0:
            raise ValueError(f"d_model must be > 0, got {d_model}")
        if max_len <= 0:
            raise ValueError(f"max_len must be > 0, got {max_len}")

        self.d_model = d_model
        self.max_len = max_len

        position = np.arange(max_len, dtype=np.float64).reshape(max_len, 1)
        div_term = np.exp(
            np.arange(0, d_model, 2, dtype=np.float64) * (-np.log(10000.0) / d_model)
        )

        pe = np.zeros((1, max_len, d_model), dtype=np.float64)
        pe[0, :, 0::2] = np.sin(position * div_term)

        if d_model % 2 == 0:
            pe[0, :, 1::2] = np.cos(position * div_term)
        else:
            pe[0, :, 1::2] = np.cos(position * div_term[:-1])

        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        if x.data.ndim != 3:
            raise ValueError(
                f"PositionalEncoding expects 3D input (B, T, D), got shape {x.shape}"
            )

        b, t, d = x.shape

        if d != self.d_model:
            raise ValueError(f"Expected last dim {self.d_model}, got {d}")

        if t > self.max_len:
            raise ValueError(
                f"Sequence length {t} exceeds max_len={self.max_len}"
            )

        pos = self.pe.data[:, :t, :]
        return x + pos