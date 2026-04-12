from __future__ import annotations

from kernel.core.tensor import Tensor
from kernel.nn.module import Module
from kernel.nn.modules.container import ModuleList
from kernel.nn.modules.transformer import PositionalEncoding, TransformerBlock


class TransformerEncoder(Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        num_layers: int,
        dropout_p: float = 0.0,
        max_len: int = 5000,
        use_positional_encoding: bool = True,
        activation: str = "relu",
    ):
        super().__init__()

        if num_layers <= 0:
            raise ValueError(f"num_layers must be > 0, got {num_layers}")

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.num_layers = num_layers
        self.dropout_p = dropout_p
        self.use_positional_encoding = use_positional_encoding
        self.activation = activation

        if use_positional_encoding:
            self.positional_encoding = PositionalEncoding(
                d_model=d_model,
                max_len=max_len,
            )
        else:
            self.positional_encoding = None

        self.layers = ModuleList(
            [
                TransformerBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    dropout_p=dropout_p,
                    activation=activation,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, x: Tensor, mask: Tensor | None = None):
        if x.data.ndim != 3:
            raise ValueError(
                f"TransformerEncoder expects 3D input (B, T, D), got shape {x.shape}"
            )

        if x.shape[-1] != self.d_model:
            raise ValueError(
                f"Expected last dim {self.d_model}, got {x.shape[-1]}"
            )

        if self.positional_encoding is not None:
            x = self.positional_encoding(x)

        attn_weights_all = []

        for layer in self.layers:
            x, attn_weights = layer(x, mask=mask)
            attn_weights_all.append(attn_weights)

        return x, attn_weights_all