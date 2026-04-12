from __future__ import annotations

from kernel.core.tensor import Tensor
from kernel.nn.module import Module
from kernel.nn.layers.linear import Linear
from kernel.nn.modules.encoder import TransformerEncoder


class TransformerEncoderClassifier(Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        num_layers: int,
        num_classes: int,
        dropout_p: float = 0.0,
        max_len: int = 5000,
        use_positional_encoding: bool = True,
        activation: str = "relu",
        pooling: str = "mean",
    ):
        super().__init__()

        if num_classes <= 0:
            raise ValueError(f"num_classes must be > 0, got {num_classes}")

        if pooling not in {"mean", "cls", "last"}:
            raise ValueError(
                f"Unsupported pooling: {pooling}. Expected 'mean', 'cls', or 'last'."
            )

        self.d_model = d_model
        self.num_classes = num_classes
        self.activation = activation
        self.pooling = pooling

        self.encoder = TransformerEncoder(
            d_model=d_model,
            num_heads=num_heads,
            d_ff=d_ff,
            num_layers=num_layers,
            dropout_p=dropout_p,
            max_len=max_len,
            use_positional_encoding=use_positional_encoding,
            activation=activation,
        )

        self.classifier = Linear(d_model, num_classes)

    def _pool(self, encoded: Tensor) -> Tensor:
        if self.pooling == "mean":
            return encoded.mean(axis=1)
        if self.pooling == "cls":
            return encoded[:, 0, :]
        if self.pooling == "last":
            return encoded[:, -1, :]
        raise RuntimeError(f"Unexpected pooling mode: {self.pooling}")

    def forward(self, x: Tensor, mask: Tensor | None = None):
        if x.data.ndim != 3:
            raise ValueError(
                f"TransformerEncoderClassifier expects 3D input (B, T, D), got shape {x.shape}"
            )

        if x.shape[-1] != self.d_model:
            raise ValueError(
                f"Expected last dim {self.d_model}, got {x.shape[-1]}"
            )

        encoded, attn_weights_all = self.encoder(x, mask=mask)
        pooled = self._pool(encoded)
        logits = self.classifier(pooled)

        return logits, attn_weights_all