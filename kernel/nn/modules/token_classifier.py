from __future__ import annotations

import numpy as np

from kernel.core.tensor import Tensor
from kernel.nn.module import Module
from kernel.nn.layers.linear import Linear
from kernel.nn.layers.embedding import Embedding
from kernel.nn.modules.encoder import TransformerEncoder


class TokenTransformerClassifier(Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        num_layers: int,
        num_classes: int,
        dropout_p: float = 0.0,
        max_len: int = 5000,
        activation: str = "relu",
        pooling: str = "mean",
        use_cls_token: bool = False,
    ):
        super().__init__()

        if pooling not in {"mean", "cls", "last"}:
            raise ValueError(
                f"Unsupported pooling: {pooling}. Expected 'mean', 'cls', or 'last'."
            )

        if use_cls_token and pooling != "cls":
            raise ValueError(
                "use_cls_token=True is only valid when pooling='cls'"
            )

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_classes = num_classes
        self.pooling = pooling
        self.use_cls_token = use_cls_token

        self.embedding = Embedding(vocab_size, d_model)

        if use_cls_token:
            cls_init = np.zeros((1, 1, d_model), dtype=np.float32)
            self.cls_token = Tensor(cls_init, requires_grad=True)
        else:
            self.cls_token = None

        encoder_max_len = max_len + 1 if use_cls_token else max_len

        self.encoder = TransformerEncoder(
            d_model=d_model,
            num_heads=num_heads,
            d_ff=d_ff,
            num_layers=num_layers,
            dropout_p=dropout_p,
            max_len=encoder_max_len,
            use_positional_encoding=True,
            activation=activation,
        )
        self.classifier = Linear(d_model, num_classes)

    def _prepend_cls_token(self, x: Tensor) -> Tensor:
        # x: (B, T, D)
        b, _, d = x.shape

        cls = self.cls_token
        # (1, 1, D) -> (B, 1, D) через broadcast-safe композицию
        cls = cls + x[:, :1, :] * 0.0

        return Tensor.cat([cls, x], axis=1)

    def _extend_mask_for_cls(self, mask: Tensor) -> Tensor:
        # mask: (B, 1, 1, T) -> (B, 1, 1, T+1)
        b = mask.shape[0]
        cls_valid = Tensor(
            np.zeros((b, 1, 1, 1), dtype=np.float32),
            requires_grad=False,
            device=mask.device,
        )
        return Tensor.cat([cls_valid, mask], axis=-1)

    def _pool(self, encoded: Tensor) -> Tensor:
        if self.pooling == "mean":
            return encoded.mean(axis=1)
        if self.pooling == "cls":
            return encoded[:, 0, :]
        if self.pooling == "last":
            return encoded[:, -1, :]
        raise RuntimeError(f"Unexpected pooling mode: {self.pooling}")

    def forward(self, token_ids, mask: Tensor | None = None):
        x = self.embedding(token_ids)  # (B, T, D)

        if self.use_cls_token:
            x = self._prepend_cls_token(x)
            if mask is not None:
                mask = self._extend_mask_for_cls(mask)

        encoded, attn_all = self.encoder(x, mask=mask)
        pooled = self._pool(encoded)
        logits = self.classifier(pooled)
        return logits, attn_all