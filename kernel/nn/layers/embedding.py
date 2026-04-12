from __future__ import annotations

import numpy as np

from kernel.core.tensor import Tensor
from kernel.nn.module import Module
from kernel.autograd.ops.tensor_ops import embedding_lookup


class Embedding(Module):
    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()

        if num_embeddings <= 0:
            raise ValueError(
                f"num_embeddings must be > 0, got {num_embeddings}"
            )
        if embedding_dim <= 0:
            raise ValueError(
                f"embedding_dim must be > 0, got {embedding_dim}"
            )

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        scale = np.sqrt(1.0 / embedding_dim)
        weight = np.random.randn(num_embeddings, embedding_dim) * scale
        self.weight = Tensor(weight.astype(np.float64), requires_grad=True)

    def forward(self, indices) -> Tensor:
        if isinstance(indices, Tensor):
            indices_data = indices.data
        else:
            indices_data = np.asarray(indices)

        if indices_data.ndim < 1:
            raise ValueError(
                f"Embedding expects at least 1D indices, got shape {indices_data.shape}"
            )

        # индексы должны быть integer
        if indices_data.dtype.kind not in {"i", "u"}:
            indices_data = indices_data.astype(np.int64)

        if indices_data.min() < 0 or indices_data.max() >= self.num_embeddings:
            raise ValueError(
                f"Embedding indices out of range [0, {self.num_embeddings}), "
                f"got min={indices_data.min()}, max={indices_data.max()}"
            )

        return embedding_lookup(self.weight, indices_data)