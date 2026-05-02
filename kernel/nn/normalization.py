from __future__ import annotations

import numpy as np

from kernel.core.tensor import Tensor
from kernel.nn.module import Module
from kernel.autograd.ops import layer_norm

class BatchNorm1d(Module):
    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
    ):
        super().__init__()

        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine

        if affine:
            self.weight = Tensor(
                np.ones(num_features, dtype=np.float32),
                requires_grad=True,
            )
            self.bias = Tensor(
                np.zeros(num_features, dtype=np.float32),
                requires_grad=True,
            )
        else:
            self.weight = None
            self.bias = None

        self.register_buffer(
            "running_mean",
            np.zeros(num_features, dtype=np.float32),
        )
        self.register_buffer(
            "running_var",
            np.ones(num_features, dtype=np.float32),
        )

    def forward(self, x: Tensor) -> Tensor:
        if x.data.ndim != 2:
            raise ValueError(
                f"BatchNorm1d expects 2D input (batch, features), got shape {x.data.shape}"
            )

        if x.data.shape[1] != self.num_features:
            raise ValueError(
                f"Expected {self.num_features} features, got {x.data.shape[1]}"
            )

        if self.training:
            batch_mean = x.data.mean(axis=0)
            batch_var = x.data.var(axis=0)

            self.running_mean.data = (
                (1.0 - self.momentum) * self.running_mean.data
                + self.momentum * batch_mean
            )
            self.running_var.data = (
                (1.0 - self.momentum) * self.running_var.data
                + self.momentum * batch_var
            )

            mean = Tensor(batch_mean, requires_grad=False, device=x.device)
            var = Tensor(batch_var, requires_grad=False, device=x.device)
        else:
            mean = self.running_mean
            var = self.running_var

        x_hat = (x - mean) / (var + self.eps).sqrt()

        if self.affine:
            return x_hat * self.weight + self.bias
        return x_hat
class BatchNorm2d(Module):
    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
    ):
        super().__init__()

        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine

        if affine:
            self.weight = Tensor(
                np.ones((1, num_features, 1, 1), dtype=np.float32),
                requires_grad=True,
            )
            self.bias = Tensor(
                np.zeros((1, num_features, 1, 1), dtype=np.float32),
                requires_grad=True,
            )
        else:
            self.weight = None
            self.bias = None

        self.register_buffer(
            "running_mean",
            np.zeros((1, num_features, 1, 1), dtype=np.float32),
        )
        self.register_buffer(
            "running_var",
            np.ones((1, num_features, 1, 1), dtype=np.float32),
        )

    def forward(self, x: Tensor) -> Tensor:
        if x.data.ndim != 4:
            raise ValueError(
                f"BatchNorm2d expects 4D input (N, C, H, W), got shape {x.data.shape}"
            )

        if x.data.shape[1] != self.num_features:
            raise ValueError(
                f"Expected {self.num_features} channels, got {x.data.shape[1]}"
            )

        if self.training:
            batch_mean = x.data.mean(axis=(0, 2, 3), keepdims=True)
            batch_var = x.data.var(axis=(0, 2, 3), keepdims=True)

            self.running_mean.data = (
                (1.0 - self.momentum) * self.running_mean.data
                + self.momentum * batch_mean
            )
            self.running_var.data = (
                (1.0 - self.momentum) * self.running_var.data
                + self.momentum * batch_var
            )

            mean = Tensor(batch_mean, requires_grad=False, device=x.device)
            var = Tensor(batch_var, requires_grad=False, device=x.device)
        else:
            mean = self.running_mean
            var = self.running_var

        x_hat = (x - mean) / (var + self.eps).sqrt()

        if self.affine:
            return x_hat * self.weight + self.bias
        return x_hat

class LayerNorm(Module):
    def __init__(
        self,
        normalized_shape: int | tuple[int, ...],
        eps: float = 1e-5,
        affine: bool = True,
    ):
        super().__init__()

        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)

        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.affine = affine

        if affine:
            self.weight = Tensor(
                np.ones(self.normalized_shape, dtype=np.float32),
                requires_grad=True,
            )
            self.bias = Tensor(
                np.zeros(self.normalized_shape, dtype=np.float32),
                requires_grad=True,
            )
        else:
            self.weight = None
            self.bias = None

    def forward(self, x: Tensor) -> Tensor:
        if x.data.shape[-len(self.normalized_shape):] != self.normalized_shape:
            raise ValueError(
                f"Expected trailing shape {self.normalized_shape}, got {x.data.shape}"
            )

        return layer_norm(
            x,
            self.normalized_shape,
            eps=self.eps,
            weight=self.weight,
            bias=self.bias,
        )
class GroupNorm(Module):
    def __init__(
            self,
            num_groups: int,
            num_channels: int,
            eps: float = 1e-5,
            affine: bool = True,
    ):
        super().__init__()

        if num_channels % num_groups != 0:
            raise ValueError(
                f"num_channels ({num_channels}) must be divisible by num_groups ({num_groups})"
            )

        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine

        if affine:
            self.weight = Tensor(
                np.ones((1, num_channels, 1, 1), dtype=np.float32),
                requires_grad=True,
            )
            self.bias = Tensor(
                np.zeros((1, num_channels, 1, 1), dtype=np.float32),
                requires_grad=True,
            )
        else:
            self.weight = None
            self.bias = None

    def forward(self, x: Tensor) -> Tensor:
        if x.data.ndim != 4:
            raise ValueError(
                f"GroupNorm expects 4D input (N, C, H, W), got shape {x.data.shape}"
            )

        n, c, h, w = x.data.shape
        if c != self.num_channels:
            raise ValueError(
                f"Expected {self.num_channels} channels, got {c}"
            )

        g = self.num_groups
        xg = x.data.reshape(n, g, c // g, h, w)

        mean = xg.mean(axis=(2, 3, 4), keepdims=True)
        var = xg.var(axis=(2, 3, 4), keepdims=True)

        x_hat = (xg - mean) / np.sqrt(var + self.eps)
        x_hat = x_hat.reshape(n, c, h, w)

        out = Tensor(x_hat, requires_grad=x.requires_grad)

        # Чтобы сохранить корректный граф, affine применяем через Tensor ops
        if self.affine:
            return out * self.weight + self.bias
        return out