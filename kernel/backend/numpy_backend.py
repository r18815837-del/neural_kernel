# neural_kernel/backend/numpy_backend.py

from __future__ import annotations

import numpy as np

from .base import Backend


class NumpyBackend(Backend):
    name = "cpu"
    xp = np

    def is_available(self) -> bool:
        return True

    def array(self, data, dtype=None):
        return np.array(data, dtype=dtype)

    def asarray(self, data, dtype=None):
        return np.asarray(data, dtype=dtype)

    def zeros(self, shape, dtype=None):
        return np.zeros(shape, dtype=dtype)

    def ones(self, shape, dtype=None):
        return np.ones(shape, dtype=dtype)

    def empty(self, shape, dtype=None):
        return np.empty(shape, dtype=dtype)

    def zeros_like(self, x):
        return np.zeros_like(x)

    def ones_like(self, x):
        return np.ones_like(x)

    def empty_like(self, x):
        return np.empty_like(x)

    def copy(self, x):
        return np.array(x, copy=True)

    def to_cpu(self, x):
        return x

    def from_cpu(self, x):
        return np.asarray(x)

    def synchronize(self) -> None:
        return None


numpy_backend = NumpyBackend()