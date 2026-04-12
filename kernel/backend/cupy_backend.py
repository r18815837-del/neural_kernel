# neural_kernel/backend/cupy_backend.py

from __future__ import annotations

from .base import Backend

try:
    import cupy as cp
    _cupy_import_error = None
except Exception as e:
    cp = None
    _cupy_import_error = e


class CuPyBackend(Backend):
    name = "cuda"
    xp = cp

    def is_available(self) -> bool:
        if cp is None:
            return False
        try:
            _ = cp.cuda.runtime.getDeviceCount()
            return True
        except Exception:
            return False

    def _check_available(self) -> None:
        if cp is None:
            raise RuntimeError(f"CuPy is not available: {_cupy_import_error}")
        try:
            _ = cp.cuda.runtime.getDeviceCount()
        except Exception as e:
            raise RuntimeError(f"CUDA runtime is unavailable: {e}")

    def array(self, data, dtype=None):
        self._check_available()
        return cp.array(data, dtype=dtype)

    def asarray(self, data, dtype=None):
        self._check_available()
        return cp.asarray(data, dtype=dtype)

    def zeros(self, shape, dtype=None):
        self._check_available()
        return cp.zeros(shape, dtype=dtype)

    def ones(self, shape, dtype=None):
        self._check_available()
        return cp.ones(shape, dtype=dtype)

    def empty(self, shape, dtype=None):
        self._check_available()
        return cp.empty(shape, dtype=dtype)

    def zeros_like(self, x):
        self._check_available()
        return cp.zeros_like(x)

    def ones_like(self, x):
        self._check_available()
        return cp.ones_like(x)

    def empty_like(self, x):
        self._check_available()
        return cp.empty_like(x)

    def copy(self, x):
        self._check_available()
        return cp.array(x, copy=True)

    def to_cpu(self, x):
        self._check_available()
        return cp.asnumpy(x)

    def from_cpu(self, x):
        self._check_available()
        return cp.asarray(x)

    def synchronize(self) -> None:
        self._check_available()
        cp.cuda.Stream.null.synchronize()


cupy_backend = CuPyBackend()