# neural_kernel/backend/base.py

from __future__ import annotations


class Backend:
    """Minimal backend contract for array creation and device transfers."""

    name: str
    xp: object

    def is_available(self) -> bool:
        raise NotImplementedError

    def array(self, data, dtype=None):
        raise NotImplementedError

    def asarray(self, data, dtype=None):
        raise NotImplementedError

    def zeros(self, shape, dtype=None):
        raise NotImplementedError

    def ones(self, shape, dtype=None):
        raise NotImplementedError

    def empty(self, shape, dtype=None):
        raise NotImplementedError

    def zeros_like(self, x):
        raise NotImplementedError

    def ones_like(self, x):
        raise NotImplementedError

    def empty_like(self, x):
        raise NotImplementedError

    def copy(self, x):
        raise NotImplementedError

    def to_cpu(self, x):
        raise NotImplementedError

    def from_cpu(self, x):
        raise NotImplementedError

    def synchronize(self) -> None:
        raise NotImplementedError