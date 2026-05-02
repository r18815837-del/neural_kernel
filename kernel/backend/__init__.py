# neural_kernel/backend/__init__.py

from __future__ import annotations

from .base import Backend
from .numpy_backend import numpy_backend
from .cupy_backend import cupy_backend


def normalize_device(device) -> str:
    if device is None:
        return "cpu"

    if not isinstance(device, str):
        raise TypeError(
            f"Device must be a string or None, got {type(device).__name__}"
        )

    device = device.strip().lower()

    if device not in {"cpu", "cuda"}:
        raise ValueError(
            f"Unsupported device: {device}. Expected 'cpu' or 'cuda'."
        )

    return device


def is_cuda_available() -> bool:
    return cupy_backend.is_available()


def get_backend(device) -> Backend:
    device = normalize_device(device)

    if device == "cpu":
        return numpy_backend

    if device == "cuda":
        if not cupy_backend.is_available():
            raise RuntimeError(
                "CuPy backend requested, but CUDA/CuPy is unavailable."
            )
        return cupy_backend

    raise ValueError(f"Unsupported device: {device}")


__all__ = [
    "Backend",
    "numpy_backend",
    "cupy_backend",
    "normalize_device",
    "is_cuda_available",
    "get_backend",
]