"""API route modules."""
from __future__ import annotations

from .code import router as code_router
from .cognition import router as cognition_router
from .generation import router as generation_router
from .health import router as health_router
from .lifecycle import router as lifecycle_router
from .client import router as client_router
from .sessions import router as sessions_router

__all__ = [
    "code_router",
    "cognition_router",
    "generation_router",
    "health_router",
    "lifecycle_router",
    "client_router",
    "sessions_router",
]
