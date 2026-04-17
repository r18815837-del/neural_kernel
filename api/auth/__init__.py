"""Authentication & authorization layer for Neural Kernel API."""
from __future__ import annotations

from .config import AuthConfig, get_auth_config
from .models import AuthContext, ApiKeyPrincipal, BearerPrincipal
from .errors import AuthenticationError, AuthorizationError, auth_error_response
from .service import AuthService

# FastAPI-dependent exports — available when fastapi is importable
# from .dependencies import require_client_auth, optional_auth, get_auth_context

__all__ = [
    "AuthConfig",
    "get_auth_config",
    "AuthContext",
    "ApiKeyPrincipal",
    "BearerPrincipal",
    "AuthenticationError",
    "AuthorizationError",
    "auth_error_response",
    "AuthService",
]
