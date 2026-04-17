"""FastAPI dependency functions for auth — inject into route signatures."""
from __future__ import annotations

from typing import Optional

from fastapi import Header, Request

from .api_key import API_KEY_HEADER
from .errors import AuthenticationError, auth_error_response
from .models import ANONYMOUS, AuthContext
from .service import AuthService, get_auth_service


async def get_auth_context(
    request: Request,
    x_api_key: Optional[str] = Header(None, alias=API_KEY_HEADER),
    authorization: Optional[str] = Header(None),
) -> AuthContext:
    """Resolve auth context from request headers.

    Does NOT enforce auth — use ``require_client_auth`` for that.
    Returns anonymous context if no credentials are provided and
    auth is disabled.
    """
    service = get_auth_service()

    # Anonymous path check
    if service.is_anonymous_path(request.url.path):
        if not x_api_key and not authorization:
            return ANONYMOUS

    try:
        return service.authenticate(api_key=x_api_key, authorization=authorization)
    except AuthenticationError:
        # If auth is not enabled, allow through as anonymous
        if not service.config.auth_enabled:
            return ANONYMOUS
        raise


async def require_client_auth(
    request: Request,
    x_api_key: Optional[str] = Header(None, alias=API_KEY_HEADER),
    authorization: Optional[str] = Header(None),
) -> AuthContext:
    """Dependency that REQUIRES valid authentication.

    Raises HTTPException(401) if auth fails.
    Use this on all ``/api/v1/client/...`` endpoints.
    """
    service = get_auth_service()

    # If auth is globally disabled (dev mode), allow anonymous
    if not service.config.auth_enabled:
        return ANONYMOUS

    # Anonymous-path bypass
    if service.is_anonymous_path(request.url.path):
        return ANONYMOUS

    try:
        ctx = service.authenticate(api_key=x_api_key, authorization=authorization)
    except AuthenticationError as exc:
        raise _to_http_exc(exc)

    if ctx.is_anonymous:
        raise _to_http_exc(
            AuthenticationError("Authentication required", "auth_required")
        )

    return ctx


async def optional_auth(
    request: Request,
    x_api_key: Optional[str] = Header(None, alias=API_KEY_HEADER),
    authorization: Optional[str] = Header(None),
) -> AuthContext:
    """Dependency that accepts both authenticated and anonymous.

    Never raises — returns anonymous on failure.
    """
    service = get_auth_service()
    try:
        return service.authenticate(api_key=x_api_key, authorization=authorization)
    except AuthenticationError:
        return ANONYMOUS


def _to_http_exc(exc: AuthenticationError):
    """Convert AuthenticationError → FastAPI HTTPException."""
    from fastapi import HTTPException
    return HTTPException(
        status_code=401,
        detail={
            "code": exc.code,
            "message": exc.message,
            "retryable": False,
        },
    )
