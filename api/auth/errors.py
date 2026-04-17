"""Auth error types and HTTP error response helpers."""
from __future__ import annotations

from typing import Any, Dict


class AuthenticationError(Exception):
    """Raised when credentials are missing or invalid (-> 401)."""

    def __init__(self, message: str = "Authentication required", code: str = "auth_required") -> None:
        self.message = message
        self.code = code
        super().__init__(message)


class AuthorizationError(Exception):
    """Raised when the caller lacks permissions (-> 403)."""

    def __init__(self, message: str = "Insufficient permissions", code: str = "forbidden") -> None:
        self.message = message
        self.code = code
        super().__init__(message)


def auth_error_response(
    status_code: int = 401,
    code: str = "auth_required",
    message: str = "Authentication required",
    details: Dict[str, Any] | None = None,
    retryable: bool = False,
):
    """Build a client-friendly JSON error response.

    Compatible with integration/dto/errors.py format::

        {"code": "...", "message": "...", "retryable": false}

    Returns a fastapi.responses.JSONResponse (imported lazily).
    """
    from fastapi.responses import JSONResponse

    body: dict[str, object] = {
        "code": code,
        "message": message,
        "retryable": retryable,
    }
    if details:
        body["details"] = details
    return JSONResponse(status_code=status_code, content=body)
