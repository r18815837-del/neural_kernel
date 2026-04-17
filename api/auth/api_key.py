"""API key extraction and validation."""
from __future__ import annotations

import hmac

from .config import AuthConfig
from .errors import AuthenticationError
from .models import ApiKeyPrincipal, AuthContext

# Header name — stable client contract
API_KEY_HEADER = "X-API-Key"


def validate_api_key(raw_key: str | None, config: AuthConfig) -> AuthContext:
    """Validate an API key and return an AuthContext.

    Raises:
        AuthenticationError: if key is missing or invalid.
    """
    if not config.has_api_key:
        raise AuthenticationError(
            message="API key auth not configured on server",
            code="auth_not_configured",
        )

    if not raw_key:
        raise AuthenticationError(
            message="Missing API key — send X-API-Key header",
            code="api_key_missing",
        )

    if not _constant_time_eq(raw_key, config.client_api_key):
        raise AuthenticationError(
            message="Invalid API key",
            code="api_key_invalid",
        )

    principal = ApiKeyPrincipal(client_id="api_key_client")
    return AuthContext(
        authenticated=True,
        auth_type="api_key",
        client_id=principal.client_id,
        scopes=list(principal.scopes),
        raw_subject=raw_key[:8] + "…",  # truncated for logging safety
    )


def _constant_time_eq(a: str, b: str) -> bool:
    """Timing-safe string comparison."""
    return hmac.compare_digest(a.encode("utf-8"), b.encode("utf-8"))
