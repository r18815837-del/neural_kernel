"""Bearer / JWT token validation.

Decodes JWT tokens using PyJWT, verifies signature + standard claims,
extracts Neural Kernel-specific claims (user_id, org_id, role, scopes)
and returns a fully populated AuthContext.

Supported algorithms:
  - HS256 / HS384 / HS512 — via NK_JWT_SECRET
  - RS256 / RS384 / RS512, ES256 / ES384 — via NK_JWT_PUBLIC_KEY (PEM)

Expected JWT claims (all optional except ``sub``):
  - sub        — subject (required, used as fallback client_id)
  - client_id  — NK client identifier
  - user_id    — end-user identifier
  - org_id     — organization / tenant identifier
  - role       — canonical role (admin | operator | client | viewer)
  - scope      — space-separated scope string  (or ``scopes`` as list)
  - iss, aud, exp, iat, nbf — standard JWT claims
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from .config import AuthConfig
from .errors import AuthenticationError
from .models import (
    ALL_ROLES,
    AuthContext,
    BearerPrincipal,
    ROLE_CLIENT,
    scopes_for_role,
)

logger = logging.getLogger(__name__)

# PyJWT is an optional runtime dependency — import lazily so the module
# can be imported even when jwt is not installed (for static analysis,
# test mocking, etc.).
_jwt_module = None


def _get_jwt():
    """Lazy-import PyJWT."""
    global _jwt_module
    if _jwt_module is None:
        try:
            import jwt
            _jwt_module = jwt
        except ImportError:
            raise AuthenticationError(
                message="JWT support unavailable — install PyJWT",
                code="jwt_not_installed",
            )
    return _jwt_module


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------


def validate_bearer_token(
    raw_token: str | None,
    config: AuthConfig,
) -> AuthContext:
    """Validate a bearer token and return an AuthContext.

    Raises:
        AuthenticationError on any validation failure.
    """
    if not config.bearer_enabled:
        raise AuthenticationError(
            message="Bearer token auth is not enabled",
            code="bearer_not_enabled",
        )

    if not raw_token:
        raise AuthenticationError(
            message="Missing bearer token",
            code="bearer_token_missing",
        )

    if not config.has_jwt_key:
        raise AuthenticationError(
            message="JWT signing key not configured — set NK_JWT_SECRET or NK_JWT_PUBLIC_KEY",
            code="jwt_key_missing",
        )

    jwt = _get_jwt()

    # Build decode options
    decode_kwargs: Dict[str, Any] = {
        "key": config.jwt_decode_key,
        "algorithms": [config.jwt_algorithm],
        "leeway": config.jwt_leeway,
    }
    if config.jwt_issuer:
        decode_kwargs["issuer"] = config.jwt_issuer
    if config.jwt_audience:
        decode_kwargs["audience"] = config.jwt_audience
    else:
        # If no audience configured, don't require it
        decode_kwargs["options"] = {"verify_aud": False}

    try:
        claims = jwt.decode(raw_token, **decode_kwargs)
    except jwt.ExpiredSignatureError:
        raise AuthenticationError(
            message="Token has expired",
            code="token_expired",
        )
    except jwt.InvalidIssuerError:
        raise AuthenticationError(
            message="Invalid token issuer",
            code="token_invalid_issuer",
        )
    except jwt.InvalidAudienceError:
        raise AuthenticationError(
            message="Invalid token audience",
            code="token_invalid_audience",
        )
    except jwt.DecodeError as exc:
        raise AuthenticationError(
            message=f"Token decode failed: {exc}",
            code="token_decode_error",
        )
    except jwt.InvalidTokenError as exc:
        raise AuthenticationError(
            message=f"Invalid token: {exc}",
            code="token_invalid",
        )

    return _claims_to_context(claims)


# ------------------------------------------------------------------
# Claims → AuthContext
# ------------------------------------------------------------------


def _claims_to_context(claims: Dict[str, Any]) -> AuthContext:
    """Map validated JWT claims to an AuthContext."""
    subject = claims.get("sub")
    if not subject:
        raise AuthenticationError(
            message="Token missing required 'sub' claim",
            code="token_missing_sub",
        )

    # Extract NK-specific claims
    client_id = claims.get("client_id") or subject
    user_id = claims.get("user_id")
    org_id = claims.get("org_id")
    role = claims.get("role")

    # Validate role if present
    if role and role not in ALL_ROLES:
        logger.warning("Unknown role '%s' in JWT — defaulting to 'client'", role)
        role = ROLE_CLIENT

    role = role or ROLE_CLIENT

    # Scopes — accept either space-separated string or list
    scopes = _parse_scopes(claims)
    if not scopes:
        scopes = scopes_for_role(role)

    principal = BearerPrincipal(
        subject=subject,
        client_id=client_id,
        user_id=user_id,
        org_id=org_id,
        role=role,
        scopes=scopes,
        issuer=claims.get("iss"),
        expires_at=claims.get("exp"),
    )
    return principal.to_auth_context()


def _parse_scopes(claims: Dict[str, Any]) -> list[str]:
    """Extract scopes from JWT claims.

    Supports:
      - ``scope``: space-separated string (OAuth2 convention)
      - ``scopes``: list of strings (convenience)
    """
    # Try "scope" (string)
    scope_str = claims.get("scope")
    if isinstance(scope_str, str) and scope_str.strip():
        return scope_str.strip().split()

    # Try "scopes" (list)
    scope_list = claims.get("scopes")
    if isinstance(scope_list, list):
        return [s for s in scope_list if isinstance(s, str)]

    return []
