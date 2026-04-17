"""Auth configuration — reads env vars with safe dev defaults."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass(frozen=True)
class AuthConfig:
    """Immutable auth configuration.

    Env vars:
        NK_AUTH_ENABLED                — "true" to enforce auth (default: false)
        NK_CLIENT_API_KEY              — shared secret for X-API-Key header
        NK_AUTH_BEARER_ENABLED         — "true" to enable bearer/JWT path
        NK_AUTH_ANONYMOUS_PATHS        — comma-separated paths that skip auth
        NK_ALLOW_LEGACY_OWNERLESS_ACCESS — "true" to let authenticated clients
                                           see pre-ownership resources

        JWT-specific:
        NK_JWT_SECRET                  — HMAC secret (HS256/HS384/HS512)
        NK_JWT_PUBLIC_KEY              — RSA/EC public key PEM (RS256/ES256)
        NK_JWT_ALGORITHM               — algorithm, default HS256
        NK_JWT_ISSUER                  — expected ``iss`` claim (optional)
        NK_JWT_AUDIENCE                — expected ``aud`` claim (optional)
        NK_JWT_LEEWAY                  — seconds of clock skew tolerance (0)
    """

    # --- General auth ---
    auth_enabled: bool = False
    client_api_key: str = ""
    bearer_enabled: bool = False

    # --- Ownership policy ---
    allow_legacy_ownerless_access: bool = True

    # --- JWT ---
    jwt_secret: str = ""            # for HMAC algorithms
    jwt_public_key: str = ""        # for RSA / EC algorithms (PEM string)
    jwt_algorithm: str = "HS256"
    jwt_issuer: Optional[str] = None
    jwt_audience: Optional[str] = None
    jwt_leeway: int = 0             # seconds of clock-skew tolerance

    # --- Anonymous paths ---
    anonymous_paths: List[str] = field(default_factory=lambda: [
        "/api/v1/health",
        "/api/v1/info",
        "/docs",
        "/openapi.json",
        "/redoc",
    ])

    # --- Derived helpers ---

    @property
    def has_api_key(self) -> bool:
        return bool(self.client_api_key)

    @property
    def has_jwt_key(self) -> bool:
        """True if there is a signing key configured for JWT."""
        return bool(self.jwt_secret) or bool(self.jwt_public_key)

    @property
    def jwt_decode_key(self) -> str:
        """Return whichever key should be used for decoding JWT tokens."""
        return self.jwt_public_key or self.jwt_secret

    # --- Factory ---

    @classmethod
    def from_env(cls) -> AuthConfig:
        """Load config from environment variables."""
        return cls(
            auth_enabled=_bool_env("NK_AUTH_ENABLED", False),
            client_api_key=os.getenv("NK_CLIENT_API_KEY", ""),
            bearer_enabled=_bool_env("NK_AUTH_BEARER_ENABLED", False),
            allow_legacy_ownerless_access=_bool_env("NK_ALLOW_LEGACY_OWNERLESS_ACCESS", True),
            jwt_secret=os.getenv("NK_JWT_SECRET", ""),
            jwt_public_key=os.getenv("NK_JWT_PUBLIC_KEY", ""),
            jwt_algorithm=os.getenv("NK_JWT_ALGORITHM", "HS256"),
            jwt_issuer=os.getenv("NK_JWT_ISSUER") or None,
            jwt_audience=os.getenv("NK_JWT_AUDIENCE") or None,
            jwt_leeway=int(os.getenv("NK_JWT_LEEWAY", "0")),
            anonymous_paths=_parse_anonymous_paths(),
        )


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _bool_env(name: str, default: bool) -> bool:
    raw = os.getenv(name, str(default).lower())
    return raw.lower() in ("true", "1", "yes")


def _parse_anonymous_paths() -> list[str]:
    """Parse NK_AUTH_ANONYMOUS_PATHS or return defaults."""
    raw = os.getenv("NK_AUTH_ANONYMOUS_PATHS")
    if raw:
        return [p.strip() for p in raw.split(",") if p.strip()]
    return [
        "/api/v1/health",
        "/api/v1/info",
        "/docs",
        "/openapi.json",
        "/redoc",
    ]


# ------------------------------------------------------------------
# Singleton
# ------------------------------------------------------------------

_config_instance: AuthConfig | None = None


def get_auth_config() -> AuthConfig:
    """Return the singleton AuthConfig (loaded once from env)."""
    global _config_instance
    if _config_instance is None:
        _config_instance = AuthConfig.from_env()
    return _config_instance


def reset_auth_config() -> None:
    """Reset singleton — used by tests."""
    global _config_instance
    _config_instance = None
