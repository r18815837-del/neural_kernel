"""Central auth service — single entry point for credential validation."""
from __future__ import annotations

import logging
from typing import Optional

from .api_key import API_KEY_HEADER, validate_api_key
from .bearer import validate_bearer_token
from .config import AuthConfig, get_auth_config
from .errors import AuthenticationError
from .models import ANONYMOUS, AuthContext

logger = logging.getLogger(__name__)

# Authorization header prefix
_BEARER_PREFIX = "Bearer "


class AuthService:
    """Validates credentials and produces an AuthContext.

    Tries auth methods in order:
    1. API key (``X-API-Key`` header)
    2. Bearer token (``Authorization: Bearer ...`` header)
    3. Falls through → anonymous or 401, depending on config.
    """

    def __init__(self, config: AuthConfig | None = None) -> None:
        self._config = config or get_auth_config()

    @property
    def config(self) -> AuthConfig:
        return self._config

    def authenticate(
        self,
        api_key: Optional[str] = None,
        authorization: Optional[str] = None,
    ) -> AuthContext:
        """Attempt authentication from available credentials.

        Args:
            api_key: Value of the ``X-API-Key`` header.
            authorization: Value of the ``Authorization`` header.

        Returns:
            AuthContext — always populated; may be anonymous.

        Raises:
            AuthenticationError: if auth is enabled but credentials
                are missing or invalid.
        """
        # If auth is disabled, return anonymous-ok immediately
        if not self._config.auth_enabled:
            return ANONYMOUS

        # Try API key first (preferred for machine-to-machine)
        if api_key and self._config.has_api_key:
            return validate_api_key(api_key, self._config)

        # Try bearer token
        bearer_token = self._extract_bearer(authorization)
        if bearer_token and self._config.bearer_enabled:
            return validate_bearer_token(bearer_token, self._config)

        # Nothing matched — decide based on what was provided
        if api_key:
            # Key was provided but didn't match config
            return validate_api_key(api_key, self._config)

        if authorization:
            if bearer_token:
                return validate_bearer_token(bearer_token, self._config)
            raise AuthenticationError(
                message="Unsupported authorization scheme",
                code="auth_scheme_unsupported",
            )

        # No credentials at all
        raise AuthenticationError(
            message="Authentication required — send X-API-Key or Authorization header",
            code="auth_required",
        )

    def is_anonymous_path(self, path: str) -> bool:
        """Check if the given URL path is exempt from auth."""
        for anon_path in self._config.anonymous_paths:
            if path == anon_path or path.startswith(anon_path + "/"):
                return True
        return False

    @staticmethod
    def _extract_bearer(authorization: str | None) -> str | None:
        """Extract the raw token from ``Authorization: Bearer <token>``."""
        if not authorization:
            return None
        if authorization.startswith(_BEARER_PREFIX):
            return authorization[len(_BEARER_PREFIX):].strip() or None
        return None


# Singleton
_service_instance: AuthService | None = None


def get_auth_service() -> AuthService:
    """Return the singleton AuthService."""
    global _service_instance
    if _service_instance is None:
        _service_instance = AuthService()
    return _service_instance


def reset_auth_service() -> None:
    """Reset singleton — used by tests."""
    global _service_instance
    _service_instance = None
