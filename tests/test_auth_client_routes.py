"""Tests for auth integration with client routes — simulated request flow."""
from __future__ import annotations
import os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.auth.config import AuthConfig
from api.auth.service import AuthService
from api.auth.models import ANONYMOUS, AuthContext
from api.auth.errors import AuthenticationError


def test_client_endpoint_requires_auth_when_enabled():
    """Simulate: auth enabled, no key → 401."""
    svc = AuthService(config=AuthConfig(auth_enabled=True, client_api_key="my-key"))
    path = "/api/v1/client/status/p1"
    assert not svc.is_anonymous_path(path)
    try:
        svc.authenticate()
        assert False, "Should require auth"
    except AuthenticationError:
        pass


def test_client_endpoint_with_valid_key():
    """Simulate: auth enabled, valid key → authenticated."""
    svc = AuthService(config=AuthConfig(auth_enabled=True, client_api_key="my-key"))
    ctx = svc.authenticate(api_key="my-key")
    assert ctx.authenticated is True
    assert ctx.auth_type == "api_key"


def test_client_endpoint_with_invalid_key():
    """Simulate: auth enabled, wrong key → 401."""
    svc = AuthService(config=AuthConfig(auth_enabled=True, client_api_key="my-key"))
    try:
        svc.authenticate(api_key="wrong-key")
        assert False, "Should reject"
    except AuthenticationError as e:
        assert e.code == "api_key_invalid"


def test_health_endpoint_anonymous():
    """Health endpoint should be in anonymous paths."""
    svc = AuthService(config=AuthConfig(auth_enabled=True, client_api_key="key"))
    assert svc.is_anonymous_path("/api/v1/health") is True
    assert svc.is_anonymous_path("/api/v1/info") is True


def test_dev_mode_allows_all():
    """Auth disabled → everything passes as anonymous."""
    svc = AuthService(config=AuthConfig(auth_enabled=False))
    ctx = svc.authenticate()
    assert ctx.is_anonymous is True
    ctx2 = svc.authenticate(api_key="anything")
    assert ctx2.is_anonymous is True


def test_auth_context_fields_for_client():
    """AuthContext should have all expected fields for routes."""
    svc = AuthService(config=AuthConfig(auth_enabled=True, client_api_key="secret"))
    ctx = svc.authenticate(api_key="secret")
    assert isinstance(ctx.authenticated, bool)
    assert isinstance(ctx.auth_type, str)
    assert isinstance(ctx.client_id, str)
    assert isinstance(ctx.scopes, list)
    assert ctx.raw_subject is not None


def test_error_shape_stable():
    """Auth errors must have code and message for client parsing."""
    svc = AuthService(config=AuthConfig(auth_enabled=True, client_api_key="secret"))
    try:
        svc.authenticate()
    except AuthenticationError as e:
        assert hasattr(e, "code")
        assert hasattr(e, "message")
        assert isinstance(e.code, str)
        assert isinstance(e.message, str)


def test_api_key_header_name():
    """Verify the header name is stable."""
    from api.auth.api_key import API_KEY_HEADER
    assert API_KEY_HEADER == "X-API-Key"


def test_docs_openapi_anonymous():
    """OpenAPI and docs paths should be anonymous."""
    svc = AuthService(config=AuthConfig(auth_enabled=True, client_api_key="key"))
    assert svc.is_anonymous_path("/docs") is True
    assert svc.is_anonymous_path("/openapi.json") is True
    assert svc.is_anonymous_path("/redoc") is True


def test_config_from_env_defaults():
    """Default config has auth disabled."""
    # Clear env vars to get defaults
    for k in ("NK_AUTH_ENABLED", "NK_CLIENT_API_KEY", "NK_AUTH_BEARER_ENABLED"):
        os.environ.pop(k, None)
    cfg = AuthConfig.from_env()
    assert cfg.auth_enabled is False
    assert cfg.client_api_key == ""
    assert cfg.bearer_enabled is False


def test_config_from_env_enabled():
    """Config reads env vars correctly."""
    os.environ["NK_AUTH_ENABLED"] = "true"
    os.environ["NK_CLIENT_API_KEY"] = "env-key-123"
    try:
        cfg = AuthConfig.from_env()
        assert cfg.auth_enabled is True
        assert cfg.client_api_key == "env-key-123"
    finally:
        os.environ.pop("NK_AUTH_ENABLED", None)
        os.environ.pop("NK_CLIENT_API_KEY", None)


if __name__ == "__main__":
    import traceback
    tests = [v for k, v in list(globals().items()) if k.startswith("test_") and callable(v)]
    passed = failed = 0
    for t in tests:
        try: t(); passed += 1; print(f"  PASS: {t.__name__}")
        except: failed += 1; print(f"  FAIL: {t.__name__}"); traceback.print_exc()
    print(f"\n{passed} passed, {failed} failed out of {passed+failed}")
