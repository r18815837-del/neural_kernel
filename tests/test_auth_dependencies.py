"""Tests for auth service and dependency behavior."""
from __future__ import annotations
import os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.auth.config import AuthConfig
from api.auth.service import AuthService
from api.auth.models import AuthContext, ANONYMOUS
from api.auth.errors import AuthenticationError


def test_auth_disabled_returns_anonymous():
    svc = AuthService(config=AuthConfig(auth_enabled=False))
    ctx = svc.authenticate()
    assert ctx.is_anonymous is True
    assert ctx is ANONYMOUS


def test_auth_enabled_no_creds_raises():
    svc = AuthService(config=AuthConfig(auth_enabled=True, client_api_key="secret"))
    try:
        svc.authenticate()
        assert False, "Should raise"
    except AuthenticationError as e:
        assert e.code == "auth_required"


def test_auth_enabled_valid_key():
    svc = AuthService(config=AuthConfig(auth_enabled=True, client_api_key="secret"))
    ctx = svc.authenticate(api_key="secret")
    assert ctx.authenticated is True
    assert ctx.auth_type == "api_key"


def test_auth_enabled_invalid_key():
    svc = AuthService(config=AuthConfig(auth_enabled=True, client_api_key="secret"))
    try:
        svc.authenticate(api_key="wrong")
        assert False, "Should raise"
    except AuthenticationError as e:
        assert e.code == "api_key_invalid"


def test_bearer_not_enabled():
    svc = AuthService(config=AuthConfig(auth_enabled=True, client_api_key="", bearer_enabled=False))
    try:
        svc.authenticate(authorization="Bearer some-token")
        assert False, "Should raise"
    except AuthenticationError:
        pass  # Expected: bearer not enabled or not implemented


def test_bearer_enabled_but_not_implemented():
    svc = AuthService(config=AuthConfig(auth_enabled=True, bearer_enabled=True))
    try:
        svc.authenticate(authorization="Bearer fake-jwt-token")
        assert False, "Should raise"
    except AuthenticationError as e:
        assert e.code == "bearer_not_implemented"


def test_unsupported_auth_scheme():
    svc = AuthService(config=AuthConfig(auth_enabled=True, client_api_key="secret"))
    try:
        svc.authenticate(authorization="Basic dXNlcjpwYXNz")
        assert False, "Should raise"
    except AuthenticationError as e:
        assert e.code == "auth_scheme_unsupported"


def test_anonymous_path_check():
    svc = AuthService(config=AuthConfig(
        auth_enabled=True,
        anonymous_paths=["/api/v1/health", "/api/v1/info"],
    ))
    assert svc.is_anonymous_path("/api/v1/health") is True
    assert svc.is_anonymous_path("/api/v1/info") is True
    assert svc.is_anonymous_path("/api/v1/client/status/p1") is False
    assert svc.is_anonymous_path("/api/v1/health/deep") is True  # prefix match


def test_auth_context_has_scope():
    ctx = AuthContext(authenticated=True, scopes=["client", "read"])
    assert ctx.has_scope("client") is True
    assert ctx.has_scope("admin") is False


def test_auth_context_is_anonymous():
    assert ANONYMOUS.is_anonymous is True
    ctx = AuthContext(authenticated=True, auth_type="api_key")
    assert ctx.is_anonymous is False


if __name__ == "__main__":
    import traceback
    tests = [v for k, v in list(globals().items()) if k.startswith("test_") and callable(v)]
    passed = failed = 0
    for t in tests:
        try: t(); passed += 1; print(f"  PASS: {t.__name__}")
        except: failed += 1; print(f"  FAIL: {t.__name__}"); traceback.print_exc()
    print(f"\n{passed} passed, {failed} failed out of {passed+failed}")
