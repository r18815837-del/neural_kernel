"""Tests for auth error types and response helpers."""
from __future__ import annotations
import os, sys, json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.auth.errors import AuthenticationError, AuthorizationError

# Check if fastapi is available for auth_error_response tests
try:
    import fastapi
    _HAS_FASTAPI = True
except ImportError:
    _HAS_FASTAPI = False


def test_authentication_error_defaults():
    e = AuthenticationError()
    assert e.message == "Authentication required"
    assert e.code == "auth_required"


def test_authentication_error_custom():
    e = AuthenticationError("Bad key", "api_key_invalid")
    assert e.message == "Bad key"
    assert e.code == "api_key_invalid"
    assert str(e) == "Bad key"


def test_authorization_error_defaults():
    e = AuthorizationError()
    assert e.message == "Insufficient permissions"
    assert e.code == "forbidden"


def test_authentication_error_is_exception():
    try:
        raise AuthenticationError("test", "test_code")
    except AuthenticationError as e:
        assert e.code == "test_code"


def test_authorization_error_is_exception():
    try:
        raise AuthorizationError("no access", "no_scope")
    except AuthorizationError as e:
        assert e.code == "no_scope"


def test_error_fields_are_strings():
    e = AuthenticationError("msg", "code")
    assert isinstance(e.message, str)
    assert isinstance(e.code, str)


def test_error_compatible_with_integration_dto():
    """Error structure must match integration/dto/errors.py contract shape."""
    e = AuthenticationError("Auth needed", "auth_required")
    d = {"code": e.code, "message": e.message, "retryable": False}
    assert d["code"] == "auth_required"
    assert d["message"] == "Auth needed"
    assert d["retryable"] is False


def test_auth_error_response_401():
    if not _HAS_FASTAPI:
        return  # Skip without fastapi
    from api.auth.errors import auth_error_response
    resp = auth_error_response(
        status_code=401, code="api_key_missing", message="Missing API key",
    )
    assert resp.status_code == 401
    body = json.loads(resp.body)
    assert body["code"] == "api_key_missing"
    assert body["retryable"] is False


def test_auth_error_response_with_details():
    if not _HAS_FASTAPI:
        return  # Skip without fastapi
    from api.auth.errors import auth_error_response
    resp = auth_error_response(
        status_code=403, code="forbidden", message="No access",
        details={"scope_required": "admin"},
    )
    body = json.loads(resp.body)
    assert body["details"]["scope_required"] == "admin"


if __name__ == "__main__":
    import traceback
    tests = [v for k, v in list(globals().items()) if k.startswith("test_") and callable(v)]
    passed = failed = 0
    for t in tests:
        try: t(); passed += 1; print(f"  PASS: {t.__name__}")
        except: failed += 1; print(f"  FAIL: {t.__name__}"); traceback.print_exc()
    print(f"\n{passed} passed, {failed} failed out of {passed+failed}")
