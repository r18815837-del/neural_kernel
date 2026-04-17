"""Tests for API key validation logic."""
from __future__ import annotations
import os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import directly to avoid __init__.py pulling fastapi
from api.auth.config import AuthConfig
from api.auth.api_key import validate_api_key
from api.auth.errors import AuthenticationError


def _cfg(key: str = "test-secret-key-123") -> AuthConfig:
    return AuthConfig(auth_enabled=True, client_api_key=key)


def test_valid_key():
    ctx = validate_api_key("test-secret-key-123", _cfg())
    assert ctx.authenticated is True
    assert ctx.auth_type == "api_key"
    assert ctx.client_id == "api_key_client"
    assert "client" in ctx.scopes


def test_missing_key():
    try:
        validate_api_key(None, _cfg())
        assert False, "Should raise"
    except AuthenticationError as e:
        assert e.code == "api_key_missing"


def test_empty_key():
    try:
        validate_api_key("", _cfg())
        assert False, "Should raise"
    except AuthenticationError as e:
        assert e.code == "api_key_missing"


def test_invalid_key():
    try:
        validate_api_key("wrong-key", _cfg())
        assert False, "Should raise"
    except AuthenticationError as e:
        assert e.code == "api_key_invalid"


def test_no_key_configured():
    cfg = AuthConfig(auth_enabled=True, client_api_key="")
    try:
        validate_api_key("anything", cfg)
        assert False, "Should raise"
    except AuthenticationError as e:
        assert e.code == "auth_not_configured"


def test_raw_subject_truncated():
    ctx = validate_api_key("test-secret-key-123", _cfg())
    assert ctx.raw_subject is not None
    assert len(ctx.raw_subject) < 20


def test_timing_safe_comparison():
    try:
        validate_api_key("short", _cfg("a-much-longer-key-that-differs"))
        assert False, "Should raise"
    except AuthenticationError as e:
        assert e.code == "api_key_invalid"


if __name__ == "__main__":
    import traceback
    tests = [v for k, v in list(globals().items()) if k.startswith("test_") and callable(v)]
    passed = failed = 0
    for t in tests:
        try: t(); passed += 1; print(f"  PASS: {t.__name__}")
        except: failed += 1; print(f"  FAIL: {t.__name__}"); traceback.print_exc()
    print(f"\n{passed} passed, {failed} failed out of {passed+failed}")
