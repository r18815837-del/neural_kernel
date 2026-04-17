"""Tests for JWT bearer token validation.

Validates token decode, claim extraction, role/scope mapping,
error handling for expired/invalid tokens.
"""
from __future__ import annotations

import time
from datetime import datetime, timedelta

import pytest

# PyJWT must be available for these tests
jwt = pytest.importorskip("jwt")

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from api.auth.bearer import validate_bearer_token, _claims_to_context
from api.auth.config import AuthConfig
from api.auth.errors import AuthenticationError
from api.auth.models import (
    ROLE_ADMIN, ROLE_CLIENT, ROLE_OPERATOR, ROLE_VIEWER,
    SCOPE_PROJECT_CREATE, SCOPE_PROJECT_READ,
)


SECRET = "test-secret-key-for-jwt"
ALGORITHM = "HS256"


def _config(**overrides) -> AuthConfig:
    defaults = dict(
        auth_enabled=True,
        bearer_enabled=True,
        jwt_secret=SECRET,
        jwt_algorithm=ALGORITHM,
    )
    defaults.update(overrides)
    return AuthConfig(**defaults)


def _encode(claims: dict, secret: str = SECRET, algorithm: str = ALGORITHM) -> str:
    return jwt.encode(claims, secret, algorithm=algorithm)


def _valid_claims(**extra) -> dict:
    now = int(time.time())
    base = {
        "sub": "user-123",
        "iat": now,
        "exp": now + 3600,
    }
    base.update(extra)
    return base


# ------------------------------------------------------------------
# Happy path
# ------------------------------------------------------------------


class TestJWTValidation:
    def test_basic_valid_token(self):
        token = _encode(_valid_claims())
        ctx = validate_bearer_token(token, _config())
        assert ctx.authenticated is True
        assert ctx.auth_type == "bearer"
        assert ctx.raw_subject == "user-123"

    def test_client_id_from_sub_fallback(self):
        token = _encode(_valid_claims())
        ctx = validate_bearer_token(token, _config())
        assert ctx.client_id == "user-123"  # sub used as fallback

    def test_explicit_client_id(self):
        token = _encode(_valid_claims(client_id="my-client"))
        ctx = validate_bearer_token(token, _config())
        assert ctx.client_id == "my-client"

    def test_user_id_extraction(self):
        token = _encode(_valid_claims(user_id="uid-456"))
        ctx = validate_bearer_token(token, _config())
        assert ctx.user_id == "uid-456"

    def test_org_id_extraction(self):
        token = _encode(_valid_claims(org_id="org-abc"))
        ctx = validate_bearer_token(token, _config())
        assert ctx.org_id == "org-abc"

    def test_role_extraction(self):
        token = _encode(_valid_claims(role="operator"))
        ctx = validate_bearer_token(token, _config())
        assert ctx.role == ROLE_OPERATOR

    def test_unknown_role_defaults_to_client(self):
        token = _encode(_valid_claims(role="superuser"))
        ctx = validate_bearer_token(token, _config())
        assert ctx.role == ROLE_CLIENT

    def test_default_role_is_client(self):
        token = _encode(_valid_claims())
        ctx = validate_bearer_token(token, _config())
        assert ctx.role == ROLE_CLIENT


# ------------------------------------------------------------------
# Scopes
# ------------------------------------------------------------------


class TestJWTScopes:
    def test_space_separated_scope(self):
        token = _encode(_valid_claims(scope="project:create project:read"))
        ctx = validate_bearer_token(token, _config())
        assert "project:create" in ctx.scopes
        assert "project:read" in ctx.scopes

    def test_scopes_list(self):
        token = _encode(_valid_claims(scopes=["project:create", "admin"]))
        ctx = validate_bearer_token(token, _config())
        assert "admin" in ctx.scopes

    def test_default_scopes_from_role(self):
        token = _encode(_valid_claims(role="admin"))
        ctx = validate_bearer_token(token, _config())
        assert SCOPE_PROJECT_CREATE in ctx.scopes
        assert "admin" in ctx.scopes

    def test_explicit_scopes_override_role_defaults(self):
        token = _encode(_valid_claims(role="admin", scope="custom:only"))
        ctx = validate_bearer_token(token, _config())
        assert ctx.scopes == ["custom:only"]


# ------------------------------------------------------------------
# Issuer / Audience validation
# ------------------------------------------------------------------


class TestJWTIssuerAudience:
    def test_valid_issuer(self):
        token = _encode(_valid_claims(iss="neural-kernel"))
        ctx = validate_bearer_token(token, _config(jwt_issuer="neural-kernel"))
        assert ctx.authenticated

    def test_invalid_issuer_raises(self):
        token = _encode(_valid_claims(iss="wrong-issuer"))
        with pytest.raises(AuthenticationError) as exc_info:
            validate_bearer_token(token, _config(jwt_issuer="neural-kernel"))
        assert exc_info.value.code == "token_invalid_issuer"

    def test_valid_audience(self):
        token = _encode(_valid_claims(aud="nk-api"))
        ctx = validate_bearer_token(token, _config(jwt_audience="nk-api"))
        assert ctx.authenticated

    def test_invalid_audience_raises(self):
        token = _encode(_valid_claims(aud="wrong-aud"))
        with pytest.raises(AuthenticationError) as exc_info:
            validate_bearer_token(token, _config(jwt_audience="nk-api"))
        assert exc_info.value.code == "token_invalid_audience"


# ------------------------------------------------------------------
# Error cases
# ------------------------------------------------------------------


class TestJWTErrors:
    def test_expired_token(self):
        claims = _valid_claims()
        claims["exp"] = int(time.time()) - 100
        token = _encode(claims)
        with pytest.raises(AuthenticationError) as exc_info:
            validate_bearer_token(token, _config())
        assert exc_info.value.code == "token_expired"

    def test_bad_signature(self):
        token = _encode(_valid_claims(), secret="wrong-secret")
        with pytest.raises(AuthenticationError) as exc_info:
            validate_bearer_token(token, _config())
        assert exc_info.value.code in ("token_decode_error", "token_invalid")

    def test_missing_sub_claim(self):
        claims = _valid_claims()
        del claims["sub"]
        token = _encode(claims)
        with pytest.raises(AuthenticationError) as exc_info:
            validate_bearer_token(token, _config())
        assert exc_info.value.code == "token_missing_sub"

    def test_missing_token(self):
        with pytest.raises(AuthenticationError) as exc_info:
            validate_bearer_token(None, _config())
        assert exc_info.value.code == "bearer_token_missing"

    def test_bearer_not_enabled(self):
        token = _encode(_valid_claims())
        with pytest.raises(AuthenticationError) as exc_info:
            validate_bearer_token(token, _config(bearer_enabled=False))
        assert exc_info.value.code == "bearer_not_enabled"

    def test_no_jwt_key_configured(self):
        token = _encode(_valid_claims())
        with pytest.raises(AuthenticationError) as exc_info:
            validate_bearer_token(token, _config(jwt_secret=""))
        assert exc_info.value.code == "jwt_key_missing"

    def test_malformed_token(self):
        with pytest.raises(AuthenticationError) as exc_info:
            validate_bearer_token("not.a.jwt", _config())
        assert "decode" in exc_info.value.code or "invalid" in exc_info.value.code

    def test_leeway_allows_slightly_expired(self):
        claims = _valid_claims()
        claims["exp"] = int(time.time()) - 3
        token = _encode(claims)
        ctx = validate_bearer_token(token, _config(jwt_leeway=10))
        assert ctx.authenticated


# ------------------------------------------------------------------
# BearerPrincipal.to_auth_context
# ------------------------------------------------------------------


class TestBearerPrincipalToContext:
    def test_principal_conversion(self):
        from api.auth.models import BearerPrincipal
        p = BearerPrincipal(
            subject="sub-1",
            client_id="cli-1",
            user_id="usr-1",
            org_id="org-1",
            role="operator",
        )
        ctx = p.to_auth_context()
        assert ctx.authenticated is True
        assert ctx.auth_type == "bearer"
        assert ctx.client_id == "cli-1"
        assert ctx.user_id == "usr-1"
        assert ctx.org_id == "org-1"
        assert ctx.role == "operator"
        assert SCOPE_PROJECT_CREATE in ctx.scopes

    def test_principal_sub_as_client_id_fallback(self):
        from api.auth.models import BearerPrincipal
        p = BearerPrincipal(subject="sub-1")
        ctx = p.to_auth_context()
        assert ctx.client_id == "sub-1"
