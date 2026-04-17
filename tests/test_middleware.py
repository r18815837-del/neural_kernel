"""Tests for API middleware — error handling, request IDs, rate limiting."""

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from api.middleware import (
    RateLimitMiddleware,
    RequestIdMiddleware,
    RequestLoggingMiddleware,
    error_response,
    register_exception_handlers,
)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _make_app(rate_limit: int = 5) -> FastAPI:
    """Build a tiny FastAPI app with all middleware wired up."""
    app = FastAPI()
    app.add_middleware(RequestIdMiddleware)
    app.add_middleware(RequestLoggingMiddleware)
    app.add_middleware(RateLimitMiddleware, max_requests=rate_limit, window_seconds=60)
    register_exception_handlers(app)

    @app.get("/ok")
    async def ok():
        return {"status": "ok"}

    @app.get("/fail")
    async def fail():
        raise RuntimeError("boom")

    @app.get("/bad")
    async def bad():
        raise ValueError("bad input")

    @app.get("/health")
    async def health():
        return {"up": True}

    @app.get("/api/v1/health")
    async def api_health():
        return {"up": True}

    return app


# ------------------------------------------------------------------
# error_response helper
# ------------------------------------------------------------------


class TestErrorResponse:
    def test_basic_error(self):
        resp = error_response(400, "bad", code="bad_request")
        assert resp.status_code == 400
        body = resp.body  # JSONResponse stores bytes
        assert b"bad_request" in body

    def test_error_with_details(self):
        resp = error_response(422, "invalid", details=[{"field": "x"}])
        assert resp.status_code == 422
        assert b"field" in resp.body

    def test_error_with_request_id(self):
        resp = error_response(500, "oops", request_id="req-123")
        assert b"req-123" in resp.body


# ------------------------------------------------------------------
# Request ID middleware
# ------------------------------------------------------------------


class TestRequestIdMiddleware:
    def test_generates_request_id(self):
        client = TestClient(_make_app())
        r = client.get("/ok")
        assert r.status_code == 200
        assert "x-request-id" in r.headers
        # Should be a valid UUID-ish string
        assert len(r.headers["x-request-id"]) >= 10

    def test_preserves_client_request_id(self):
        client = TestClient(_make_app())
        r = client.get("/ok", headers={"X-Request-ID": "my-custom-id"})
        assert r.headers["x-request-id"] == "my-custom-id"


# ------------------------------------------------------------------
# Global exception handlers
# ------------------------------------------------------------------


class TestExceptionHandlers:
    def test_unhandled_exception_returns_500(self):
        client = TestClient(_make_app(), raise_server_exceptions=False)
        r = client.get("/fail")
        assert r.status_code == 500
        data = r.json()
        assert data["error"]["code"] == "internal_error"
        assert "request_id" in data["error"]

    def test_value_error_returns_400(self):
        client = TestClient(_make_app(), raise_server_exceptions=False)
        r = client.get("/bad")
        assert r.status_code == 400
        assert r.json()["error"]["code"] == "bad_request"
        assert "bad input" in r.json()["error"]["message"]


# ------------------------------------------------------------------
# Rate limiting
# ------------------------------------------------------------------


class TestRateLimiting:
    def test_allows_requests_under_limit(self):
        client = TestClient(_make_app(rate_limit=5))
        for _ in range(5):
            r = client.get("/ok")
            assert r.status_code == 200

    def test_blocks_over_limit(self):
        client = TestClient(_make_app(rate_limit=3))
        for _ in range(3):
            client.get("/ok")
        r = client.get("/ok")
        assert r.status_code == 429
        assert "rate_limit_exceeded" in r.json()["error"]["code"]
        assert "retry_after_seconds" in r.json()["error"]["details"]

    def test_rate_limit_headers(self):
        client = TestClient(_make_app(rate_limit=10))
        r = client.get("/ok")
        assert "x-ratelimit-limit" in r.headers
        assert r.headers["x-ratelimit-limit"] == "10"
        assert "x-ratelimit-remaining" in r.headers

    def test_exempt_path_bypasses_limit(self):
        client = TestClient(_make_app(rate_limit=1))
        # Use up the limit
        client.get("/ok")
        # Exempt path should still work
        r = client.get("/api/v1/health")
        assert r.status_code == 200
