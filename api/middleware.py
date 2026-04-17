"""API middleware — request IDs, rate limiting, error handling."""
from __future__ import annotations

import logging
import time
import uuid
from collections import defaultdict
from typing import Callable

from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse
from pydantic import ValidationError
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger("neural_kernel.api")


def error_response(
    status_code: int,
    message: str,
    code: str = "internal_error",
    details: dict | list | None = None,
    request_id: str | None = None,
) -> JSONResponse:
    body: dict = {"error": {"code": code, "message": message}}
    if details is not None:
        body["error"]["details"] = details
    if request_id:
        body["error"]["request_id"] = request_id
    return JSONResponse(status_code=status_code, content=body)


class RequestIdMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        request.state.request_id = request_id
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        start = time.perf_counter()
        response = await call_next(request)
        duration_ms = (time.perf_counter() - start) * 1000
        request_id = getattr(request.state, "request_id", "-")
        logger.info(
            "%s %s -> %d (%.1fms) [%s]",
            request.method, request.url.path,
            response.status_code, duration_ms, request_id,
        )
        return response


class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(
        self,
        app,
        max_requests: int = 60,
        window_seconds: int = 60,
        exempt_paths: tuple[str, ...] = ("/api/v1/health", "/docs", "/openapi.json"),
    ):
        super().__init__(app)
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.exempt_paths = exempt_paths
        self._requests: dict[str, list[float]] = defaultdict(list)

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        if request.url.path in self.exempt_paths:
            return await call_next(request)

        client_ip = request.client.host if request.client else "unknown"
        now = time.time()
        cutoff = now - self.window_seconds

        timestamps = self._requests[client_ip]
        self._requests[client_ip] = [t for t in timestamps if t > cutoff]
        timestamps = self._requests[client_ip]

        if len(timestamps) >= self.max_requests:
            retry_after = int(self.window_seconds - (now - timestamps[0])) + 1
            request_id = getattr(request.state, "request_id", None)
            return error_response(
                status_code=429,
                message="Too many requests. Please slow down.",
                code="rate_limit_exceeded",
                details={"retry_after_seconds": retry_after},
                request_id=request_id,
            )

        timestamps.append(now)
        response = await call_next(request)

        remaining = max(0, self.max_requests - len(timestamps))
        response.headers["X-RateLimit-Limit"] = str(self.max_requests)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        response.headers["X-RateLimit-Reset"] = str(int(cutoff + self.window_seconds))
        return response


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(ValidationError)
    async def pydantic_validation_handler(request: Request, exc: ValidationError):
        request_id = getattr(request.state, "request_id", None)
        errors = []
        for e in exc.errors():
            errors.append({
                "field": " -> ".join(str(loc) for loc in e["loc"]),
                "message": e["msg"],
                "type": e["type"],
            })
        return error_response(
            status_code=422, message="Validation error",
            code="validation_error", details=errors, request_id=request_id,
        )

    @app.exception_handler(ValueError)
    async def value_error_handler(request: Request, exc: ValueError):
        request_id = getattr(request.state, "request_id", None)
        return error_response(
            status_code=400, message=str(exc),
            code="bad_request", request_id=request_id,
        )

    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        request_id = getattr(request.state, "request_id", None)
        logger.exception(
            "Unhandled exception on %s %s [%s]",
            request.method, request.url.path, request_id,
        )
        return error_response(
            status_code=500,
            message="An internal error occurred. Please try again later.",
            code="internal_error", request_id=request_id,
        )
