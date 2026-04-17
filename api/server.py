"""FastAPI server for Neural Kernel."""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.config import ApiConfig
from api.middleware import (
    RateLimitMiddleware,
    RequestIdMiddleware,
    RequestLoggingMiddleware,
    register_exception_handlers,
)
from api.routes import (
    code_router,
    cognition_router,
    generation_router,
    health_router,
    lifecycle_router,
    client_router,
    sessions_router,
)

logger = logging.getLogger("neural_kernel.api")


def create_app(config: ApiConfig | None = None) -> FastAPI:
    if config is None:
        config = ApiConfig()

    logging.basicConfig(
        level=logging.DEBUG if config.debug else logging.INFO,
        format="%(asctime)s %(levelname)-8s [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        logger.info("Neural Kernel API started on %s:%s", config.host, config.port)
        yield
        logger.info("Neural Kernel API shutting down")

    app = FastAPI(
        title="Neural Kernel API",
        version="0.1.0",
        debug=config.debug,
        lifespan=lifespan,
    )

    app.add_middleware(RequestIdMiddleware)
    app.add_middleware(RequestLoggingMiddleware)
    app.add_middleware(
        RateLimitMiddleware,
        max_requests=config.rate_limit_rpm,
        window_seconds=60,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    register_exception_handlers(app)

    app.include_router(health_router)
    app.include_router(cognition_router)
    app.include_router(code_router)
    app.include_router(generation_router)
    app.include_router(lifecycle_router)
    app.include_router(client_router)
    app.include_router(sessions_router)

    app.state.config = config
    return app


app = create_app()
