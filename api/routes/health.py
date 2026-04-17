"""Health check and service info routes."""
from __future__ import annotations

import time

from fastapi import APIRouter

from api.schemas import HealthResponse, InfoResponse

router = APIRouter(prefix="/api/v1", tags=["health"])

_start_time = time.time()

# Pipeline agent roles in execution order
_PIPELINE_ROLES = [
    "generalist",       # Intake / requirement parser
    "product_manager",  # Feature grouping, MVP scope
    "architect",        # Project structure, tech design
    "backend",          # Code generation per feature
    "docs",             # README, API docs
    "qa",               # Test strategy, test files
    "release",          # Dockerfile, CI/CD, packaging
]

# Features supported by the registry-based feature orchestrator
_SUPPORTED_FEATURES = [
    "auth",
    "roles",
    "admin_panel",
    "client_database",
    "export",
    "appointments",
]


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Health check endpoint.

    Returns status, version, uptime, and availability of subsystems.
    """
    uptime = time.time() - _start_time

    # Check LLM availability
    llm_available = False
    try:
        from llm.config import LLMConfig
        cfg = LLMConfig.from_env()
        llm_available = bool(cfg.api_key)
    except Exception:
        pass

    return HealthResponse(
        status="ok",
        version="0.2.0",
        uptime_seconds=uptime,
        llm_available=llm_available,
        pipeline_ready=True,
    )


@router.get("/info", response_model=InfoResponse)
async def api_info() -> InfoResponse:
    """API information — endpoints, supported features, pipeline agents."""
    return InfoResponse(
        name="Neural Kernel API",
        version="0.2.0",
        description="AI-powered project generation API",
        supported_features=_SUPPORTED_FEATURES,
        endpoints={
            "generate": "POST /api/v1/generate",
            "status": "GET  /api/v1/status/{project_id}",
            "download": "GET  /api/v1/download/{project_id}",
            "download_info": "GET  /api/v1/download/{project_id}/info",
            "projects": "GET  /api/v1/projects",
            "health": "GET  /api/v1/health",
            "info": "GET  /api/v1/info",
        },
        pipeline_agents=_PIPELINE_ROLES,
    )
