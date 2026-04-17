"""Client-facing routes — clean DTO responses via integration boundary.

These endpoints mirror the existing generation routes but return
client-friendly DTOs through the ClientContractService.  The existing
routes remain unchanged for backward compatibility.

Endpoint prefix: ``/api/v1/client``
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse

from api.auth.access import (
    AccessDeniedError,
    verify_project_access,
    filter_projects_by_access,
    get_caller_client_id,
    get_caller_org_id,
)
from api.auth.dependencies import require_client_auth
from api.auth.models import AuthContext
from api.dependencies import (
    get_access_policy,
    get_contract_service,
    get_project_store,
    get_store,
)
from integration.client_contract import ClientContractService
from persistence import BaseStore
from persistence.access import OwnershipAccessPolicy

router = APIRouter(prefix="/api/v1/client", tags=["client"])


# ------------------------------------------------------------------
# Project status — client-facing
# ------------------------------------------------------------------


@router.get(
    "/status/{project_id}",
    summary="Get project status (client-friendly)",
    description="Returns a flat, Flutter-ready status DTO with pre-computed fields.",
)
async def client_project_status(
    project_id: str,
    auth: AuthContext = Depends(require_client_auth),
    project_store: Dict[str, Any] = Depends(get_project_store),
    store: BaseStore = Depends(get_store),
    contract: ClientContractService = Depends(get_contract_service),
    policy: OwnershipAccessPolicy = Depends(get_access_policy),
) -> JSONResponse:
    try:
        # In-memory (active)
        if project_id in project_store:
            p = project_store[project_id]
            verify_project_access(p, auth, policy, project_id=project_id)
            dto = contract.project_status(project_id, p)
            return JSONResponse(content=dto.to_dict())

        # Persistent
        stored = store.get_project(project_id)
        if stored:
            verify_project_access(stored, auth, policy, project_id=project_id)
            dto = contract.project_status_from_stored(stored)
            return JSONResponse(content=dto.to_dict())

    except AccessDeniedError:
        pass  # fall through to 404

    err = contract.not_found("project", project_id)
    return JSONResponse(status_code=404, content=err.to_dict())


# ------------------------------------------------------------------
# Project list — client-facing
# ------------------------------------------------------------------


@router.get(
    "/projects",
    summary="List projects (client-friendly)",
    description="Returns paginated project list with clean DTOs.",
)
async def client_list_projects(
    user_id: str | None = None,
    limit: int = 50,
    offset: int = 0,
    auth: AuthContext = Depends(require_client_auth),
    store: BaseStore = Depends(get_store),
    contract: ClientContractService = Depends(get_contract_service),
    policy: OwnershipAccessPolicy = Depends(get_access_policy),
) -> JSONResponse:
    # Use DB-level owner + org filter for efficiency
    caller_cid = get_caller_client_id(auth)
    caller_oid = get_caller_org_id(auth)
    projects = store.list_projects(
        user_id=user_id,
        limit=limit,
        offset=offset,
        owner_client_id=caller_cid,
        org_id=caller_oid,
    )
    # Belt-and-suspenders: policy filter on results
    projects = filter_projects_by_access(projects, auth, policy)
    dto = contract.project_list(projects, limit=limit, offset=offset)
    return JSONResponse(content=dto.to_dict())


# ------------------------------------------------------------------
# Download info — client-facing
# ------------------------------------------------------------------


@router.get(
    "/download/{project_id}/info",
    summary="Artifact metadata (client-friendly)",
    description="Returns artifact metadata DTO with download_url, features, tech_stack.",
)
async def client_download_info(
    project_id: str,
    auth: AuthContext = Depends(require_client_auth),
    project_store: Dict[str, Any] = Depends(get_project_store),
    store: BaseStore = Depends(get_store),
    contract: ClientContractService = Depends(get_contract_service),
    policy: OwnershipAccessPolicy = Depends(get_access_policy),
) -> JSONResponse:
    artifact_path: str | None = None
    project_name: str = ""
    features: List[Any] = []
    tech_stack: Dict[str, Any] = {}
    created_at = None

    try:
        if project_id in project_store:
            p = project_store[project_id]
            verify_project_access(p, auth, policy, project_id=project_id)
            if p["status"] != "completed":
                err = contract.error(
                    "not_ready",
                    f"Project not completed. Current status: {p['status']}",
                    retryable=p["status"] in ("pending", "in_progress"),
                )
                return JSONResponse(status_code=400, content=err.to_dict())
            artifact_path = p.get("artifact_path")
            project_name = p.get("project_name", "")
            features = p.get("features_detected", [])
            tech_stack = p.get("tech_stack") or {}
            created_at = p.get("completed_at")
        else:
            stored = store.get_project(project_id)
            if not stored:
                err = contract.not_found("project", project_id)
                return JSONResponse(status_code=404, content=err.to_dict())
            verify_project_access(stored, auth, policy, project_id=project_id)
            artifact_path = stored.artifact_path
            project_name = stored.project_name or ""
            try:
                features = json.loads(stored.features_json) if stored.features_json else []
            except (json.JSONDecodeError, TypeError):
                pass
            try:
                tech_stack = json.loads(stored.tech_stack_json) if stored.tech_stack_json else {}
            except (json.JSONDecodeError, TypeError):
                pass
            created_at = stored.updated_at

    except AccessDeniedError:
        err = contract.not_found("project", project_id)
        return JSONResponse(status_code=404, content=err.to_dict())

    if not artifact_path or not Path(artifact_path).exists():
        err = contract.not_found("artifact", project_id)
        return JSONResponse(status_code=404, content=err.to_dict())

    dto = contract.artifact_metadata(
        project_id=project_id,
        project_name=project_name,
        artifact_path=artifact_path,
        features=features,
        tech_stack=tech_stack,
        created_at=created_at,
    )
    if dto is None:
        err = contract.not_found("artifact", project_id)
        return JSONResponse(status_code=404, content=err.to_dict())

    return JSONResponse(content=dto.to_dict())
