"""Lifecycle, versioning, and cleanup routes."""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query

from api.dependencies import get_cleanup_service, get_lifecycle, get_version_manager
from api.schemas import (
    AllowedTransitionsResponse, ArtifactVersionItem, ArtifactVersionsResponse,
    CleanupResponse, RetentionResponse, TransitionResponse,
)
from persistence.lifecycle import (
    ArtifactVersionManager, CleanupService, InvalidTransitionError, ProjectLifecycle,
)

router = APIRouter(prefix="/api/v1", tags=["lifecycle"])


@router.post("/projects/{project_id}/transition", response_model=TransitionResponse)
async def transition_project(
    project_id: str,
    new_status: str = Query(...),
    error: Optional[str] = Query(None),
    lifecycle: ProjectLifecycle = Depends(get_lifecycle),
) -> TransitionResponse:
    old = lifecycle.current_status(project_id)
    if old is None:
        raise HTTPException(status_code=404, detail="Project not found")
    try:
        lifecycle.transition(project_id, new_status, error=error)
    except InvalidTransitionError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    return TransitionResponse(project_id=project_id, old_status=old, new_status=new_status)


@router.get("/projects/{project_id}/transitions", response_model=AllowedTransitionsResponse)
async def allowed_transitions(
    project_id: str,
    lifecycle: ProjectLifecycle = Depends(get_lifecycle),
) -> AllowedTransitionsResponse:
    current = lifecycle.current_status(project_id)
    if current is None:
        raise HTTPException(status_code=404, detail="Project not found")
    return AllowedTransitionsResponse(
        project_id=project_id, current_status=current,
        allowed=lifecycle.allowed_transitions(current),
    )


@router.post("/projects/{project_id}/archive", response_model=TransitionResponse)
async def archive_project(
    project_id: str,
    lifecycle: ProjectLifecycle = Depends(get_lifecycle),
) -> TransitionResponse:
    old = lifecycle.current_status(project_id)
    if old is None:
        raise HTTPException(status_code=404, detail="Project not found")
    try:
        lifecycle.transition(project_id, "archived")
    except InvalidTransitionError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    return TransitionResponse(project_id=project_id, old_status=old, new_status="archived")


@router.post("/projects/{project_id}/retry", response_model=TransitionResponse)
async def retry_project(
    project_id: str,
    lifecycle: ProjectLifecycle = Depends(get_lifecycle),
) -> TransitionResponse:
    old = lifecycle.current_status(project_id)
    if old is None:
        raise HTTPException(status_code=404, detail="Project not found")
    try:
        lifecycle.transition(project_id, "pending")
    except InvalidTransitionError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    return TransitionResponse(project_id=project_id, old_status=old, new_status="pending")


@router.get("/projects/{project_id}/versions", response_model=ArtifactVersionsResponse)
async def list_versions(
    project_id: str,
    vm: ArtifactVersionManager = Depends(get_version_manager),
) -> ArtifactVersionsResponse:
    versions = vm.list_versions(project_id)
    items = [ArtifactVersionItem(**v) for v in versions]
    return ArtifactVersionsResponse(project_id=project_id, versions=items, total=len(items))


@router.post("/projects/{project_id}/retain", response_model=RetentionResponse)
async def retain_versions(
    project_id: str,
    keep: int = Query(3, ge=1, le=100),
    vm: ArtifactVersionManager = Depends(get_version_manager),
) -> RetentionResponse:
    deleted = vm.retain_latest(project_id, keep=keep)
    return RetentionResponse(project_id=project_id, kept=keep, deleted_files=deleted)


@router.post("/admin/cleanup", response_model=CleanupResponse)
async def run_cleanup(
    cleanup: CleanupService = Depends(get_cleanup_service),
) -> CleanupResponse:
    return CleanupResponse(**cleanup.run())
