"""Project generation routes."""
from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query
from fastapi.responses import FileResponse

from api.auth.dependencies import optional_auth
from api.auth.models import AuthContext
from api.dependencies import get_assistant_manager, get_project_store, get_store
from api.schemas import (
    AgentResultDetail,
    FeatureDetail,
    GenerateProjectRequest,
    GenerateProjectResponse,
    PipelineDetails,
    ProjectDownloadInfo,
    ProjectListItem,
    ProjectListResponse,
    ProjectStatusResponse,
    ScaffoldValidation,
    TechStackDetail,
)
from nk_app.core.assistant_manager import AssistantManager
from persistence import BaseStore, StoredProject, StoredArtifact, RequestLog
from runtime.specs.client_request import ClientRequest

router = APIRouter(prefix="/api/v1", tags=["generation"])



def _generate_project_task(
    project_id: str,
    request: GenerateProjectRequest,
    manager: AssistantManager,
    project_store: Dict[str, Any],
    store: BaseStore,
    output_root: str,
) -> None:
    start_time = datetime.utcnow()

    try:
        project_store[project_id]["status"] = "in_progress"
        project_store[project_id]["progress"] = 0.1
        project_store[project_id]["started_at"] = start_time

        store.update_project_status(project_id, "in_progress")

        # Build ClientRequest
        user_id = request.user_id or str(uuid.uuid4())
        session_id = request.session_id or str(uuid.uuid4())

        client_request = ClientRequest(
            raw_text=request.text,
            user_id=user_id,
            session_id=session_id,
            metadata=request.metadata or {},
        )

        project_store[project_id]["progress"] = 0.3

        # Generate the project
        result = manager.generate_project(
            client_request=client_request,
            output_root=output_root,
        )

        project_store[project_id]["progress"] = 0.9

        # ---- Extract rich details from payload ----
        payload = result.payload or {}

        # Features
        features_detected: List[Dict[str, str]] = []
        project_name = ""
        tech_stack_dict: Dict[str, Any] = {}

        spec_data = payload.get("project_spec", {})
        if spec_data:
            project_name = spec_data.get("project_name", "")
            tech_stack_dict = spec_data.get("tech_stack", {}) or {}

        # Features can come from the spec or from agent results
        features_raw = spec_data.get("features", [])
        # Also check client_brief
        brief = payload.get("client_brief", {})
        if not features_raw and brief:
            features_raw = brief.get("requested_features", [])

        for f in (features_raw if isinstance(features_raw, list) else []):
            if isinstance(f, dict):
                features_detected.append({
                    "name": f.get("name", ""),
                    "description": f.get("description", ""),
                    "priority": f.get("priority", "medium"),
                })
            elif isinstance(f, str):
                features_detected.append({
                    "name": f, "description": "", "priority": "medium",
                })

        # Agent pipeline details
        agent_details: List[Dict[str, Any]] = []
        agent_results_raw = payload.get("agent_results", [])
        for ar in (agent_results_raw if isinstance(agent_results_raw, list) else []):
            if isinstance(ar, dict):
                outputs = ar.get("outputs", {})
                agent_details.append({
                    "agent_name": ar.get("agent_name", ""),
                    "role": ar.get("role", ""),
                    "success": ar.get("success", False),
                    "message": ar.get("message", ""),
                    "llm_powered": (ar.get("metadata") or {}).get("llm_powered", False),
                    "outputs_keys": list(outputs.keys()) if isinstance(outputs, dict) else [],
                })

        # Scaffold validation
        scaffold_data = payload.get("scaffold_validation", {})

        end_time = datetime.utcnow()
        duration_ms = int((end_time - start_time).total_seconds() * 1000)

        # Artifact
        artifact_path = None
        artifact_size = None
        if result.success and result.artifacts:
            artifact_path = result.artifacts[0]
            p = Path(artifact_path)
            if p.exists():
                artifact_size = p.stat().st_size

        # Update in-memory store
        project_store[project_id].update({
            "status": "completed" if result.success else "failed",
            "progress": 1.0,
            "message": result.message,
            "success": result.success,
            "artifacts": result.artifacts,
            "error": result.error,
            "artifact_path": artifact_path,
            "artifact_size_bytes": artifact_size,
            "project_name": project_name,
            "tech_stack": tech_stack_dict,
            "features_detected": features_detected,
            "agent_details": agent_details,
            "scaffold_validation": scaffold_data,
            "completed_at": end_time,
            "duration_ms": duration_ms,
        })

        # Persistent store
        store.update_project_status(
            project_id,
            "completed" if result.success else "failed",
            error=result.error,
        )

        if artifact_path and Path(artifact_path).exists():
            # Propagate ownership from in-memory project store
            _owner_cid = project_store[project_id].get("owner_client_id")
            _owner_uid = project_store[project_id].get("owner_user_id")
            _org_id = project_store[project_id].get("org_id")
            stored_artifact = StoredArtifact(
                artifact_id=str(uuid.uuid4()),
                project_id=project_id,
                filename=Path(artifact_path).name,
                file_path=str(artifact_path),
                file_size_bytes=artifact_size or 0,
                format=request.output_format,
                created_at=end_time,
                owner_client_id=_owner_cid,
                owner_user_id=_owner_uid,
                org_id=_org_id,
            )
            store.save_artifact(stored_artifact)

        # Log
        request_log = RequestLog(
            log_id=str(uuid.uuid4()),
            project_id=project_id,
            user_id=user_id,
            raw_text=request.text,
            parsed_features=[f["name"] for f in features_detected],
            parsed_tech_stack=tech_stack_dict,
            processing_time_ms=duration_ms,
            status="success" if result.success else "failed",
            created_at=end_time,
        )
        store.log_request(request_log)

    except Exception as e:
        end_time = datetime.utcnow()
        duration_ms = int((end_time - start_time).total_seconds() * 1000)
        project_store[project_id].update({
            "status": "failed",
            "error": str(e),
            "message": f"Generation failed: {e}",
            "progress": 1.0,
            "completed_at": end_time,
            "duration_ms": duration_ms,
        })
        store.update_project_status(project_id, "failed", error=str(e))



@router.post("/generate", response_model=GenerateProjectResponse)
async def generate_project(
    request: GenerateProjectRequest,
    background_tasks: BackgroundTasks,
    manager: AssistantManager = Depends(get_assistant_manager),
    project_store: Dict[str, Any] = Depends(get_project_store),
    store: BaseStore = Depends(get_store),
    auth: AuthContext = Depends(optional_auth),
) -> GenerateProjectResponse:
    project_id = str(uuid.uuid4())
    now = datetime.utcnow()
    user_id = request.user_id or str(uuid.uuid4())

    # Resolve ownership from auth context
    owner_client_id = auth.client_id if auth.authenticated else None
    owner_user_id = auth.user_id if auth.authenticated else None
    org_id = auth.org_id if auth.authenticated else None

    # In-memory store
    project_store[project_id] = {
        "status": "pending",
        "message": "Generation queued",
        "created_at": now,
        "progress": 0.0,
        "request": request.model_dump(),
        "artifacts": [],
        "error": None,
        "tech_stack": None,
        "features_detected": [],
        "artifact_path": None,
        "artifact_size_bytes": None,
        "project_name": "",
        "agent_details": [],
        "scaffold_validation": {},
        "started_at": None,
        "completed_at": None,
        "duration_ms": None,
        "owner_client_id": owner_client_id,
        "owner_user_id": owner_user_id,
        "org_id": org_id,
    }

    # Persistent store
    stored_project = StoredProject(
        project_id=project_id,
        user_id=user_id,
        session_id=request.session_id,
        raw_text=request.text,
        project_name="",
        summary=request.text[:200],
        project_type="application",
        features_json="[]",
        tech_stack_json="{}",
        status="pending",
        created_at=now,
        updated_at=now,
        owner_client_id=owner_client_id,
        owner_user_id=owner_user_id,
        org_id=org_id,
    )
    store.save_project(stored_project)

    # Queue background generation
    background_tasks.add_task(
        _generate_project_task,
        project_id,
        request,
        manager,
        project_store,
        store,
        "build",
    )

    return GenerateProjectResponse(
        project_id=project_id,
        status="pending",
        message="Generation queued",
        created_at=now,
    )


@router.get("/status/{project_id}", response_model=ProjectStatusResponse)
async def get_project_status(
    project_id: str,
    project_store: Dict[str, Any] = Depends(get_project_store),
    store: BaseStore = Depends(get_store),
) -> ProjectStatusResponse:
    # Check in-memory store first (active tasks)
    if project_id in project_store:
        p = project_store[project_id]

        # Build nested models
        features = [
            FeatureDetail(**f) if isinstance(f, dict) else FeatureDetail(name=f)
            for f in p.get("features_detected", [])
        ]

        tech = None
        ts = p.get("tech_stack")
        if isinstance(ts, dict) and any(ts.values()):
            tech = TechStackDetail(**{
                k: ts.get(k) for k in ("backend", "frontend", "database", "mobile", "deployment")
            })

        # Pipeline details
        pipeline = None
        agent_details = p.get("agent_details", [])
        if agent_details:
            agents = [AgentResultDetail(**a) for a in agent_details]
            sv = p.get("scaffold_validation", {})
            scaffold = None
            if sv:
                scaffold = ScaffoldValidation(
                    valid=not sv.get("missing_files") and not sv.get("empty_files"),
                    valid_files=sv.get("valid_files", []),
                    missing_files=sv.get("missing_files", []),
                    empty_files=sv.get("empty_files", []),
                )
            pipeline = PipelineDetails(
                agents=agents,
                total_agents=len(agents),
                successful_agents=sum(1 for a in agents if a.success),
                failed_agents=sum(1 for a in agents if not a.success),
                scaffold_validation=scaffold,
            )

        return ProjectStatusResponse(
            project_id=project_id,
            status=p["status"],
            message=p["message"],
            progress=p["progress"],
            artifact_path=p.get("artifact_path"),
            artifact_size_bytes=p.get("artifact_size_bytes"),
            project_name=p.get("project_name"),
            features_detected=features,
            tech_stack=tech,
            pipeline=pipeline,
            started_at=p.get("started_at"),
            completed_at=p.get("completed_at"),
            duration_ms=p.get("duration_ms"),
            error=p.get("error"),
        )

    # Fall back to persistent store
    stored = store.get_project(project_id)
    if stored:
        features_list = []
        try:
            raw = json.loads(stored.features_json) if stored.features_json else []
            features_list = [
                FeatureDetail(name=f) if isinstance(f, str) else FeatureDetail(**f)
                for f in raw
            ]
        except (json.JSONDecodeError, TypeError):
            pass

        tech = None
        try:
            ts = json.loads(stored.tech_stack_json) if stored.tech_stack_json else {}
            if isinstance(ts, dict) and any(ts.values()):
                tech = TechStackDetail(**{
                    k: ts.get(k)
                    for k in ("backend", "frontend", "database", "mobile", "deployment")
                })
        except (json.JSONDecodeError, TypeError):
            pass

        progress = 1.0 if stored.status in ("completed", "failed") else 0.0

        return ProjectStatusResponse(
            project_id=project_id,
            status=stored.status,
            message=stored.error_message or f"Project {stored.status}",
            progress=progress,
            artifact_path=stored.artifact_path,
            project_name=stored.project_name,
            features_detected=features_list,
            tech_stack=tech,
            error=stored.error_message,
        )

    raise HTTPException(status_code=404, detail="Project not found")


@router.get("/download/{project_id}")
async def download_project(
    project_id: str,
    project_store: Dict[str, Any] = Depends(get_project_store),
    store: BaseStore = Depends(get_store),
) -> FileResponse:
    artifact_path = None
    status = None

    if project_id in project_store:
        project = project_store[project_id]
        status = project["status"]
        artifact_path = project.get("artifact_path")
    else:
        stored = store.get_project(project_id)
        if stored:
            status = stored.status
            artifact_path = stored.artifact_path
        else:
            raise HTTPException(status_code=404, detail="Project not found")

    if status != "completed":
        raise HTTPException(
            status_code=400,
            detail=f"Project not completed. Current status: {status}",
        )

    if not artifact_path:
        raise HTTPException(
            status_code=404, detail="Artifact not found for this project"
        )

    path = Path(artifact_path)
    if not path.exists():
        raise HTTPException(
            status_code=404, detail="Artifact file not found on disk"
        )

    # Determine content type
    content_type = "application/zip" if path.suffix == ".zip" else "application/octet-stream"

    return FileResponse(
        path=path,
        filename=path.name,
        media_type=content_type,
        headers={
            "Content-Disposition": f'attachment; filename="{path.name}"',
        },
    )


@router.get("/download/{project_id}/info", response_model=ProjectDownloadInfo)
async def download_info(
    project_id: str,
    project_store: Dict[str, Any] = Depends(get_project_store),
    store: BaseStore = Depends(get_store),
) -> ProjectDownloadInfo:
    artifact_path = None
    features: List[str] = []

    if project_id in project_store:
        p = project_store[project_id]
        if p["status"] != "completed":
            raise HTTPException(status_code=400, detail="Not completed yet")
        artifact_path = p.get("artifact_path")
        features = [
            f["name"] if isinstance(f, dict) else f
            for f in p.get("features_detected", [])
        ]
    else:
        stored = store.get_project(project_id)
        if not stored:
            raise HTTPException(status_code=404, detail="Project not found")
        artifact_path = stored.artifact_path
        try:
            features = json.loads(stored.features_json) if stored.features_json else []
        except (json.JSONDecodeError, TypeError):
            pass

    if not artifact_path:
        raise HTTPException(status_code=404, detail="Artifact not found")

    path = Path(artifact_path)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Artifact file missing on disk")

    content_type = "application/zip" if path.suffix == ".zip" else "application/octet-stream"

    return ProjectDownloadInfo(
        project_id=project_id,
        filename=path.name,
        file_size=path.stat().st_size,
        content_type=content_type,
        features_included=features,
    )


@router.get("/projects", response_model=ProjectListResponse)
async def list_projects(
    user_id: str | None = None,
    limit: int = Query(50, ge=1, le=200, description="Max projects to return"),
    offset: int = Query(0, ge=0, description="Pagination offset"),
    store: BaseStore = Depends(get_store),
) -> ProjectListResponse:
    projects = store.list_projects(user_id=user_id, limit=limit, offset=offset)

    items = []
    for p in projects:
        features = []
        try:
            features = json.loads(p.features_json) if p.features_json else []
            if isinstance(features, list):
                features = [
                    f if isinstance(f, str) else f.get("name", "")
                    for f in features
                ]
        except (json.JSONDecodeError, TypeError):
            pass

        items.append(ProjectListItem(
            project_id=p.project_id,
            project_name=p.project_name,
            status=p.status,
            created_at=p.created_at,
            features=features,
            artifact_path=p.artifact_path,
        ))

    return ProjectListResponse(
        projects=items,
        total=len(items),
        limit=limit,
        offset=offset,
    )
