"""Map internal structures → client-facing DTOs.

Single responsibility: take raw internal data (dicts, persistence models,
Pydantic responses) and produce clean DTOs.  Never exposes internal
blobs to the client.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .dto.create_project import CreateProjectResponseDTO
from .dto.project_status import (
    FeatureItemDTO,
    ProjectStatusDTO,
    QualityScoreDTO,
    TechStackDTO,
)
from .dto.artifact_metadata import ArtifactMetadataDTO
from .dto.project_list import ProjectListItemDTO, ProjectListResponseDTO
from .dto.lifecycle import (
    LifecycleTransitionDTO,
    LifecycleStateDTO,
    VersionInfoDTO,
    RetainVersionsResponseDTO,
)
from .dto.errors import ErrorDTO, NotFoundErrorDTO, ValidationErrorDTO
from .status_mapper import StatusMapper


class ResponseMapper:
    """Stateless mapper from internal representations to DTOs."""

    def __init__(self, base_url: str = "/api/v1") -> None:
        self._base_url = base_url.rstrip("/")

    # ------------------------------------------------------------------
    # Create project
    # ------------------------------------------------------------------

    def to_create_project_dto(
        self,
        project_id: str,
        status: str = "pending",
        message: str = "Generation queued",
        project_name: str = "",
        created_at: datetime | str | None = None,
        artifact_path: str | None = None,
    ) -> CreateProjectResponseDTO:
        ts = self._iso(created_at) if created_at else self._iso(datetime.utcnow())
        return CreateProjectResponseDTO(
            project_id=project_id,
            project_name=project_name,
            status=status,
            message=message,
            created_at=ts,
            artifact_available=StatusMapper.artifact_available(status, artifact_path),
            artifact_name=Path(artifact_path).name if artifact_path else None,
        )

    # ------------------------------------------------------------------
    # Project status
    # ------------------------------------------------------------------

    def to_project_status_dto(
        self,
        project_id: str,
        data: Dict[str, Any],
    ) -> ProjectStatusDTO:
        """Map an in-memory project store dict → ProjectStatusDTO."""
        status = data.get("status", "pending")
        agent_details = data.get("agent_details", [])
        total, ok, fail = StatusMapper.agent_counts(agent_details)
        artifact_path = data.get("artifact_path")

        # Quality
        quality_raw = StatusMapper.quality_score(data)
        quality = QualityScoreDTO(**quality_raw) if quality_raw else None

        return ProjectStatusDTO(
            project_id=project_id,
            project_name=data.get("project_name", ""),
            status=status,
            status_label=StatusMapper.status_label(status),
            message=StatusMapper.derive_message(
                status, data.get("message"), data.get("error"),
            ),
            created_at=self._iso(data.get("created_at")),
            updated_at=self._iso(data.get("started_at")),
            completed_at=self._iso(data.get("completed_at")),
            progress_percent=StatusMapper.progress_percent(
                status, data.get("progress"),
            ),
            artifact_available=StatusMapper.artifact_available(status, artifact_path),
            artifact_name=Path(artifact_path).name if artifact_path else None,
            artifact_size_bytes=data.get("artifact_size_bytes"),
            download_url=self._download_url(project_id) if StatusMapper.artifact_available(status, artifact_path) else None,
            features=self._map_features(data.get("features_detected", [])),
            tech_stack=self._map_tech_stack(data.get("tech_stack")),
            quality=quality,
            execution_ready=StatusMapper.execution_ready(data),
            llm_used=StatusMapper.llm_used(agent_details),
            agent_count=total,
            successful_agent_count=ok,
            failed_agent_count=fail,
            error=data.get("error"),
        )

    def to_project_status_dto_from_stored(
        self,
        stored: Any,  # StoredProject
    ) -> ProjectStatusDTO:
        """Map a StoredProject persistence model → ProjectStatusDTO."""
        features_raw = self._parse_json(stored.features_json, [])
        tech_raw = self._parse_json(stored.tech_stack_json, {})

        status = stored.status
        artifact_path = stored.artifact_path

        return ProjectStatusDTO(
            project_id=stored.project_id,
            project_name=stored.project_name or "",
            status=status,
            status_label=StatusMapper.status_label(status),
            message=StatusMapper.derive_message(status, None, stored.error_message),
            created_at=self._iso(stored.created_at),
            updated_at=self._iso(stored.updated_at),
            progress_percent=StatusMapper.progress_percent(status),
            artifact_available=StatusMapper.artifact_available(status, artifact_path),
            artifact_name=Path(artifact_path).name if artifact_path else None,
            download_url=self._download_url(stored.project_id) if StatusMapper.artifact_available(status, artifact_path) else None,
            features=self._map_features(features_raw),
            tech_stack=self._map_tech_stack(tech_raw),
            error=stored.error_message,
        )

    # ------------------------------------------------------------------
    # Artifact metadata
    # ------------------------------------------------------------------

    def to_artifact_metadata_dto(
        self,
        project_id: str,
        project_name: str,
        artifact_path: str,
        features: List[Any] | None = None,
        tech_stack: Dict[str, Any] | None = None,
        created_at: datetime | str | None = None,
        version: int | None = None,
        packaging_format: str = "zip",
    ) -> ArtifactMetadataDTO:
        p = Path(artifact_path)
        size = p.stat().st_size if p.exists() else 0
        return ArtifactMetadataDTO(
            project_id=project_id,
            project_name=project_name,
            artifact_name=p.name,
            artifact_size_bytes=size,
            packaging_format=packaging_format,
            download_url=self._download_url(project_id),
            features=self._map_features(features or []),
            tech_stack=self._map_tech_stack(tech_stack),
            created_at=self._iso(created_at) if created_at else "",
            version=version,
        )

    # ------------------------------------------------------------------
    # Project list
    # ------------------------------------------------------------------

    def to_project_list_dto(
        self,
        stored_projects: List[Any],  # List[StoredProject]
        total: int | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> ProjectListResponseDTO:
        items = []
        for p in stored_projects:
            features_raw = self._parse_json(p.features_json, [])
            feature_names = [
                f if isinstance(f, str) else (f.get("name", "") if isinstance(f, dict) else "")
                for f in features_raw
            ]
            items.append(ProjectListItemDTO(
                project_id=p.project_id,
                project_name=p.project_name or "",
                status=p.status,
                status_label=StatusMapper.status_label(p.status),
                created_at=self._iso(p.created_at),
                features=feature_names,
                artifact_available=StatusMapper.artifact_available(p.status, p.artifact_path),
            ))
        return ProjectListResponseDTO(
            projects=items,
            total=total if total is not None else len(items),
            limit=limit,
            offset=offset,
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def to_transition_dto(
        self,
        project_id: str,
        old_status: str,
        new_status: str,
        message: str = "",
    ) -> LifecycleTransitionDTO:
        return LifecycleTransitionDTO(
            project_id=project_id,
            old_status=old_status,
            new_status=new_status,
            message=message or f"Transitioned from {old_status} to {new_status}",
        )

    def to_lifecycle_state_dto(
        self,
        project_id: str,
        current_status: str,
        allowed: List[str],
    ) -> LifecycleStateDTO:
        return LifecycleStateDTO(
            project_id=project_id,
            current_status=current_status,
            allowed_transitions=allowed,
        )

    def to_version_info_dto(self, data: Dict[str, Any]) -> VersionInfoDTO:
        return VersionInfoDTO(
            artifact_id=data.get("artifact_id", ""),
            version=data.get("version", 0),
            filename=data.get("filename", ""),
            file_size_bytes=data.get("file_size_bytes", 0),
            format=data.get("format", ""),
            created_at=str(data.get("created_at", "")),
            exists_on_disk=data.get("exists_on_disk", False),
        )

    # ------------------------------------------------------------------
    # Errors
    # ------------------------------------------------------------------

    def to_error_dto(
        self,
        code: str,
        message: str,
        details: Dict[str, Any] | None = None,
        retryable: bool = False,
    ) -> ErrorDTO:
        return ErrorDTO(
            code=code,
            message=message,
            details=details or {},
            retryable=retryable,
        )

    def to_not_found_dto(
        self,
        resource_type: str = "project",
        resource_id: str = "",
        message: str = "",
    ) -> NotFoundErrorDTO:
        return NotFoundErrorDTO(
            code=f"{resource_type}_not_found",
            message=message or f"{resource_type.title()} not found",
            resource_type=resource_type,
            resource_id=resource_id,
        )

    def to_validation_error_dto(
        self,
        message: str,
        field_errors: List[Dict[str, str]] | None = None,
    ) -> ValidationErrorDTO:
        return ValidationErrorDTO(
            code="validation_error",
            message=message,
            field_errors=field_errors or [],
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _download_url(self, project_id: str) -> str:
        return f"{self._base_url}/download/{project_id}"

    @staticmethod
    def _iso(dt: datetime | str | None) -> str:
        if dt is None:
            return ""
        if isinstance(dt, str):
            return dt
        return dt.isoformat()

    @staticmethod
    def _parse_json(raw: str | None, default: Any) -> Any:
        if not raw:
            return default
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return default

    @staticmethod
    def _map_features(raw: List[Any]) -> List[FeatureItemDTO]:
        result: List[FeatureItemDTO] = []
        for f in raw:
            if isinstance(f, dict):
                result.append(FeatureItemDTO(
                    name=f.get("name", ""),
                    description=f.get("description", ""),
                    priority=f.get("priority", "medium"),
                ))
            elif isinstance(f, str):
                result.append(FeatureItemDTO(name=f))
        return result

    @staticmethod
    def _map_tech_stack(raw: Dict[str, Any] | None) -> TechStackDTO | None:
        if not raw or not isinstance(raw, dict):
            return None
        if not any(raw.get(k) for k in ("backend", "frontend", "database", "mobile", "deployment")):
            return None
        return TechStackDTO(
            backend=raw.get("backend"),
            frontend=raw.get("frontend"),
            database=raw.get("database"),
            mobile=raw.get("mobile"),
            deployment=raw.get("deployment"),
        )
