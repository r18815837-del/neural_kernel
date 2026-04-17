"""Client contract facade — single entry point for API routes.

Routes call this service instead of manually assembling DTO fields.
This keeps route handlers thin and ensures all external responses
pass through the integration boundary.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from .artifact_service import ArtifactService
from .dto.create_project import CreateProjectResponseDTO
from .dto.project_status import ProjectStatusDTO
from .dto.artifact_metadata import ArtifactMetadataDTO
from .dto.project_list import ProjectListResponseDTO
from .dto.lifecycle import (
    LifecycleStateDTO,
    LifecycleTransitionDTO,
    RetainVersionsResponseDTO,
    VersionInfoDTO,
)
from .dto.errors import ErrorDTO, NotFoundErrorDTO, ValidationErrorDTO
from .response_mapper import ResponseMapper
from .status_mapper import StatusMapper


class ClientContractService:
    """Facade consumed by API route handlers.

    Usage in a route::

        contract = ClientContractService()
        dto = contract.project_status(project_id, project_store[project_id])
        return dto.to_dict()
    """

    def __init__(
        self,
        base_url: str = "/api/v1",
    ) -> None:
        self._mapper = ResponseMapper(base_url=base_url)
        self._artifact = ArtifactService(mapper=self._mapper, base_url=base_url)

    # ------------------------------------------------------------------
    # Properties for route-level access
    # ------------------------------------------------------------------

    @property
    def mapper(self) -> ResponseMapper:
        return self._mapper

    @property
    def artifact_service(self) -> ArtifactService:
        return self._artifact

    # ------------------------------------------------------------------
    # Create project
    # ------------------------------------------------------------------

    def create_project_response(
        self,
        project_id: str,
        status: str = "pending",
        message: str = "Generation queued",
        project_name: str = "",
        created_at: datetime | str | None = None,
    ) -> CreateProjectResponseDTO:
        return self._mapper.to_create_project_dto(
            project_id=project_id,
            status=status,
            message=message,
            project_name=project_name,
            created_at=created_at,
        )

    # ------------------------------------------------------------------
    # Project status
    # ------------------------------------------------------------------

    def project_status(
        self,
        project_id: str,
        data: Dict[str, Any],
    ) -> ProjectStatusDTO:
        """Map in-memory project data → ProjectStatusDTO."""
        return self._mapper.to_project_status_dto(project_id, data)

    def project_status_from_stored(
        self,
        stored: Any,
    ) -> ProjectStatusDTO:
        """Map persistence StoredProject → ProjectStatusDTO."""
        return self._mapper.to_project_status_dto_from_stored(stored)

    # ------------------------------------------------------------------
    # Artifact metadata
    # ------------------------------------------------------------------

    def artifact_metadata(
        self,
        project_id: str,
        project_name: str,
        artifact_path: str,
        features: List[Any] | None = None,
        tech_stack: Dict[str, Any] | None = None,
        created_at: Any = None,
        version: int | None = None,
        packaging_format: str = "zip",
    ) -> ArtifactMetadataDTO | None:
        return self._artifact.get_metadata(
            project_id=project_id,
            project_name=project_name,
            artifact_path=artifact_path,
            features=features,
            tech_stack=tech_stack,
            created_at=created_at,
            version=version,
            packaging_format=packaging_format,
        )

    # ------------------------------------------------------------------
    # Project list
    # ------------------------------------------------------------------

    def project_list(
        self,
        stored_projects: List[Any],
        total: int | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> ProjectListResponseDTO:
        return self._mapper.to_project_list_dto(
            stored_projects, total=total, limit=limit, offset=offset,
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def transition_response(
        self,
        project_id: str,
        old_status: str,
        new_status: str,
    ) -> LifecycleTransitionDTO:
        return self._mapper.to_transition_dto(project_id, old_status, new_status)

    def lifecycle_state(
        self,
        project_id: str,
        current_status: str,
        allowed: List[str],
    ) -> LifecycleStateDTO:
        return self._mapper.to_lifecycle_state_dto(project_id, current_status, allowed)

    # ------------------------------------------------------------------
    # Errors
    # ------------------------------------------------------------------

    def not_found(
        self,
        resource_type: str = "project",
        resource_id: str = "",
    ) -> NotFoundErrorDTO:
        return self._mapper.to_not_found_dto(resource_type, resource_id)

    def validation_error(
        self,
        message: str,
        field_errors: List[Dict[str, str]] | None = None,
    ) -> ValidationErrorDTO:
        return self._mapper.to_validation_error_dto(message, field_errors)

    def error(
        self,
        code: str,
        message: str,
        retryable: bool = False,
        details: Dict[str, Any] | None = None,
    ) -> ErrorDTO:
        return self._mapper.to_error_dto(code, message, details, retryable)
