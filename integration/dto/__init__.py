"""Client-facing data transfer objects.

These DTOs define the stable external contract. Internal structures
(Pydantic response models, persistence dataclasses, workflow payloads)
are mapped into these before leaving the API boundary.
"""
from __future__ import annotations

from .create_project import CreateProjectResponseDTO
from .project_status import ProjectStatusDTO
from .artifact_metadata import ArtifactMetadataDTO
from .project_list import ProjectListItemDTO, ProjectListResponseDTO
from .lifecycle import (
    LifecycleTransitionDTO,
    LifecycleStateDTO,
    VersionInfoDTO,
    RetainVersionsResponseDTO,
)
from .errors import ErrorDTO, ValidationErrorDTO, NotFoundErrorDTO

__all__ = [
    "CreateProjectResponseDTO",
    "ProjectStatusDTO",
    "ArtifactMetadataDTO",
    "ProjectListItemDTO",
    "ProjectListResponseDTO",
    "LifecycleTransitionDTO",
    "LifecycleStateDTO",
    "VersionInfoDTO",
    "RetainVersionsResponseDTO",
    "ErrorDTO",
    "ValidationErrorDTO",
    "NotFoundErrorDTO",
]
