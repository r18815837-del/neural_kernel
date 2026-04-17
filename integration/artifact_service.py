"""Artifact service — safe access to artifact metadata and download URLs."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from .dto.artifact_metadata import ArtifactMetadataDTO
from .dto.project_status import FeatureItemDTO, TechStackDTO
from .response_mapper import ResponseMapper


class ArtifactService:
    """Provides artifact-related operations for external clients.

    Wraps persistence and file-system checks behind a clean interface
    so that routes never touch Path objects or store internals directly.
    """

    def __init__(
        self,
        mapper: ResponseMapper | None = None,
        base_url: str = "/api/v1",
    ) -> None:
        self._mapper = mapper or ResponseMapper(base_url=base_url)
        self._base_url = base_url.rstrip("/")

    # ------------------------------------------------------------------
    # Artifact availability
    # ------------------------------------------------------------------

    def is_available(self, artifact_path: str | None) -> bool:
        """Return True if the artifact file exists on disk."""
        if not artifact_path:
            return False
        return Path(artifact_path).exists()

    def file_size(self, artifact_path: str | None) -> int | None:
        """Return file size in bytes, or None if missing."""
        if not artifact_path:
            return None
        p = Path(artifact_path)
        return p.stat().st_size if p.exists() else None

    # ------------------------------------------------------------------
    # Download URL
    # ------------------------------------------------------------------

    def download_url(self, project_id: str) -> str:
        return f"{self._base_url}/download/{project_id}"

    # ------------------------------------------------------------------
    # Metadata DTO
    # ------------------------------------------------------------------

    def get_metadata(
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
        """Build an ArtifactMetadataDTO if the artifact exists."""
        if not self.is_available(artifact_path):
            return None
        return self._mapper.to_artifact_metadata_dto(
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
    # Version helpers
    # ------------------------------------------------------------------

    def version_exists(self, artifact_path: str | None, version: int = 1) -> bool:
        """Check if a specific version file exists."""
        if not artifact_path:
            return False
        p = Path(artifact_path)
        if version <= 1:
            return p.exists()
        # Versioned filename pattern: name_v{N}.ext
        stem = p.stem
        suffix = p.suffix
        versioned = p.parent / f"{stem}_v{version}{suffix}"
        return versioned.exists()
