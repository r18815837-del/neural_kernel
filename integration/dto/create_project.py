"""DTO returned immediately when a generation job is queued."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class CreateProjectResponseDTO:
    """Stable client-facing response for POST /generate."""

    project_id: str
    project_name: str
    status: str  # pending | in_progress | completed | failed
    message: str
    created_at: str  # ISO 8601
    artifact_available: bool = False
    artifact_name: Optional[str] = None

    def to_dict(self) -> dict[str, object]:
        return {
            "project_id": self.project_id,
            "project_name": self.project_name,
            "status": self.status,
            "message": self.message,
            "created_at": self.created_at,
            "artifact_available": self.artifact_available,
            "artifact_name": self.artifact_name,
        }
