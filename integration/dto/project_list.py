"""DTOs for the project list screen."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass(frozen=True)
class ProjectListItemDTO:
    """Single row in the project list — minimal, scannable."""

    project_id: str
    project_name: str
    status: str
    status_label: str
    created_at: str  # ISO 8601
    features: List[str] = field(default_factory=list)
    artifact_available: bool = False

    def to_dict(self) -> dict[str, object]:
        return {
            "project_id": self.project_id,
            "project_name": self.project_name,
            "status": self.status,
            "status_label": self.status_label,
            "created_at": self.created_at,
            "features": self.features,
            "artifact_available": self.artifact_available,
        }


@dataclass(frozen=True)
class ProjectListResponseDTO:
    """Paginated project list."""

    projects: List[ProjectListItemDTO] = field(default_factory=list)
    total: int = 0
    limit: int = 50
    offset: int = 0

    def to_dict(self) -> dict[str, object]:
        return {
            "projects": [p.to_dict() for p in self.projects],
            "total": self.total,
            "limit": self.limit,
            "offset": self.offset,
        }
