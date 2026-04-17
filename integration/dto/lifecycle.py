"""DTOs for lifecycle and versioning actions."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass(frozen=True)
class LifecycleTransitionDTO:
    """Result of a status transition."""

    project_id: str
    old_status: str
    new_status: str
    message: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "project_id": self.project_id,
            "old_status": self.old_status,
            "new_status": self.new_status,
            "message": self.message,
        }


@dataclass(frozen=True)
class LifecycleStateDTO:
    """Current lifecycle state with allowed transitions."""

    project_id: str
    current_status: str
    allowed_transitions: List[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "project_id": self.project_id,
            "current_status": self.current_status,
            "allowed_transitions": self.allowed_transitions,
        }


@dataclass(frozen=True)
class VersionInfoDTO:
    """Single artifact version."""

    artifact_id: str
    version: int
    filename: str
    file_size_bytes: int
    format: str
    created_at: str  # ISO 8601
    exists_on_disk: bool = False

    def to_dict(self) -> dict[str, object]:
        return {
            "artifact_id": self.artifact_id,
            "version": self.version,
            "filename": self.filename,
            "file_size_bytes": self.file_size_bytes,
            "format": self.format,
            "created_at": self.created_at,
            "exists_on_disk": self.exists_on_disk,
        }


@dataclass(frozen=True)
class RetainVersionsResponseDTO:
    """Result of version retention/pruning."""

    project_id: str
    kept: int
    deleted_files: List[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "project_id": self.project_id,
            "kept": self.kept,
            "deleted_files": self.deleted_files,
        }
