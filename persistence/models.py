from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional


@dataclass
class StoredProject:
    """Data model for persisted projects."""

    project_id: str
    user_id: Optional[str]
    session_id: Optional[str]
    raw_text: str
    project_name: str
    summary: str
    project_type: str
    features_json: str
    tech_stack_json: str
    status: str  # pending, in_progress, completed, failed
    created_at: datetime
    updated_at: datetime
    artifact_path: Optional[str] = None
    error_message: Optional[str] = None

    # Ownership fields — nullable for backward compatibility with
    # pre-ownership rows and internal/admin-created projects.
    owner_client_id: Optional[str] = None
    owner_user_id: Optional[str] = None
    org_id: Optional[str] = None

    def to_dict(self) -> Dict[str, object]:
        """Convert to dictionary representation."""
        d: Dict[str, object] = {
            "project_id": self.project_id,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "raw_text": self.raw_text,
            "project_name": self.project_name,
            "summary": self.summary,
            "project_type": self.project_type,
            "features_json": self.features_json,
            "tech_stack_json": self.tech_stack_json,
            "status": self.status,
            "artifact_path": self.artifact_path,
            "error_message": self.error_message,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }
        if self.owner_client_id is not None:
            d["owner_client_id"] = self.owner_client_id
        if self.owner_user_id is not None:
            d["owner_user_id"] = self.owner_user_id
        if self.org_id is not None:
            d["org_id"] = self.org_id
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> StoredProject:
        """Create from dictionary representation."""
        return cls(
            project_id=str(data["project_id"]),
            user_id=data.get("user_id") if data.get("user_id") else None,
            session_id=data.get("session_id") if data.get("session_id") else None,
            raw_text=str(data["raw_text"]),
            project_name=str(data["project_name"]),
            summary=str(data["summary"]),
            project_type=str(data["project_type"]),
            features_json=str(data["features_json"]),
            tech_stack_json=str(data["tech_stack_json"]),
            status=str(data["status"]),
            artifact_path=data.get("artifact_path") if data.get("artifact_path") else None,
            error_message=data.get("error_message") if data.get("error_message") else None,
            created_at=datetime.fromisoformat(str(data["created_at"])),
            updated_at=datetime.fromisoformat(str(data["updated_at"])),
            owner_client_id=str(data["owner_client_id"]) if data.get("owner_client_id") else None,
            owner_user_id=str(data["owner_user_id"]) if data.get("owner_user_id") else None,
            org_id=str(data["org_id"]) if data.get("org_id") else None,
        )


@dataclass
class StoredSession:
    """Data model for persisted sessions."""

    session_id: str
    user_id: Optional[str]
    messages: List[Dict[str, str]] = field(default_factory=list)
    metadata: Dict[str, object] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, object]:
        """Convert to dictionary representation."""
        return {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "messages": self.messages,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> StoredSession:
        """Create from dictionary representation."""
        return cls(
            session_id=str(data["session_id"]),
            user_id=data.get("user_id") if data.get("user_id") else None,
            messages=data.get("messages", []),
            metadata=data.get("metadata", {}),
            created_at=datetime.fromisoformat(str(data["created_at"])),
            updated_at=datetime.fromisoformat(str(data["updated_at"])),
        )


@dataclass
class StoredArtifact:
    """Data model for persisted artifacts."""

    artifact_id: str
    project_id: str
    filename: str
    file_path: str
    file_size_bytes: int
    format: str  # zip, folder, repo, etc.
    created_at: datetime

    # Ownership — denormalized from project for fast artifact-level checks.
    owner_client_id: Optional[str] = None
    owner_user_id: Optional[str] = None
    org_id: Optional[str] = None

    def to_dict(self) -> Dict[str, object]:
        """Convert to dictionary representation."""
        d: Dict[str, object] = {
            "artifact_id": self.artifact_id,
            "project_id": self.project_id,
            "filename": self.filename,
            "file_path": self.file_path,
            "file_size_bytes": self.file_size_bytes,
            "format": self.format,
            "created_at": self.created_at.isoformat(),
        }
        if self.owner_client_id is not None:
            d["owner_client_id"] = self.owner_client_id
        if self.owner_user_id is not None:
            d["owner_user_id"] = self.owner_user_id
        if self.org_id is not None:
            d["org_id"] = self.org_id
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> StoredArtifact:
        """Create from dictionary representation."""
        return cls(
            artifact_id=str(data["artifact_id"]),
            project_id=str(data["project_id"]),
            filename=str(data["filename"]),
            file_path=str(data["file_path"]),
            file_size_bytes=int(data["file_size_bytes"]),
            format=str(data["format"]),
            created_at=datetime.fromisoformat(str(data["created_at"])),
            owner_client_id=str(data["owner_client_id"]) if data.get("owner_client_id") else None,
            owner_user_id=str(data["owner_user_id"]) if data.get("owner_user_id") else None,
            org_id=str(data["org_id"]) if data.get("org_id") else None,
        )


@dataclass
class RequestLog:
    """Data model for request logs."""

    log_id: str
    project_id: Optional[str]
    user_id: Optional[str]
    raw_text: str
    parsed_features: List[str] = field(default_factory=list)
    parsed_tech_stack: Dict[str, object] = field(default_factory=dict)
    processing_time_ms: int = 0
    status: str = "pending"  # pending, success, failed
    created_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, object]:
        """Convert to dictionary representation."""
        return {
            "log_id": self.log_id,
            "project_id": self.project_id,
            "user_id": self.user_id,
            "raw_text": self.raw_text,
            "parsed_features": self.parsed_features,
            "parsed_tech_stack": self.parsed_tech_stack,
            "processing_time_ms": self.processing_time_ms,
            "status": self.status,
            "created_at": self.created_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> RequestLog:
        """Create from dictionary representation."""
        return cls(
            log_id=str(data["log_id"]),
            project_id=data.get("project_id") if data.get("project_id") else None,
            user_id=data.get("user_id") if data.get("user_id") else None,
            raw_text=str(data["raw_text"]),
            parsed_features=data.get("parsed_features", []),
            parsed_tech_stack=data.get("parsed_tech_stack", {}),
            processing_time_ms=int(data.get("processing_time_ms", 0)),
            status=str(data.get("status", "pending")),
            created_at=datetime.fromisoformat(str(data["created_at"])),
        )
