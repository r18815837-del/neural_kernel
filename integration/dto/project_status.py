"""DTO for the project status screen — everything Flutter needs in one shot."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class FeatureItemDTO:
    """Flat feature representation for clients."""

    name: str
    description: str = ""
    priority: str = "medium"

    def to_dict(self) -> dict[str, str]:
        return {"name": self.name, "description": self.description, "priority": self.priority}


@dataclass(frozen=True)
class TechStackDTO:
    """Flat tech-stack representation — no nesting."""

    backend: Optional[str] = None
    frontend: Optional[str] = None
    database: Optional[str] = None
    mobile: Optional[str] = None
    deployment: Optional[str] = None

    def to_dict(self) -> dict[str, object]:
        return {k: v for k, v in {
            "backend": self.backend,
            "frontend": self.frontend,
            "database": self.database,
            "mobile": self.mobile,
            "deployment": self.deployment,
        }.items() if v is not None}


@dataclass(frozen=True)
class QualityScoreDTO:
    """Aggregated quality metrics for the generated project."""

    scaffold_valid: Optional[bool] = None
    execution_ready: Optional[bool] = None
    consistency_ok: Optional[bool] = None
    overall_score: Optional[float] = None  # 0.0-1.0

    def to_dict(self) -> dict[str, object]:
        d: dict[str, object] = {}
        if self.scaffold_valid is not None:
            d["scaffold_valid"] = self.scaffold_valid
        if self.execution_ready is not None:
            d["execution_ready"] = self.execution_ready
        if self.consistency_ok is not None:
            d["consistency_ok"] = self.consistency_ok
        if self.overall_score is not None:
            d["overall_score"] = self.overall_score
        return d


@dataclass(frozen=True)
class ProjectStatusDTO:
    """Complete status DTO — one request, one screen."""

    project_id: str
    project_name: str
    status: str
    status_label: str  # human-readable, localization-friendly
    message: str

    # Timestamps — always ISO 8601 strings
    created_at: str
    updated_at: Optional[str] = None
    completed_at: Optional[str] = None

    # Progress
    progress_percent: int = 0  # 0-100

    # Artifact
    artifact_available: bool = False
    artifact_name: Optional[str] = None
    artifact_size_bytes: Optional[int] = None
    download_url: Optional[str] = None

    # Project content
    features: List[FeatureItemDTO] = field(default_factory=list)
    tech_stack: Optional[TechStackDTO] = None

    # Quality
    quality: Optional[QualityScoreDTO] = None
    execution_ready: Optional[bool] = None

    # Agent pipeline summary (flat)
    llm_used: bool = False
    agent_count: int = 0
    successful_agent_count: int = 0
    failed_agent_count: int = 0

    # Error
    error: Optional[str] = None

    def to_dict(self) -> dict[str, object]:
        d: dict[str, object] = {
            "project_id": self.project_id,
            "project_name": self.project_name,
            "status": self.status,
            "status_label": self.status_label,
            "message": self.message,
            "created_at": self.created_at,
            "progress_percent": self.progress_percent,
            "artifact_available": self.artifact_available,
            "llm_used": self.llm_used,
            "agent_count": self.agent_count,
            "successful_agent_count": self.successful_agent_count,
            "failed_agent_count": self.failed_agent_count,
        }
        if self.updated_at is not None:
            d["updated_at"] = self.updated_at
        if self.completed_at is not None:
            d["completed_at"] = self.completed_at
        if self.artifact_name is not None:
            d["artifact_name"] = self.artifact_name
        if self.artifact_size_bytes is not None:
            d["artifact_size_bytes"] = self.artifact_size_bytes
        if self.download_url is not None:
            d["download_url"] = self.download_url
        if self.features:
            d["features"] = [f.to_dict() for f in self.features]
        if self.tech_stack is not None:
            d["tech_stack"] = self.tech_stack.to_dict()
        if self.quality is not None:
            d["quality"] = self.quality.to_dict()
        if self.execution_ready is not None:
            d["execution_ready"] = self.execution_ready
        if self.error is not None:
            d["error"] = self.error
        return d
