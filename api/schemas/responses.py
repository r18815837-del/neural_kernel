"""Pydantic models for API responses."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


# ------------------------------------------------------------------
# Nested detail models
# ------------------------------------------------------------------


class AgentResultDetail(BaseModel):
    """Summary of a single agent's execution within the pipeline."""

    agent_name: str = Field(..., description="Agent class name")
    role: str = Field(..., description="Pipeline role (e.g. 'architect')")
    success: bool = Field(..., description="Whether the agent succeeded")
    message: str = Field("", description="Human-readable summary")
    llm_powered: bool = Field(False, description="True if LLM was used (not fallback)")
    outputs_keys: List[str] = Field(
        default_factory=list,
        description="Top-level keys in the agent's output dict",
    )


class FeatureDetail(BaseModel):
    """Detected / planned feature."""

    name: str = Field(..., description="Canonical feature name")
    description: str = Field("", description="Feature description")
    priority: str = Field("medium", description="Priority level")


class TechStackDetail(BaseModel):
    """Resolved technology stack."""

    backend: Optional[str] = None
    frontend: Optional[str] = None
    database: Optional[str] = None
    mobile: Optional[str] = None
    deployment: Optional[str] = None


class ScaffoldValidation(BaseModel):
    """Scaffold validation result."""

    valid: bool = Field(..., description="Overall validity")
    valid_files: List[str] = Field(default_factory=list)
    missing_files: List[str] = Field(default_factory=list)
    empty_files: List[str] = Field(default_factory=list)


class ExecutionCheckResult(BaseModel):
    """Result of a single execution-readiness check."""

    name: str = Field(..., description="Validator name")
    success: bool = Field(..., description="Whether check passed")
    message: str = Field("", description="Human-readable summary")
    details: Dict[str, Any] = Field(default_factory=dict)
    warnings: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)


class ExecutionValidation(BaseModel):
    """Aggregated execution-readiness validation report."""

    success: bool = Field(True, description="All checks passed")
    checks_run: int = Field(0)
    checks_passed: int = Field(0)
    checks_failed: int = Field(0)
    results: List[ExecutionCheckResult] = Field(default_factory=list)
    summary: Dict[str, Any] = Field(
        default_factory=dict,
        description="Command hints and aggregate details",
    )


class PipelineDetails(BaseModel):
    """Full pipeline execution summary."""

    agents: List[AgentResultDetail] = Field(default_factory=list)
    total_agents: int = Field(0)
    successful_agents: int = Field(0)
    failed_agents: int = Field(0)
    scaffold_validation: Optional[ScaffoldValidation] = None
    execution_validation: Optional[ExecutionValidation] = None


# ------------------------------------------------------------------
# Main response models
# ------------------------------------------------------------------


class GenerateProjectResponse(BaseModel):
    """Response model for project generation request.

    Returned immediately when a generation job is queued.
    """

    project_id: str = Field(..., description="Unique project identifier")
    status: Literal["pending", "in_progress", "completed", "failed"] = Field(
        "pending", description="Current generation status"
    )
    message: str = Field(..., description="Status message")
    created_at: datetime = Field(..., description="Creation timestamp")


class ProjectStatusResponse(BaseModel):
    """Response model for project status query.

    Contains full pipeline details once generation is in_progress or completed.
    """

    project_id: str = Field(..., description="Project identifier")
    status: Literal["pending", "in_progress", "completed", "failed"] = Field(
        ..., description="Generation status"
    )
    message: str = Field(..., description="Status message")
    progress: float = Field(
        0.0, ge=0.0, le=1.0, description="Progress 0-1"
    )

    # Artifact
    artifact_path: Optional[str] = Field(
        None, description="Path to artifact if completed"
    )
    artifact_size_bytes: Optional[int] = Field(
        None, description="Artifact file size in bytes"
    )

    # Project details
    project_name: Optional[str] = Field(None, description="Resolved project name")
    features_detected: List[FeatureDetail] = Field(
        default_factory=list, description="Detected features"
    )
    tech_stack: Optional[TechStackDetail] = Field(
        None, description="Technology stack info"
    )

    # Pipeline
    pipeline: Optional[PipelineDetails] = Field(
        None, description="Agent pipeline execution summary"
    )

    # Timing
    started_at: Optional[datetime] = Field(None, description="Generation start time")
    completed_at: Optional[datetime] = Field(None, description="Generation end time")
    duration_ms: Optional[int] = Field(None, description="Total processing time in ms")

    error: Optional[str] = Field(None, description="Error message if failed")


class ProjectDownloadInfo(BaseModel):
    """Response model for download metadata."""

    project_id: str = Field(..., description="Project identifier")
    filename: str = Field(..., description="Downloadable filename")
    file_size: int = Field(..., ge=0, description="File size in bytes")
    content_type: str = Field("application/zip", description="MIME type")
    features_included: List[str] = Field(
        default_factory=list, description="Features in this artifact"
    )


class HealthResponse(BaseModel):
    """Response model for health check."""

    status: str = Field(..., description="Health status")
    version: str = Field(..., description="API version")
    uptime_seconds: float = Field(..., ge=0, description="Uptime in seconds")
    llm_available: bool = Field(False, description="Whether LLM backend is connected")
    pipeline_ready: bool = Field(True, description="Whether pipeline is operational")


class InfoResponse(BaseModel):
    """Response model for /info endpoint."""

    name: str = Field("Neural Kernel API", description="Service name")
    version: str = Field("0.1.0", description="API version")
    description: str = Field(
        "AI-powered project generation API",
        description="Service description",
    )
    supported_features: List[str] = Field(
        default_factory=list,
        description="Feature names the registry can generate",
    )
    endpoints: Dict[str, str] = Field(
        default_factory=dict, description="Available endpoint paths"
    )
    pipeline_agents: List[str] = Field(
        default_factory=list,
        description="Agent roles in the execution pipeline",
    )


class ProjectListItem(BaseModel):
    """Compact project summary for list responses."""

    project_id: str
    project_name: str = ""
    status: str
    created_at: datetime
    features: List[str] = Field(default_factory=list)
    artifact_path: Optional[str] = None


class ProjectListResponse(BaseModel):
    """Response for GET /projects."""

    projects: List[ProjectListItem] = Field(default_factory=list)
    total: int = Field(0)
    limit: int = Field(50)
    offset: int = Field(0)


# ------------------------------------------------------------------
# Lifecycle / versioning models
# ------------------------------------------------------------------


class TransitionResponse(BaseModel):
    """Response after a status transition."""

    project_id: str = Field(..., description="Project identifier")
    old_status: str = Field(..., description="Previous status")
    new_status: str = Field(..., description="Current status after transition")


class AllowedTransitionsResponse(BaseModel):
    """Possible transitions for a given status."""

    project_id: str = Field(..., description="Project identifier")
    current_status: str = Field(..., description="Current status")
    allowed: List[str] = Field(default_factory=list, description="Reachable statuses")


class ArtifactVersionItem(BaseModel):
    """Single artifact version."""

    artifact_id: str
    version: int = 0
    filename: str = ""
    file_size_bytes: int = 0
    format: str = ""
    created_at: str = ""
    exists_on_disk: bool = False


class ArtifactVersionsResponse(BaseModel):
    """All artifact versions for a project."""

    project_id: str
    versions: List[ArtifactVersionItem] = Field(default_factory=list)
    total: int = 0


class RetentionResponse(BaseModel):
    """Result of retain_latest call."""

    project_id: str
    kept: int
    deleted_files: List[str] = Field(default_factory=list)


class CleanupResponse(BaseModel):
    """Result of cleanup run."""

    failed_archived: int = Field(0, description="Failed projects archived")
    artifacts_deleted: int = Field(0, description="Old artifact files deleted")
    orphan_dirs_removed: int = Field(0, description="Orphan build dirs removed")


class ConsistencyIssueResponse(BaseModel):
    """A single consistency issue."""

    check: str
    severity: str
    message: str
    file_path: str = ""
    expected: str = ""
    actual: str = ""


class ConsistencyReportResponse(BaseModel):
    """Full consistency report for a project."""

    project_id: str
    is_consistent: bool
    checks_run: int = 0
    errors: int = 0
    warnings: int = 0
    issues: List[ConsistencyIssueResponse] = Field(default_factory=list)


class ErrorResponse(BaseModel):
    """Response model for API errors."""

    error: str = Field(..., description="Error type")
    detail: str = Field(..., description="Error detail")
    status_code: int = Field(..., description="HTTP status code")
