from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from integration.artifact_service import ArtifactService
from integration.dto import (
    ArtifactMetadataDTO,
    CreateProjectResponse,
    ProjectStatusDTO,
)
from integration.response_mapper import ResponseMapper
from runtime.base.result import TaskResult


@pytest.fixture
def artifacts_dir(tmp_path):
    """Create a temporary artifacts directory."""
    return tmp_path / "artifacts"


@pytest.fixture
def artifact_service(artifacts_dir):
    """Create an ArtifactService with temporary directory."""
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    return ArtifactService(artifacts_root=str(artifacts_dir))


@pytest.fixture
def sample_task_result():
    """Create a sample TaskResult."""
    return TaskResult(
        success=True,
        message="Generation completed successfully",
        payload={
            "project_spec": {
                "tech_stack": {"backend": "FastAPI", "frontend": "React"}
            },
            "client_brief": {"requested_features": ["auth", "admin_panel"]},
        },
    )


@pytest.fixture
def sample_failed_result():
    """Create a failed TaskResult."""
    return TaskResult(
        success=False,
        message="Generation failed",
        error="Build timeout",
        payload={},
    )


class TestResponseMapperBasics:
    """Test basic ResponseMapper functionality."""

    def test_mapper_task_result_to_create_response(self, sample_task_result):
        """Test converting TaskResult to CreateProjectResponse."""
        response = ResponseMapper.task_result_to_create_response(
            project_id="proj_123", result=sample_task_result
        )

        assert isinstance(response, CreateProjectResponse)
        assert response.project_id == "proj_123"
        assert response.status == "completed"
        assert response.message == "Generation completed successfully"

    def test_mapper_failed_result_to_create_response(self, sample_failed_result):
        """Test converting failed TaskResult to response."""
        response = ResponseMapper.task_result_to_create_response(
            project_id="proj_123", result=sample_failed_result
        )

        assert response.status == "failed"
        assert response.message == "Generation failed"

    def test_mapper_task_result_to_project_status(self, sample_task_result):
        """Test converting TaskResult to ProjectStatusDTO."""
        status = ResponseMapper.task_result_to_project_status(
            project_id="proj_123",
            result=sample_task_result,
            progress=1.0,
            artifact_path="/artifacts/proj_123.zip",
        )

        assert isinstance(status, ProjectStatusDTO)
        assert status.project_id == "proj_123"
        assert status.status == "completed"
        assert status.progress == 1.0
        assert status.artifact_path == "/artifacts/proj_123.zip"

    def test_mapper_extracts_tech_stack(self, sample_task_result):
        """Test that mapper extracts tech stack from result."""
        status = ResponseMapper.task_result_to_project_status(
            project_id="proj_123", result=sample_task_result
        )

        assert status.tech_stack == {"backend": "FastAPI", "frontend": "React"}

    def test_mapper_extracts_features(self, sample_task_result):
        """Test that mapper extracts features from result."""
        status = ResponseMapper.task_result_to_project_status(
            project_id="proj_123", result=sample_task_result
        )

        assert status.features == ["auth", "admin_panel"]

    def test_mapper_handles_missing_tech_stack(self):
        """Test mapper handles missing tech_stack in result."""
        result = TaskResult(
            success=True,
            message="Done",
            payload={"client_brief": {"requested_features": ["auth"]}},
        )
        status = ResponseMapper.task_result_to_project_status(
            project_id="proj_123", result=result
        )

        assert status.tech_stack is None
        assert status.features == ["auth"]

    def test_mapper_handles_missing_features(self):
        """Test mapper handles missing features in result."""
        result = TaskResult(
            success=True,
            message="Done",
            payload={"project_spec": {"tech_stack": {"backend": "FastAPI"}}},
        )
        status = ResponseMapper.task_result_to_project_status(
            project_id="proj_123", result=result
        )

        assert status.tech_stack == {"backend": "FastAPI"}
        assert status.features == []

    def test_mapper_creates_artifact_metadata(self):
        """Test creating artifact metadata."""
        metadata = ResponseMapper.create_artifact_metadata(
            project_id="proj_123",
            filename="project.zip",
            file_size_bytes=1024000,
            artifact_format="zip",
            tech_stack={"backend": "FastAPI"},
            features=["auth", "admin_panel"],
        )

        assert isinstance(metadata, ArtifactMetadataDTO)
        assert metadata.project_id == "proj_123"
        assert metadata.filename == "project.zip"
        assert metadata.file_size_bytes == 1024000
        assert metadata.features == ["auth", "admin_panel"]

    def test_mapper_converts_tech_stack_to_strings(self):
        """Test that tech_stack values are converted to strings."""
        metadata = ResponseMapper.create_artifact_metadata(
            project_id="proj_123",
            filename="project.zip",
            file_size_bytes=1000,
            tech_stack={"backend": "FastAPI", "memory": 8192, "cpu": "multi"},
            features=[],
        )

        assert isinstance(metadata.tech_stack_summary["backend"], str)
        assert isinstance(metadata.tech_stack_summary["memory"], str)


class TestCreateProjectResponse:
    """Test CreateProjectResponse DTO."""

    def test_create_response_with_all_fields(self):
        """Test creating response with all fields."""
        now = datetime.utcnow()
        response = CreateProjectResponse(
            project_id="proj_123",
            status="pending",
            message="Request queued",
            created_at=now,
        )

        assert response.project_id == "proj_123"
        assert response.status == "pending"
        assert response.message == "Request queued"
        assert response.created_at == now

    def test_create_response_status_values(self):
        """Test valid status values."""
        statuses = ["pending", "in_progress", "completed", "failed"]
        now = datetime.utcnow()

        for status in statuses:
            response = CreateProjectResponse(
                project_id="proj_123",
                status=status,
                message="Test",
                created_at=now,
            )
            assert response.status == status


class TestProjectStatusDTO:
    """Test ProjectStatusDTO DTO."""

    def test_status_dto_with_defaults(self):
        """Test creating ProjectStatusDTO with defaults."""
        status = ProjectStatusDTO(
            project_id="proj_123",
            status="pending",
            message="Processing",
        )

        assert status.project_id == "proj_123"
        assert status.progress == 0.0
        assert status.features == []
        assert status.tech_stack is None

    def test_status_dto_with_all_fields(self):
        """Test creating ProjectStatusDTO with all fields."""
        status = ProjectStatusDTO(
            project_id="proj_123",
            status="completed",
            message="Done",
            progress=1.0,
            features=["auth", "admin"],
            tech_stack={"backend": "FastAPI"},
            artifact_path="/artifacts/proj_123.zip",
            error=None,
        )

        assert status.progress == 1.0
        assert len(status.features) == 2
        assert status.tech_stack["backend"] == "FastAPI"

    def test_status_dto_with_error(self):
        """Test ProjectStatusDTO with error."""
        status = ProjectStatusDTO(
            project_id="proj_123",
            status="failed",
            message="Failed",
            error="Timeout occurred",
        )

        assert status.status == "failed"
        assert status.error == "Timeout occurred"

    def test_status_dto_progress_tracking(self):
        """Test tracking progress through multiple states."""
        # Start
        status1 = ProjectStatusDTO(
            project_id="proj_123",
            status="pending",
            message="Queued",
            progress=0.0,
        )

        # In progress
        status2 = ProjectStatusDTO(
            project_id="proj_123",
            status="in_progress",
            message="Processing",
            progress=0.5,
        )

        # Complete
        status3 = ProjectStatusDTO(
            project_id="proj_123",
            status="completed",
            message="Done",
            progress=1.0,
            artifact_path="/artifacts/proj_123.zip",
        )

        assert status1.progress == 0.0
        assert status2.progress == 0.5
        assert status3.progress == 1.0
        assert status3.artifact_path is not None


class TestArtifactMetadataDTO:
    """Test ArtifactMetadataDTO DTO."""

    def test_artifact_metadata_creation(self):
        """Test creating artifact metadata."""
        now = datetime.utcnow()
        metadata = ArtifactMetadataDTO(
            project_id="proj_123",
            filename="project.zip",
            file_size_bytes=1024000,
            format="zip",
            created_at=now,
        )

        assert metadata.project_id == "proj_123"
        assert metadata.filename == "project.zip"
        assert metadata.format == "zip"

    def test_artifact_metadata_with_features(self):
        """Test artifact metadata with features."""
        metadata = ArtifactMetadataDTO(
            project_id="proj_123",
            filename="project.zip",
            file_size_bytes=1024000,
            format="zip",
            created_at=datetime.utcnow(),
            features=["auth", "admin_panel", "export"],
        )

        assert len(metadata.features) == 3

    def test_artifact_metadata_with_tech_stack(self):
        """Test artifact metadata with tech stack summary."""
        tech_stack = {
            "backend": "FastAPI",
            "frontend": "React",
            "database": "PostgreSQL",
        }
        metadata = ArtifactMetadataDTO(
            project_id="proj_123",
            filename="project.zip",
            file_size_bytes=1024000,
            format="zip",
            created_at=datetime.utcnow(),
            tech_stack_summary=tech_stack,
        )

        assert metadata.tech_stack_summary["backend"] == "FastAPI"


class TestArtifactService:
    """Test ArtifactService functionality."""

    def test_artifact_service_initialization(self, artifact_service):
        """Test that ArtifactService initializes properly."""
        assert artifact_service is not None
        assert artifact_service.artifacts_root is not None

    def test_get_artifact_path(self, artifact_service):
        """Test getting artifact path."""
        path = artifact_service.get_artifact_path("proj_123")
        assert str(path).endswith("proj_123.zip")

    def test_artifact_exists_returns_false_for_nonexistent(self, artifact_service):
        """Test that artifact_exists returns False for nonexistent artifacts."""
        exists = artifact_service.artifact_exists("nonexistent_proj")
        assert exists is False

    def test_artifact_exists_returns_true_for_existing(self, artifact_service):
        """Test that artifact_exists returns True for existing artifacts."""
        # Create a test artifact file
        artifact_path = artifact_service.get_artifact_path("proj_123")
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        artifact_path.write_text("test content")

        exists = artifact_service.artifact_exists("proj_123")
        assert exists is True

    def test_get_artifact_metadata_raises_for_nonexistent(self, artifact_service):
        """Test that get_artifact_metadata raises for nonexistent artifact."""
        with pytest.raises(FileNotFoundError):
            artifact_service.get_artifact_metadata("nonexistent_proj")

    def test_get_artifact_metadata_for_existing(self, artifact_service):
        """Test getting metadata for existing artifact."""
        # Create a test artifact
        artifact_path = artifact_service.get_artifact_path("proj_123")
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        artifact_path.write_text("test content" * 100)  # Make it larger

        metadata = artifact_service.get_artifact_metadata("proj_123")

        assert metadata.project_id == "proj_123"
        assert metadata.filename == "proj_123.zip"
        assert metadata.file_size_bytes > 0

    def test_get_artifact_metadata_with_tech_stack(self, artifact_service):
        """Test getting metadata with tech stack."""
        # Create a test artifact
        artifact_path = artifact_service.get_artifact_path("proj_123")
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        artifact_path.write_text("test")

        metadata = artifact_service.get_artifact_metadata(
            "proj_123",
            tech_stack={"backend": "FastAPI"},
            features=["auth"],
        )

        assert metadata.tech_stack_summary == {"backend": "FastAPI"}
        assert metadata.features == ["auth"]

    def test_delete_artifact(self, artifact_service):
        """Test deleting an artifact."""
        # Create a test artifact
        artifact_path = artifact_service.get_artifact_path("proj_123")
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        artifact_path.write_text("test")

        # Verify it exists
        assert artifact_service.artifact_exists("proj_123")

        # Delete it
        result = artifact_service.delete_artifact("proj_123")
        assert result is True
        assert not artifact_service.artifact_exists("proj_123")

    def test_delete_nonexistent_artifact_returns_false(self, artifact_service):
        """Test that deleting nonexistent artifact returns False."""
        result = artifact_service.delete_artifact("nonexistent")
        assert result is False

    def test_list_artifacts(self, artifact_service):
        """Test listing artifacts."""
        # Create test artifacts
        for i in range(3):
            artifact_path = artifact_service.get_artifact_path(f"proj_{i}")
            artifact_path.parent.mkdir(parents=True, exist_ok=True)
            artifact_path.write_text("test")

        artifacts = artifact_service.list_artifacts()
        assert len(artifacts) >= 3

    def test_get_artifact_size_mb(self, artifact_service):
        """Test getting artifact size in MB."""
        # Create a test artifact of known size
        artifact_path = artifact_service.get_artifact_path("proj_123")
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        # Create 1 MB file
        artifact_path.write_bytes(b"x" * (1024 * 1024))

        size_mb = artifact_service.get_artifact_size_mb("proj_123")
        assert 0.9 < size_mb < 1.1  # Allow for floating point imprecision


class TestResponseMapperIntegration:
    """Test ResponseMapper integration scenarios."""

    def test_complete_generation_flow(self, sample_task_result):
        """Test complete generation response flow."""
        # Initial response
        create_response = ResponseMapper.task_result_to_create_response(
            "proj_123", sample_task_result
        )

        # Status response
        status = ResponseMapper.task_result_to_project_status(
            "proj_123", sample_task_result, progress=1.0,
            artifact_path="/artifacts/proj_123.zip"
        )

        # Artifact metadata
        metadata = ResponseMapper.create_artifact_metadata(
            "proj_123",
            "proj_123.zip",
            1024000,
            features=status.features,
            tech_stack=status.tech_stack,
        )

        assert create_response.status == "completed"
        assert status.progress == 1.0
        assert metadata.project_id == "proj_123"

    def test_failed_generation_flow(self, sample_failed_result):
        """Test failed generation response flow."""
        response = ResponseMapper.task_result_to_create_response(
            "proj_123", sample_failed_result
        )

        status = ResponseMapper.task_result_to_project_status(
            "proj_123", sample_failed_result
        )

        assert response.status == "failed"
        assert status.status == "failed"
        assert status.error == "Build timeout"
