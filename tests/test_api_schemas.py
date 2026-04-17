from __future__ import annotations

from datetime import datetime

import pytest
from pydantic import ValidationError

from api.schemas.requests import GenerateProjectRequest
from api.schemas.responses import (
    ErrorResponse,
    GenerateProjectResponse,
    HealthResponse,
    ProjectDownloadInfo,
    ProjectStatusResponse,
)


class TestGenerateProjectRequest:
    """Test GenerateProjectRequest schema validation."""

    def test_minimal_request_creation(self):
        """Test creating request with minimal required fields."""
        request = GenerateProjectRequest(
            text="Build a CRM application"
        )
        assert request.text == "Build a CRM application"
        assert request.user_id is None
        assert request.session_id is None
        assert request.output_format == "zip"
        assert request.metadata is None

    def test_full_request_creation(self):
        """Test creating request with all fields."""
        request = GenerateProjectRequest(
            text="Build a CRM application",
            user_id="user_123",
            session_id="session_456",
            output_format="folder",
            metadata={"client": "acme_corp"},
        )
        assert request.text == "Build a CRM application"
        assert request.user_id == "user_123"
        assert request.session_id == "session_456"
        assert request.output_format == "folder"
        assert request.metadata == {"client": "acme_corp"}

    def test_request_missing_text_fails(self):
        """Test that request without text fails validation."""
        with pytest.raises(ValidationError):
            GenerateProjectRequest()

    def test_request_empty_text_fails(self):
        """Test that request with empty text fails validation."""
        with pytest.raises(ValidationError):
            GenerateProjectRequest(text="")

    def test_default_output_format_is_zip(self):
        """Test that default output format is zip."""
        request = GenerateProjectRequest(text="Build something")
        assert request.output_format == "zip"

    def test_valid_output_formats(self):
        """Test that valid output formats are accepted."""
        for fmt in ["zip", "folder"]:
            request = GenerateProjectRequest(
                text="Build something",
                output_format=fmt,
            )
            assert request.output_format == fmt

    def test_invalid_output_format_fails(self):
        """Test that invalid output format fails validation."""
        with pytest.raises(ValidationError):
            GenerateProjectRequest(
                text="Build something",
                output_format="invalid",
            )

    def test_metadata_is_optional(self):
        """Test that metadata is optional."""
        request = GenerateProjectRequest(text="Build something")
        assert request.metadata is None

    def test_request_serialization(self):
        """Test that request can be serialized to JSON."""
        request = GenerateProjectRequest(
            text="Build a CRM",
            user_id="user_123",
            output_format="zip",
        )
        data = request.model_dump()
        assert data["text"] == "Build a CRM"
        assert data["user_id"] == "user_123"

    def test_request_from_dict(self):
        """Test creating request from dictionary."""
        data = {
            "text": "Build something",
            "user_id": "user_123",
            "output_format": "folder",
        }
        request = GenerateProjectRequest(**data)
        assert request.text == "Build something"


class TestGenerateProjectResponse:
    """Test GenerateProjectResponse schema."""

    def test_response_creation(self):
        """Test creating a response."""
        now = datetime.utcnow()
        response = GenerateProjectResponse(
            project_id="proj_123",
            status="pending",
            message="Project generation queued",
            created_at=now,
        )
        assert response.project_id == "proj_123"
        assert response.status == "pending"
        assert response.message == "Project generation queued"
        assert response.created_at == now

    def test_response_with_all_statuses(self):
        """Test that all status values are valid."""
        statuses = ["pending", "in_progress", "completed", "failed"]
        for status in statuses:
            response = GenerateProjectResponse(
                project_id="proj_123",
                status=status,
                message="Test",
                created_at=datetime.utcnow(),
            )
            assert response.status == status

    def test_response_invalid_status_fails(self):
        """Test that invalid status fails validation."""
        with pytest.raises(ValidationError):
            GenerateProjectResponse(
                project_id="proj_123",
                status="unknown",
                message="Test",
                created_at=datetime.utcnow(),
            )

    def test_response_missing_fields_fails(self):
        """Test that missing required fields fail."""
        with pytest.raises(ValidationError):
            GenerateProjectResponse(project_id="proj_123")

    def test_response_serialization(self):
        """Test that response can be serialized."""
        now = datetime.utcnow()
        response = GenerateProjectResponse(
            project_id="proj_123",
            status="completed",
            message="Done",
            created_at=now,
        )
        data = response.model_dump()
        assert data["project_id"] == "proj_123"
        assert data["status"] == "completed"


class TestProjectStatusResponse:
    """Test ProjectStatusResponse schema."""

    def test_status_response_creation(self):
        """Test creating a status response."""
        response = ProjectStatusResponse(
            project_id="proj_123",
            status="completed",
            message="Generation complete",
            progress=1.0,
            artifact_path="/artifacts/proj_123.zip",
            features_detected=["auth", "admin_panel"],
            tech_stack={"backend": "FastAPI", "frontend": "React"},
        )
        assert response.project_id == "proj_123"
        assert response.status == "completed"
        assert response.progress == 1.0
        assert len(response.features_detected) == 2

    def test_status_response_minimal(self):
        """Test creating status response with minimal fields."""
        response = ProjectStatusResponse(
            project_id="proj_123",
            status="pending",
            message="Processing",
        )
        assert response.project_id == "proj_123"
        assert response.progress == 0.0
        assert response.features_detected == []

    def test_progress_must_be_between_0_and_1(self):
        """Test that progress must be between 0 and 1."""
        # Valid progress values
        for progress in [0.0, 0.5, 1.0]:
            response = ProjectStatusResponse(
                project_id="proj_123",
                status="in_progress",
                message="Processing",
                progress=progress,
            )
            assert response.progress == progress

        # Invalid progress values
        with pytest.raises(ValidationError):
            ProjectStatusResponse(
                project_id="proj_123",
                status="in_progress",
                message="Processing",
                progress=1.5,
            )

        with pytest.raises(ValidationError):
            ProjectStatusResponse(
                project_id="proj_123",
                status="in_progress",
                message="Processing",
                progress=-0.1,
            )

    def test_features_detected_default_empty_list(self):
        """Test that features_detected defaults to empty list."""
        response = ProjectStatusResponse(
            project_id="proj_123",
            status="pending",
            message="Processing",
        )
        assert response.features_detected == []

    def test_error_message_optional(self):
        """Test that error message is optional."""
        response = ProjectStatusResponse(
            project_id="proj_123",
            status="completed",
            message="Success",
        )
        assert response.error is None

    def test_error_message_when_failed(self):
        """Test setting error message for failed status."""
        response = ProjectStatusResponse(
            project_id="proj_123",
            status="failed",
            message="Generation failed",
            error="Build timeout exceeded",
        )
        assert response.error == "Build timeout exceeded"


class TestProjectDownloadInfo:
    """Test ProjectDownloadInfo schema."""

    def test_download_info_creation(self):
        """Test creating download info."""
        info = ProjectDownloadInfo(
            project_id="proj_123",
            filename="my_project.zip",
            file_size=1024000,
            download_url="https://api.example.com/download/proj_123",
        )
        assert info.project_id == "proj_123"
        assert info.filename == "my_project.zip"
        assert info.file_size == 1024000

    def test_file_size_must_be_non_negative(self):
        """Test that file_size must be non-negative."""
        # Valid
        info = ProjectDownloadInfo(
            project_id="proj_123",
            filename="empty.zip",
            file_size=0,
            download_url="http://example.com",
        )
        assert info.file_size == 0

        # Invalid
        with pytest.raises(ValidationError):
            ProjectDownloadInfo(
                project_id="proj_123",
                filename="test.zip",
                file_size=-100,
                download_url="http://example.com",
            )

    def test_missing_fields_fails(self):
        """Test that missing fields fail validation."""
        with pytest.raises(ValidationError):
            ProjectDownloadInfo(
                project_id="proj_123",
                filename="test.zip",
            )


class TestHealthResponse:
    """Test HealthResponse schema."""

    def test_health_response_creation(self):
        """Test creating health response."""
        response = HealthResponse(
            status="healthy",
            version="1.0.0",
            uptime_seconds=3600.0,
        )
        assert response.status == "healthy"
        assert response.version == "1.0.0"
        assert response.uptime_seconds == 3600.0

    def test_health_uptime_must_be_non_negative(self):
        """Test that uptime must be non-negative."""
        # Valid
        response = HealthResponse(
            status="healthy",
            version="1.0.0",
            uptime_seconds=0.0,
        )
        assert response.uptime_seconds == 0.0

        # Invalid
        with pytest.raises(ValidationError):
            HealthResponse(
                status="healthy",
                version="1.0.0",
                uptime_seconds=-1.0,
            )

    def test_health_response_serialization(self):
        """Test serializing health response."""
        response = HealthResponse(
            status="healthy",
            version="1.0.0",
            uptime_seconds=3600.0,
        )
        data = response.model_dump()
        assert data["status"] == "healthy"
        assert data["uptime_seconds"] == 3600.0


class TestErrorResponse:
    """Test ErrorResponse schema."""

    def test_error_response_creation(self):
        """Test creating error response."""
        response = ErrorResponse(
            error="ValidationError",
            detail="Invalid project description",
            status_code=400,
        )
        assert response.error == "ValidationError"
        assert response.detail == "Invalid project description"
        assert response.status_code == 400

    def test_error_response_with_various_status_codes(self):
        """Test error responses with different status codes."""
        codes = [400, 401, 403, 404, 500, 503]
        for code in codes:
            response = ErrorResponse(
                error="TestError",
                detail="Test detail",
                status_code=code,
            )
            assert response.status_code == code

    def test_error_response_serialization(self):
        """Test serializing error response."""
        response = ErrorResponse(
            error="NotFoundError",
            detail="Project not found",
            status_code=404,
        )
        data = response.model_dump()
        assert data["error"] == "NotFoundError"
        assert data["status_code"] == 404


class TestSchemaIntegration:
    """Test schema usage patterns."""

    def test_request_response_flow(self):
        """Test a typical request-response flow."""
        # Client sends request
        request = GenerateProjectRequest(
            text="Build a blog platform",
            user_id="user_123",
            output_format="zip",
        )

        # Server responds with initial status
        response = GenerateProjectResponse(
            project_id="proj_456",
            status="pending",
            message="Request queued",
            created_at=datetime.utcnow(),
        )

        assert request.text == "Build a blog platform"
        assert response.project_id == "proj_456"

    def test_status_update_flow(self):
        """Test status update flow."""
        # Initial status
        status = ProjectStatusResponse(
            project_id="proj_123",
            status="in_progress",
            progress=0.5,
            message="Generating files",
        )

        # Completion status
        completed = ProjectStatusResponse(
            project_id="proj_123",
            status="completed",
            progress=1.0,
            message="Generation complete",
            artifact_path="/artifacts/proj_123.zip",
        )

        assert status.progress == 0.5
        assert completed.progress == 1.0

    def test_error_flow(self):
        """Test error response flow."""
        status = ProjectStatusResponse(
            project_id="proj_123",
            status="failed",
            message="Generation failed",
            error="Timeout during processing",
        )

        error_response = ErrorResponse(
            error="GenerationTimeout",
            detail="Project generation exceeded maximum processing time",
            status_code=504,
        )

        assert status.status == "failed"
        assert error_response.status_code == 504
