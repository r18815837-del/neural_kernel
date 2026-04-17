from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from persistence.base import BaseStore
from persistence.models import RequestLog
from persistence.models import StoredArtifact
from persistence.models import StoredProject
from persistence.models import StoredSession
from persistence.sqlite_store import SQLiteStore


@pytest.fixture
def temp_db(tmp_path):
    """Create a temporary SQLite database file."""
    return tmp_path / "test.db"


@pytest.fixture
def sqlite_store(temp_db):
    """Create a SQLiteStore with a temporary database."""
    return SQLiteStore(db_path=str(temp_db))


@pytest.fixture
def sample_project():
    """Create a sample StoredProject."""
    return StoredProject(
        project_id="proj_123",
        user_id="user_456",
        session_id="sess_789",
        raw_text="Build me a CRM app",
        project_name="my_crm",
        summary="A customer relationship management system",
        project_type="application",
        features_json='["auth", "admin_panel"]',
        tech_stack_json='{"backend": "FastAPI", "frontend": "React"}',
        status="completed",
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
    )


@pytest.fixture
def sample_session():
    """Create a sample StoredSession."""
    return StoredSession(
        session_id="sess_123",
        user_id="user_456",
        messages=[
            {"role": "user", "content": "Build a CRM"},
            {"role": "assistant", "content": "I'll help you build a CRM"},
        ],
        metadata={"conversation_type": "project_creation"},
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
    )


@pytest.fixture
def sample_artifact():
    """Create a sample StoredArtifact."""
    return StoredArtifact(
        artifact_id="art_123",
        project_id="proj_123",
        filename="my_crm.zip",
        file_path="/artifacts/my_crm.zip",
        file_size_bytes=1024000,
        format="zip",
        created_at=datetime.utcnow(),
    )


@pytest.fixture
def sample_request_log():
    """Create a sample RequestLog."""
    return RequestLog(
        log_id="log_123",
        project_id="proj_123",
        user_id="user_456",
        raw_text="Build a CRM app",
        parsed_features=["auth", "admin_panel"],
        parsed_tech_stack={"backend": "FastAPI"},
        processing_time_ms=1500,
        status="success",
        created_at=datetime.utcnow(),
    )


class TestSQLiteStoreBasics:
    """Test basic SQLiteStore functionality."""

    def test_sqlite_store_initialization(self, temp_db):
        """Test that SQLiteStore initializes properly."""
        store = SQLiteStore(db_path=str(temp_db))
        assert store is not None
        assert store.db_path == Path(temp_db)

    def test_sqlite_store_creates_database(self, temp_db):
        """Test that SQLiteStore creates the database file."""
        store = SQLiteStore(db_path=str(temp_db))
        assert temp_db.exists()

    def test_sqlite_store_is_base_store(self, sqlite_store):
        """Test that SQLiteStore implements BaseStore."""
        assert isinstance(sqlite_store, BaseStore)


class TestSQLiteStoreProjectOperations:
    """Test project storage operations."""

    def test_save_project(self, sqlite_store, sample_project):
        """Test saving a project."""
        project_id = sqlite_store.save_project(sample_project)
        assert project_id == sample_project.project_id

    def test_get_project_after_save(self, sqlite_store, sample_project):
        """Test retrieving a saved project."""
        sqlite_store.save_project(sample_project)
        retrieved = sqlite_store.get_project(sample_project.project_id)

        assert retrieved is not None
        assert retrieved.project_id == sample_project.project_id
        assert retrieved.project_name == sample_project.project_name

    def test_get_nonexistent_project_returns_none(self, sqlite_store):
        """Test that getting nonexistent project returns None."""
        result = sqlite_store.get_project("nonexistent_id")
        assert result is None

    def test_update_project_status(self, sqlite_store, sample_project):
        """Test updating project status."""
        sqlite_store.save_project(sample_project)
        sqlite_store.update_project_status(
            sample_project.project_id, "failed", "Build failed"
        )

        retrieved = sqlite_store.get_project(sample_project.project_id)
        assert retrieved.status == "failed"
        assert retrieved.error_message == "Build failed"

    def test_list_projects_all(self, sqlite_store, sample_project):
        """Test listing all projects."""
        sqlite_store.save_project(sample_project)

        projects = sqlite_store.list_projects()
        assert len(projects) >= 1
        assert any(p.project_id == sample_project.project_id for p in projects)

    def test_list_projects_with_user_filter(self, sqlite_store, sample_project):
        """Test listing projects filtered by user."""
        sqlite_store.save_project(sample_project)

        projects = sqlite_store.list_projects(user_id="user_456")
        assert len(projects) >= 1
        assert all(p.user_id == "user_456" for p in projects)

    def test_list_projects_empty_when_no_matches(self, sqlite_store, sample_project):
        """Test that list returns empty when no matches."""
        sqlite_store.save_project(sample_project)

        projects = sqlite_store.list_projects(user_id="nonexistent_user")
        assert len(projects) == 0

    def test_list_projects_respects_pagination(self, sqlite_store):
        """Test that pagination works correctly."""
        # Create multiple projects
        for i in range(5):
            project = StoredProject(
                project_id=f"proj_{i}",
                user_id="user_123",
                session_id=f"sess_{i}",
                raw_text="Test",
                project_name=f"project_{i}",
                summary="Test project",
                project_type="application",
                features_json="[]",
                tech_stack_json="{}",
                status="completed",
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
            )
            sqlite_store.save_project(project)

        # Test limit and offset
        projects = sqlite_store.list_projects(limit=2, offset=0)
        assert len(projects) == 2

        projects = sqlite_store.list_projects(limit=2, offset=2)
        assert len(projects) == 2

    def test_project_round_trip_preserves_data(self, sqlite_store, sample_project):
        """Test that saving and loading preserves all data."""
        sqlite_store.save_project(sample_project)
        retrieved = sqlite_store.get_project(sample_project.project_id)

        assert retrieved.project_id == sample_project.project_id
        assert retrieved.user_id == sample_project.user_id
        assert retrieved.project_name == sample_project.project_name
        assert retrieved.features_json == sample_project.features_json
        assert retrieved.tech_stack_json == sample_project.tech_stack_json


class TestSQLiteStoreSessionOperations:
    """Test session storage operations."""

    def test_save_session(self, sqlite_store, sample_session):
        """Test saving a session."""
        session_id = sqlite_store.save_session(sample_session)
        assert session_id == sample_session.session_id

    def test_get_session_after_save(self, sqlite_store, sample_session):
        """Test retrieving a saved session."""
        sqlite_store.save_session(sample_session)
        retrieved = sqlite_store.get_session(sample_session.session_id)

        assert retrieved is not None
        assert retrieved.session_id == sample_session.session_id
        assert len(retrieved.messages) == len(sample_session.messages)

    def test_get_nonexistent_session_returns_none(self, sqlite_store):
        """Test that getting nonexistent session returns None."""
        result = sqlite_store.get_session("nonexistent_id")
        assert result is None

    def test_append_message_to_session(self, sqlite_store, sample_session):
        """Test appending a message to a session."""
        sqlite_store.save_session(sample_session)
        initial_count = len(sample_session.messages)

        sqlite_store.append_message(
            sample_session.session_id, "user", "New message"
        )

        retrieved = sqlite_store.get_session(sample_session.session_id)
        assert len(retrieved.messages) == initial_count + 1
        assert retrieved.messages[-1]["content"] == "New message"

    def test_append_message_to_nonexistent_session(self, sqlite_store):
        """Test appending to nonexistent session returns None."""
        result = sqlite_store.append_message("new_session", "user", "First message")

        # append_message returns None when session doesn't exist
        assert result is None

    def test_session_round_trip_preserves_data(self, sqlite_store, sample_session):
        """Test that saving and loading preserves all session data."""
        sqlite_store.save_session(sample_session)
        retrieved = sqlite_store.get_session(sample_session.session_id)

        assert retrieved.session_id == sample_session.session_id
        assert retrieved.user_id == sample_session.user_id
        assert len(retrieved.messages) == len(sample_session.messages)
        assert retrieved.metadata == sample_session.metadata


class TestSQLiteStoreArtifactOperations:
    """Test artifact storage operations."""

    def test_save_artifact(self, sqlite_store, sample_artifact):
        """Test saving an artifact."""
        artifact_id = sqlite_store.save_artifact(sample_artifact)
        assert artifact_id == sample_artifact.artifact_id

    def test_get_artifact_after_save(self, sqlite_store, sample_artifact):
        """Test retrieving a saved artifact."""
        sqlite_store.save_artifact(sample_artifact)
        retrieved = sqlite_store.get_artifact(sample_artifact.artifact_id)

        assert retrieved is not None
        assert retrieved.artifact_id == sample_artifact.artifact_id
        assert retrieved.filename == sample_artifact.filename

    def test_get_nonexistent_artifact_returns_none(self, sqlite_store):
        """Test that getting nonexistent artifact returns None."""
        result = sqlite_store.get_artifact("nonexistent_id")
        assert result is None

    def test_get_artifacts_by_project(self, sqlite_store, sample_artifact):
        """Test retrieving artifacts by project."""
        sqlite_store.save_artifact(sample_artifact)

        artifacts = sqlite_store.get_artifacts_by_project(sample_artifact.project_id)
        assert len(artifacts) >= 1
        assert any(a.artifact_id == sample_artifact.artifact_id for a in artifacts)

    def test_get_artifacts_empty_when_no_project(self, sqlite_store):
        """Test that artifacts returns empty for nonexistent project."""
        artifacts = sqlite_store.get_artifacts_by_project("nonexistent_project")
        assert len(artifacts) == 0

    def test_multiple_artifacts_per_project(self, sqlite_store):
        """Test storing multiple artifacts for same project."""
        for i in range(3):
            artifact = StoredArtifact(
                artifact_id=f"art_{i}",
                project_id="proj_123",
                filename=f"artifact_{i}.zip",
                file_path=f"/artifacts/artifact_{i}.zip",
                file_size_bytes=1000 * (i + 1),
                format="zip",
                created_at=datetime.utcnow(),
            )
            sqlite_store.save_artifact(artifact)

        artifacts = sqlite_store.get_artifacts_by_project("proj_123")
        assert len(artifacts) == 3

    def test_artifact_round_trip_preserves_data(
        self, sqlite_store, sample_artifact
    ):
        """Test that saving and loading preserves artifact data."""
        sqlite_store.save_artifact(sample_artifact)
        retrieved = sqlite_store.get_artifact(sample_artifact.artifact_id)

        assert retrieved.artifact_id == sample_artifact.artifact_id
        assert retrieved.project_id == sample_artifact.project_id
        assert retrieved.file_size_bytes == sample_artifact.file_size_bytes


class TestSQLiteStoreRequestLogOperations:
    """Test request log storage operations."""

    def test_log_request(self, sqlite_store, sample_request_log):
        """Test logging a request."""
        log_id = sqlite_store.log_request(sample_request_log)
        assert log_id == sample_request_log.log_id

    def test_get_request_logs_all(self, sqlite_store, sample_request_log):
        """Test retrieving all request logs."""
        sqlite_store.log_request(sample_request_log)

        logs = sqlite_store.get_request_logs()
        assert len(logs) >= 1
        assert any(log.log_id == sample_request_log.log_id for log in logs)

    def test_get_request_logs_by_project(self, sqlite_store, sample_request_log):
        """Test retrieving logs filtered by project."""
        sqlite_store.log_request(sample_request_log)

        logs = sqlite_store.get_request_logs(project_id=sample_request_log.project_id)
        assert len(logs) >= 1
        assert all(log.project_id == sample_request_log.project_id for log in logs)

    def test_get_request_logs_empty_when_no_matches(self, sqlite_store, sample_request_log):
        """Test that logs returns empty when no matches."""
        sqlite_store.log_request(sample_request_log)

        logs = sqlite_store.get_request_logs(project_id="nonexistent_project")
        assert len(logs) == 0

    def test_request_log_round_trip(self, sqlite_store, sample_request_log):
        """Test that logging and retrieving preserves data."""
        sqlite_store.log_request(sample_request_log)
        logs = sqlite_store.get_request_logs(project_id=sample_request_log.project_id)

        assert len(logs) >= 1
        log = next(log for log in logs if log.log_id == sample_request_log.log_id)
        assert log.user_id == sample_request_log.user_id
        assert log.status == sample_request_log.status
        assert log.processing_time_ms == sample_request_log.processing_time_ms


class TestStoredProjectSerialization:
    """Test StoredProject serialization."""

    def test_stored_project_to_dict(self, sample_project):
        """Test converting StoredProject to dict."""
        data = sample_project.to_dict()

        assert isinstance(data, dict)
        assert data["project_id"] == sample_project.project_id
        assert data["project_name"] == sample_project.project_name

    def test_stored_project_from_dict(self, sample_project):
        """Test creating StoredProject from dict."""
        data = sample_project.to_dict()
        restored = StoredProject.from_dict(data)

        assert restored.project_id == sample_project.project_id
        assert restored.project_name == sample_project.project_name
        assert restored.status == sample_project.status

    def test_stored_project_serialization_round_trip(self, sample_project):
        """Test full serialization round trip."""
        data = sample_project.to_dict()
        restored = StoredProject.from_dict(data)
        data2 = restored.to_dict()

        assert data == data2


class TestStoredSessionSerialization:
    """Test StoredSession serialization."""

    def test_stored_session_to_dict(self, sample_session):
        """Test converting StoredSession to dict."""
        data = sample_session.to_dict()

        assert isinstance(data, dict)
        assert data["session_id"] == sample_session.session_id
        assert len(data["messages"]) == len(sample_session.messages)

    def test_stored_session_from_dict(self, sample_session):
        """Test creating StoredSession from dict."""
        data = sample_session.to_dict()
        restored = StoredSession.from_dict(data)

        assert restored.session_id == sample_session.session_id
        assert len(restored.messages) == len(sample_session.messages)

    def test_stored_session_serialization_round_trip(self, sample_session):
        """Test full serialization round trip."""
        data = sample_session.to_dict()
        restored = StoredSession.from_dict(data)
        data2 = restored.to_dict()

        assert data == data2


class TestStoredArtifactSerialization:
    """Test StoredArtifact serialization."""

    def test_stored_artifact_to_dict(self, sample_artifact):
        """Test converting StoredArtifact to dict."""
        data = sample_artifact.to_dict()

        assert isinstance(data, dict)
        assert data["artifact_id"] == sample_artifact.artifact_id
        assert data["file_size_bytes"] == sample_artifact.file_size_bytes

    def test_stored_artifact_from_dict(self, sample_artifact):
        """Test creating StoredArtifact from dict."""
        data = sample_artifact.to_dict()
        restored = StoredArtifact.from_dict(data)

        assert restored.artifact_id == sample_artifact.artifact_id
        assert restored.file_size_bytes == sample_artifact.file_size_bytes


class TestRequestLogSerialization:
    """Test RequestLog serialization."""

    def test_request_log_to_dict(self, sample_request_log):
        """Test converting RequestLog to dict."""
        data = sample_request_log.to_dict()

        assert isinstance(data, dict)
        assert data["log_id"] == sample_request_log.log_id
        assert data["status"] == sample_request_log.status

    def test_request_log_from_dict(self, sample_request_log):
        """Test creating RequestLog from dict."""
        data = sample_request_log.to_dict()
        restored = RequestLog.from_dict(data)

        assert restored.log_id == sample_request_log.log_id
        assert restored.status == sample_request_log.status


class TestSQLiteStoreConcurrency:
    """Test concurrent access patterns."""

    def test_save_multiple_projects_sequentially(self, sqlite_store):
        """Test saving multiple projects."""
        for i in range(5):
            project = StoredProject(
                project_id=f"proj_{i}",
                user_id="user_123",
                session_id=f"sess_{i}",
                raw_text="Test",
                project_name=f"project_{i}",
                summary="Test project",
                project_type="application",
                features_json="[]",
                tech_stack_json="{}",
                status="completed",
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
            )
            sqlite_store.save_project(project)

        projects = sqlite_store.list_projects()
        assert len(projects) >= 5

    def test_multiple_updates_to_same_project(self, sqlite_store, sample_project):
        """Test updating same project multiple times."""
        sqlite_store.save_project(sample_project)

        # Update status multiple times
        for status in ["in_progress", "completed", "failed", "completed"]:
            sqlite_store.update_project_status(sample_project.project_id, status)

        retrieved = sqlite_store.get_project(sample_project.project_id)
        assert retrieved.status == "completed"
