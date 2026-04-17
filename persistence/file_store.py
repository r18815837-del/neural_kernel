from __future__ import annotations

import json
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from .base import BaseStore
from .models import StoredProject, StoredSession, StoredArtifact, RequestLog


class FileStore(BaseStore):
    """File-based JSON storage implementation."""

    def __init__(self, store_dir: str | Path = "neural_kernel_store") -> None:
        """
        Initialize the file store.

        Args:
            store_dir: Base directory for storing files
        """
        self.store_dir = Path(store_dir)
        self._lock = threading.RLock()

        # Create directories
        self._projects_dir = self.store_dir / "projects"
        self._sessions_dir = self.store_dir / "sessions"
        self._artifacts_dir = self.store_dir / "artifacts"
        self._logs_dir = self.store_dir / "logs"

        self._projects_dir.mkdir(parents=True, exist_ok=True)
        self._sessions_dir.mkdir(parents=True, exist_ok=True)
        self._artifacts_dir.mkdir(parents=True, exist_ok=True)
        self._logs_dir.mkdir(parents=True, exist_ok=True)

    # ========== Helper methods ==========

    def _read_json_file(self, path: Path) -> dict | None:
        """Read and parse a JSON file."""
        if not path.exists():
            return None

        try:
            with open(path, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return None

    def _write_json_file(self, path: Path, data: dict) -> None:
        """Write data to a JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def _get_all_files(self, directory: Path) -> List[Path]:
        """Get all JSON files in a directory."""
        if not directory.exists():
            return []
        return sorted(directory.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)

    # ========== Projects ==========

    def save_project(self, project: StoredProject) -> str:
        """Save or update a project."""
        with self._lock:
            path = self._projects_dir / f"{project.project_id}.json"
            self._write_json_file(path, project.to_dict())
            return project.project_id

    def get_project(
        self,
        project_id: str,
        *,
        owner_client_id: Optional[str] = None,
        org_id: Optional[str] = None,
    ) -> Optional[StoredProject]:
        """Retrieve a project by ID, optionally scoped to owner."""
        with self._lock:
            path = self._projects_dir / f"{project_id}.json"
            data = self._read_json_file(path)
            if data:
                project = StoredProject.from_dict(data)
                if owner_client_id is not None:
                    if (
                        project.owner_client_id is not None
                        and project.owner_client_id != owner_client_id
                    ):
                        return None
                if org_id is not None:
                    if (
                        project.org_id is not None
                        and project.org_id != org_id
                    ):
                        return None
                return project
            return None

    def update_project_status(
        self, project_id: str, status: str, error: Optional[str] = None
    ) -> None:
        """Update a project's status."""
        with self._lock:
            project = self.get_project(project_id)
            if project:
                project.status = status
                project.error_message = error
                project.updated_at = datetime.utcnow()
                self.save_project(project)

    def list_projects(
        self,
        user_id: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
        *,
        owner_client_id: Optional[str] = None,
        org_id: Optional[str] = None,
    ) -> List[StoredProject]:
        """List projects with optional filtering."""
        with self._lock:
            files = self._get_all_files(self._projects_dir)

            projects = []
            for file_path in files:
                data = self._read_json_file(file_path)
                if data:
                    project = StoredProject.from_dict(data)
                    if user_id is not None and project.user_id != user_id:
                        continue
                    if owner_client_id is not None:
                        if (
                            project.owner_client_id is not None
                            and project.owner_client_id != owner_client_id
                        ):
                            continue
                    if org_id is not None:
                        if (
                            project.org_id is not None
                            and project.org_id != org_id
                        ):
                            continue
                    projects.append(project)

            # Sort by created_at descending
            projects.sort(key=lambda p: p.created_at, reverse=True)

            # Apply pagination
            return projects[offset : offset + limit]

    # ========== Sessions ==========

    def save_session(self, session: StoredSession) -> str:
        """Save or update a session."""
        with self._lock:
            path = self._sessions_dir / f"{session.session_id}.json"
            self._write_json_file(path, session.to_dict())
            return session.session_id

    def get_session(self, session_id: str) -> Optional[StoredSession]:
        """Retrieve a session by ID."""
        with self._lock:
            path = self._sessions_dir / f"{session_id}.json"
            data = self._read_json_file(path)
            if data:
                return StoredSession.from_dict(data)
            return None

    def append_message(self, session_id: str, role: str, content: str) -> None:
        """Append a message to a session's message history."""
        with self._lock:
            session = self.get_session(session_id)
            if session:
                session.messages.append({"role": role, "content": content})
                session.updated_at = datetime.utcnow()
                self.save_session(session)

    # ========== Artifacts ==========

    def save_artifact(self, artifact: StoredArtifact) -> str:
        """Save an artifact."""
        with self._lock:
            path = self._artifacts_dir / f"{artifact.artifact_id}.json"
            self._write_json_file(path, artifact.to_dict())
            return artifact.artifact_id

    def get_artifact(self, artifact_id: str) -> Optional[StoredArtifact]:
        """Retrieve an artifact by ID."""
        with self._lock:
            path = self._artifacts_dir / f"{artifact_id}.json"
            data = self._read_json_file(path)
            if data:
                return StoredArtifact.from_dict(data)
            return None

    def get_artifacts_by_project(self, project_id: str) -> List[StoredArtifact]:
        """Retrieve all artifacts for a project."""
        with self._lock:
            files = self._get_all_files(self._artifacts_dir)

            artifacts = []
            for file_path in files:
                data = self._read_json_file(file_path)
                if data and data.get("project_id") == project_id:
                    artifacts.append(StoredArtifact.from_dict(data))

            # Sort by created_at descending
            artifacts.sort(key=lambda a: a.created_at, reverse=True)
            return artifacts

    # ========== Request Logs ==========

    def log_request(self, log: RequestLog) -> str:
        """Log a request."""
        with self._lock:
            path = self._logs_dir / f"{log.log_id}.json"
            self._write_json_file(path, log.to_dict())
            return log.log_id

    def get_request_logs(
        self, project_id: Optional[str] = None, limit: int = 100
    ) -> List[RequestLog]:
        """Retrieve request logs."""
        with self._lock:
            files = self._get_all_files(self._logs_dir)

            logs = []
            for file_path in files:
                data = self._read_json_file(file_path)
                if data:
                    log = RequestLog.from_dict(data)
                    if project_id is None or log.project_id == project_id:
                        logs.append(log)

            # Sort by created_at descending
            logs.sort(key=lambda l: l.created_at, reverse=True)

            # Apply limit
            return logs[:limit]
