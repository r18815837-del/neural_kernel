from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional

from .models import StoredProject, StoredSession, StoredArtifact, RequestLog


class BaseStore(ABC):
    """Abstract interface for data storage backends."""

    # ========== Projects ==========

    @abstractmethod
    def save_project(self, project: StoredProject) -> str:
        """
        Save or update a project.

        Args:
            project: The project to save

        Returns:
            The project_id
        """
        pass

    @abstractmethod
    def get_project(
        self,
        project_id: str,
        *,
        owner_client_id: Optional[str] = None,
        org_id: Optional[str] = None,
    ) -> Optional[StoredProject]:
        """
        Retrieve a project by ID, optionally scoped to an owner/org.

        Args:
            project_id: The project ID
            owner_client_id: If provided, only return if the project belongs
                to this client (or is ownerless, depending on policy).
            org_id: If provided, only return if the project belongs to
                this organization (or has no org).

        Returns:
            The project or None if not found / not owned
        """
        pass

    @abstractmethod
    def update_project_status(
        self, project_id: str, status: str, error: Optional[str] = None
    ) -> None:
        """
        Update a project's status.

        Args:
            project_id: The project ID
            status: New status (pending, in_progress, completed, failed)
            error: Error message if status is failed
        """
        pass

    @abstractmethod
    def list_projects(
        self,
        user_id: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
        *,
        owner_client_id: Optional[str] = None,
        org_id: Optional[str] = None,
    ) -> List[StoredProject]:
        """
        List projects with optional filtering.

        Args:
            user_id: Filter by user ID (optional)
            limit: Maximum number of results
            offset: Number of results to skip
            owner_client_id: If provided, only return projects belonging to
                this client (or ownerless, depending on policy).

        Returns:
            List of projects
        """
        pass

    # ========== Sessions ==========

    @abstractmethod
    def save_session(self, session: StoredSession) -> str:
        """
        Save or update a session.

        Args:
            session: The session to save

        Returns:
            The session_id
        """
        pass

    @abstractmethod
    def get_session(self, session_id: str) -> Optional[StoredSession]:
        """
        Retrieve a session by ID.

        Args:
            session_id: The session ID

        Returns:
            The session or None if not found
        """
        pass

    @abstractmethod
    def append_message(self, session_id: str, role: str, content: str) -> None:
        """
        Append a message to a session's message history.

        Args:
            session_id: The session ID
            role: Message role (system, user, assistant)
            content: Message content
        """
        pass

    @abstractmethod
    def list_sessions(
        self,
        user_id: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> List[StoredSession]:
        """List sessions with optional user filtering and pagination."""
        pass

    @abstractmethod
    def delete_session(self, session_id: str) -> bool:
        """Delete a session. Returns True if found and deleted."""
        pass

    @abstractmethod
    def get_session_messages(
        self,
        session_id: str,
        limit: int = 100,
        offset: int = 0,
    ) -> List[dict]:
        """Get paginated messages from a session."""
        pass

    # ========== Memory ==========

    @abstractmethod
    def save_memory_entry(
        self, key: str, answer: str, hits: int = 0, tier: str = "short"
    ) -> None:
        """Persist a memory entry to the database."""
        pass

    @abstractmethod
    def get_memory_entry(self, key: str) -> Optional[dict]:
        """Retrieve a memory entry by key."""
        pass

    @abstractmethod
    def list_memory(self, tier: Optional[str] = None, limit: int = 200) -> List[dict]:
        """List all memory entries, optionally filtered by tier."""
        pass

    @abstractmethod
    def delete_memory_entry(self, key: str) -> bool:
        """Delete a memory entry. Returns True if found."""
        pass

    # ========== Artifacts ==========

    @abstractmethod
    def save_artifact(self, artifact: StoredArtifact) -> str:
        """
        Save an artifact.

        Args:
            artifact: The artifact to save

        Returns:
            The artifact_id
        """
        pass

    @abstractmethod
    def get_artifact(self, artifact_id: str) -> Optional[StoredArtifact]:
        """
        Retrieve an artifact by ID.

        Args:
            artifact_id: The artifact ID

        Returns:
            The artifact or None if not found
        """
        pass

    @abstractmethod
    def get_artifacts_by_project(self, project_id: str) -> List[StoredArtifact]:
        """
        Retrieve all artifacts for a project.

        Args:
            project_id: The project ID

        Returns:
            List of artifacts for the project
        """
        pass

    # ========== Request Logs ==========

    @abstractmethod
    def log_request(self, log: RequestLog) -> str:
        """
        Log a request.

        Args:
            log: The request log to save

        Returns:
            The log_id
        """
        pass

    @abstractmethod
    def get_request_logs(
        self, project_id: Optional[str] = None, limit: int = 100
    ) -> List[RequestLog]:
        """
        Retrieve request logs.

        Args:
            project_id: Filter by project ID (optional)
            limit: Maximum number of results

        Returns:
            List of request logs
        """
        pass
