"""Project lifecycle — state machine, versioning, retention, cleanup.

Provides ``ProjectLifecycle`` that enforces valid status transitions,
manages artifact version numbering, and handles old-artifact cleanup.
"""

from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Set

from persistence.base import BaseStore
from persistence.models import StoredArtifact, StoredProject

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Status state machine
# ------------------------------------------------------------------

# Allowed transitions: current_status → set of valid next statuses
_TRANSITIONS: Dict[str, Set[str]] = {
    "pending": {"in_progress", "failed", "cancelled"},
    "in_progress": {"completed", "failed", "cancelled"},
    "completed": {"archived", "regenerating"},
    "failed": {"pending", "archived"},         # retry → pending
    "cancelled": {"pending", "archived"},      # retry → pending
    "archived": set(),                         # terminal
    "regenerating": {"in_progress"},           # re-enters pipeline
}


class InvalidTransitionError(Exception):
    """Raised when a status transition is not allowed."""


@dataclass
class ProjectLifecycle:
    """Enforces status transitions and manages project lifecycle.

    Usage::

        lifecycle = ProjectLifecycle(store=store)
        lifecycle.transition(project_id, "in_progress")
    """

    store: BaseStore

    def current_status(self, project_id: str) -> Optional[str]:
        """Return the current status of a project, or None if not found."""
        project = self.store.get_project(project_id)
        return project.status if project else None

    def transition(self, project_id: str, new_status: str, error: Optional[str] = None) -> str:
        """Transition a project to ``new_status``.

        Args:
            project_id: Project identifier.
            new_status: Desired status.
            error: Error message (only for ``failed`` transitions).

        Returns:
            The new status.

        Raises:
            InvalidTransitionError: If the transition is not allowed.
            ValueError: If the project is not found.
        """
        project = self.store.get_project(project_id)
        if project is None:
            raise ValueError(f"Project '{project_id}' not found")

        old_status = project.status
        allowed = _TRANSITIONS.get(old_status, set())

        if new_status not in allowed:
            raise InvalidTransitionError(
                f"Cannot transition from '{old_status}' to '{new_status}'. "
                f"Allowed: {sorted(allowed) if allowed else 'none (terminal state)'}"
            )

        self.store.update_project_status(project_id, new_status, error=error)
        logger.info("Project %s: %s → %s", project_id, old_status, new_status)
        return new_status

    def can_transition(self, project_id: str, new_status: str) -> bool:
        """Check if a transition is valid without performing it."""
        current = self.current_status(project_id)
        if current is None:
            return False
        return new_status in _TRANSITIONS.get(current, set())

    @staticmethod
    def allowed_transitions(status: str) -> List[str]:
        """Return the list of statuses reachable from ``status``."""
        return sorted(_TRANSITIONS.get(status, set()))

    @staticmethod
    def is_terminal(status: str) -> bool:
        """Return True if the status has no outgoing transitions."""
        return len(_TRANSITIONS.get(status, set())) == 0


# ------------------------------------------------------------------
# Artifact versioning
# ------------------------------------------------------------------


@dataclass
class ArtifactVersionManager:
    """Manages artifact version numbering per project.

    Each successful generation produces a new version (1, 2, 3, …).
    Keeps all versions unless pruned via ``retain_latest()``.
    """

    store: BaseStore

    def next_version(self, project_id: str) -> int:
        """Return the next version number for a project."""
        artifacts = self.store.get_artifacts_by_project(project_id)
        if not artifacts:
            return 1
        # Parse existing versions from filename or metadata
        versions = []
        for a in artifacts:
            v = self._extract_version(a.filename)
            if v is not None:
                versions.append(v)
        return max(versions, default=0) + 1

    def list_versions(self, project_id: str) -> List[Dict[str, object]]:
        """Return all artifact versions for a project."""
        artifacts = self.store.get_artifacts_by_project(project_id)
        result = []
        for a in artifacts:
            result.append({
                "artifact_id": a.artifact_id,
                "version": self._extract_version(a.filename) or 0,
                "filename": a.filename,
                "file_size_bytes": a.file_size_bytes,
                "format": a.format,
                "created_at": a.created_at.isoformat(),
                "exists_on_disk": Path(a.file_path).exists() if a.file_path else False,
            })
        return sorted(result, key=lambda x: x["version"])

    def retain_latest(self, project_id: str, keep: int = 3) -> List[str]:
        """Delete artifact files older than the latest ``keep`` versions.

        Returns a list of deleted file paths.
        """
        artifacts = self.store.get_artifacts_by_project(project_id)
        if len(artifacts) <= keep:
            return []

        # Sort by created_at descending
        sorted_arts = sorted(artifacts, key=lambda a: a.created_at, reverse=True)
        to_remove = sorted_arts[keep:]

        deleted = []
        for a in to_remove:
            if a.file_path and Path(a.file_path).exists():
                try:
                    Path(a.file_path).unlink()
                    deleted.append(a.file_path)
                    logger.info("Deleted old artifact: %s", a.file_path)
                except OSError as e:
                    logger.warning("Failed to delete %s: %s", a.file_path, e)

        return deleted

    @staticmethod
    def versioned_filename(base_name: str, version: int) -> str:
        """Generate a versioned filename: ``project_v2.zip``."""
        stem = base_name.rsplit(".", 1)[0] if "." in base_name else base_name
        ext = base_name.rsplit(".", 1)[1] if "." in base_name else "zip"
        return f"{stem}_v{version}.{ext}"

    @staticmethod
    def _extract_version(filename: str) -> Optional[int]:
        """Extract version number from ``name_v3.zip`` pattern."""
        import re
        match = re.search(r"_v(\d+)\.", filename)
        if match:
            return int(match.group(1))
        return None


# ------------------------------------------------------------------
# Cleanup service
# ------------------------------------------------------------------


@dataclass
class CleanupService:
    """Removes stale projects and artifacts based on retention rules.

    Rules:
    - Failed projects older than ``failed_retention`` are archived.
    - Archived project artifacts older than ``archive_retention`` are deleted from disk.
    - Orphan build directories (no corresponding project) are removed.
    """

    store: BaseStore
    failed_retention: timedelta = field(default_factory=lambda: timedelta(days=7))
    archive_retention: timedelta = field(default_factory=lambda: timedelta(days=30))
    build_root: str = "build"

    def run(self) -> Dict[str, int]:
        """Execute all cleanup rules. Returns counts of actions taken."""
        stats = {
            "failed_archived": self._archive_stale_failed(),
            "artifacts_deleted": self._delete_old_archived_artifacts(),
            "orphan_dirs_removed": self._remove_orphan_dirs(),
        }
        logger.info("Cleanup complete: %s", stats)
        return stats

    def _archive_stale_failed(self) -> int:
        """Archive projects that failed more than ``failed_retention`` ago."""
        cutoff = datetime.utcnow() - self.failed_retention
        projects = self.store.list_projects(limit=1000)
        count = 0
        for p in projects:
            if p.status == "failed" and p.updated_at < cutoff:
                try:
                    self.store.update_project_status(p.project_id, "archived")
                    count += 1
                except Exception as e:
                    logger.warning("Failed to archive %s: %s", p.project_id, e)
        return count

    def _delete_old_archived_artifacts(self) -> int:
        """Delete artifact files for projects archived beyond ``archive_retention``."""
        cutoff = datetime.utcnow() - self.archive_retention
        projects = self.store.list_projects(limit=1000)
        count = 0
        for p in projects:
            if p.status == "archived" and p.updated_at < cutoff:
                artifacts = self.store.get_artifacts_by_project(p.project_id)
                for a in artifacts:
                    if a.file_path and Path(a.file_path).exists():
                        try:
                            Path(a.file_path).unlink()
                            count += 1
                        except OSError:
                            pass
        return count

    def _remove_orphan_dirs(self) -> int:
        """Remove build directories with no corresponding project."""
        build = Path(self.build_root)
        if not build.exists():
            return 0

        # Collect known project names
        projects = self.store.list_projects(limit=5000)
        known_names = {p.project_name for p in projects if p.project_name}

        count = 0
        for child in build.iterdir():
            if child.is_dir() and child.name not in known_names:
                try:
                    shutil.rmtree(child)
                    count += 1
                    logger.info("Removed orphan dir: %s", child)
                except OSError as e:
                    logger.warning("Failed to remove %s: %s", child, e)
        return count
