"""Tests for persistence.lifecycle — state machine, versioning, cleanup."""
from __future__ import annotations

import os
import sys
import tempfile
import shutil
from datetime import datetime, timedelta
from typing import Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from persistence.base import BaseStore
from persistence.models import StoredProject, StoredArtifact, StoredSession, RequestLog
from persistence.lifecycle import (
    ProjectLifecycle,
    InvalidTransitionError,
    ArtifactVersionManager,
    CleanupService,
    _TRANSITIONS,
)


# ------------------------------------------------------------------
# In-memory store stub
# ------------------------------------------------------------------

class MemoryStore(BaseStore):
    """Minimal in-memory BaseStore implementation for testing."""

    def __init__(self) -> None:
        self.projects: Dict[str, StoredProject] = {}
        self.artifacts: List[StoredArtifact] = []
        self.sessions: Dict[str, StoredSession] = {}
        self.logs: List[RequestLog] = []

    # -- Projects --
    def save_project(self, project: StoredProject) -> str:
        self.projects[project.project_id] = project
        return project.project_id

    def get_project(self, project_id: str) -> Optional[StoredProject]:
        return self.projects.get(project_id)

    def update_project_status(
        self, project_id: str, status: str, error: Optional[str] = None
    ) -> None:
        p = self.projects.get(project_id)
        if p:
            p.status = status
            p.updated_at = datetime.utcnow()
            if error:
                p.error_message = error

    def list_projects(
        self, user_id: Optional[str] = None, limit: int = 50, offset: int = 0
    ) -> List[StoredProject]:
        items = list(self.projects.values())
        if user_id:
            items = [p for p in items if p.user_id == user_id]
        return items[offset: offset + limit]

    # -- Sessions --
    def save_session(self, session: StoredSession) -> str:
        self.sessions[session.session_id] = session
        return session.session_id

    def get_session(self, session_id: str) -> Optional[StoredSession]:
        return self.sessions.get(session_id)

    def append_message(self, session_id: str, role: str, content: str) -> None:
        s = self.sessions.get(session_id)
        if s:
            s.messages.append({"role": role, "content": content})

    # -- Artifacts --
    def save_artifact(self, artifact: StoredArtifact) -> str:
        self.artifacts.append(artifact)
        return artifact.artifact_id

    def get_artifact(self, artifact_id: str) -> Optional[StoredArtifact]:
        for a in self.artifacts:
            if a.artifact_id == artifact_id:
                return a
        return None

    def get_artifacts_by_project(self, project_id: str) -> List[StoredArtifact]:
        return [a for a in self.artifacts if a.project_id == project_id]

    # -- Logs --
    def log_request(self, log: RequestLog) -> str:
        self.logs.append(log)
        return log.log_id

    def get_request_logs(
        self, project_id: Optional[str] = None, limit: int = 100
    ) -> List[RequestLog]:
        items = self.logs
        if project_id:
            items = [l for l in items if l.project_id == project_id]
        return items[:limit]


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _make_project(
    store: MemoryStore,
    pid: str = "p1",
    status: str = "pending",
    name: str = "test",
    updated_at: Optional[datetime] = None,
) -> StoredProject:
    now = updated_at or datetime.utcnow()
    p = StoredProject(
        project_id=pid,
        user_id="u1",
        session_id="s1",
        raw_text="build me an app",
        project_name=name,
        summary="test",
        project_type="application",
        features_json="[]",
        tech_stack_json="{}",
        status=status,
        created_at=now,
        updated_at=now,
    )
    store.save_project(p)
    return p


def _make_artifact(
    store: MemoryStore,
    pid: str,
    aid: str,
    filename: str,
    file_path: str = "",
    created_at: Optional[datetime] = None,
) -> StoredArtifact:
    a = StoredArtifact(
        artifact_id=aid,
        project_id=pid,
        filename=filename,
        file_path=file_path,
        file_size_bytes=100,
        format="zip",
        created_at=created_at or datetime.utcnow(),
    )
    store.save_artifact(a)
    return a


# ==================================================================
# ProjectLifecycle tests
# ==================================================================

def test_current_status_found():
    store = MemoryStore()
    _make_project(store, "p1", status="pending")
    lc = ProjectLifecycle(store=store)
    assert lc.current_status("p1") == "pending"


def test_current_status_not_found():
    lc = ProjectLifecycle(store=MemoryStore())
    assert lc.current_status("nope") is None


def test_transition_valid():
    store = MemoryStore()
    _make_project(store, "p1", status="pending")
    lc = ProjectLifecycle(store=store)
    result = lc.transition("p1", "in_progress")
    assert result == "in_progress"
    assert store.get_project("p1").status == "in_progress"


def test_transition_invalid():
    store = MemoryStore()
    _make_project(store, "p1", status="pending")
    lc = ProjectLifecycle(store=store)
    try:
        lc.transition("p1", "completed")
        assert False, "Should have raised InvalidTransitionError"
    except InvalidTransitionError:
        pass


def test_transition_not_found():
    lc = ProjectLifecycle(store=MemoryStore())
    try:
        lc.transition("nope", "in_progress")
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


def test_transition_from_terminal():
    store = MemoryStore()
    _make_project(store, "p1", status="archived")
    lc = ProjectLifecycle(store=store)
    try:
        lc.transition("p1", "pending")
        assert False, "Should have raised InvalidTransitionError"
    except InvalidTransitionError:
        pass


def test_can_transition_true():
    store = MemoryStore()
    _make_project(store, "p1", status="in_progress")
    lc = ProjectLifecycle(store=store)
    assert lc.can_transition("p1", "completed") is True


def test_can_transition_false():
    store = MemoryStore()
    _make_project(store, "p1", status="pending")
    lc = ProjectLifecycle(store=store)
    assert lc.can_transition("p1", "archived") is False


def test_allowed_transitions():
    allowed = ProjectLifecycle.allowed_transitions("pending")
    assert "in_progress" in allowed
    assert "failed" in allowed
    assert "completed" not in allowed


def test_is_terminal():
    assert ProjectLifecycle.is_terminal("archived") is True
    assert ProjectLifecycle.is_terminal("pending") is False


def test_full_lifecycle_chain():
    store = MemoryStore()
    _make_project(store, "p1", status="pending")
    lc = ProjectLifecycle(store=store)
    lc.transition("p1", "in_progress")
    lc.transition("p1", "completed")
    lc.transition("p1", "archived")
    assert store.get_project("p1").status == "archived"


def test_retry_from_failed():
    store = MemoryStore()
    _make_project(store, "p1", status="failed")
    lc = ProjectLifecycle(store=store)
    lc.transition("p1", "pending")
    assert store.get_project("p1").status == "pending"


def test_transition_with_error():
    store = MemoryStore()
    _make_project(store, "p1", status="in_progress")
    lc = ProjectLifecycle(store=store)
    lc.transition("p1", "failed", error="OOM")
    p = store.get_project("p1")
    assert p.status == "failed"
    assert p.error_message == "OOM"


# ==================================================================
# ArtifactVersionManager tests
# ==================================================================

def test_next_version_empty():
    store = MemoryStore()
    vm = ArtifactVersionManager(store=store)
    assert vm.next_version("p1") == 1


def test_next_version_increments():
    store = MemoryStore()
    _make_artifact(store, "p1", "a1", "app_v1.zip")
    _make_artifact(store, "p1", "a2", "app_v2.zip")
    vm = ArtifactVersionManager(store=store)
    assert vm.next_version("p1") == 3


def test_list_versions():
    store = MemoryStore()
    _make_artifact(store, "p1", "a1", "app_v1.zip")
    _make_artifact(store, "p1", "a2", "app_v2.zip")
    vm = ArtifactVersionManager(store=store)
    versions = vm.list_versions("p1")
    assert len(versions) == 2
    assert versions[0]["version"] == 1
    assert versions[1]["version"] == 2


def test_versioned_filename():
    assert ArtifactVersionManager.versioned_filename("myapp.zip", 3) == "myapp_v3.zip"
    assert ArtifactVersionManager.versioned_filename("project", 1) == "project_v1.zip"


def test_extract_version():
    assert ArtifactVersionManager._extract_version("app_v5.zip") == 5
    assert ArtifactVersionManager._extract_version("noversion.zip") is None


def test_retain_latest_deletes_old():
    store = MemoryStore()
    tmpdir = tempfile.mkdtemp()
    try:
        # Create 4 artifacts with actual files
        for i in range(1, 5):
            fpath = os.path.join(tmpdir, f"app_v{i}.zip")
            with open(fpath, "w") as f:
                f.write("data")
            _make_artifact(
                store, "p1", f"a{i}", f"app_v{i}.zip",
                file_path=fpath,
                created_at=datetime.utcnow() - timedelta(hours=5 - i),
            )

        vm = ArtifactVersionManager(store=store)
        deleted = vm.retain_latest("p1", keep=2)
        assert len(deleted) == 2
        # The 2 newest should still exist
        assert os.path.exists(os.path.join(tmpdir, "app_v3.zip"))
        assert os.path.exists(os.path.join(tmpdir, "app_v4.zip"))
    finally:
        shutil.rmtree(tmpdir)


def test_retain_latest_noop_when_few():
    store = MemoryStore()
    _make_artifact(store, "p1", "a1", "app_v1.zip")
    vm = ArtifactVersionManager(store=store)
    deleted = vm.retain_latest("p1", keep=3)
    assert deleted == []


# ==================================================================
# CleanupService tests
# ==================================================================

def test_archive_stale_failed():
    store = MemoryStore()
    old = datetime.utcnow() - timedelta(days=10)
    _make_project(store, "p1", status="failed", updated_at=old)
    _make_project(store, "p2", status="failed")  # recent — should stay

    cs = CleanupService(store=store)
    stats = cs.run()
    assert stats["failed_archived"] == 1
    assert store.get_project("p1").status == "archived"
    assert store.get_project("p2").status == "failed"


def test_delete_old_archived_artifacts():
    store = MemoryStore()
    tmpdir = tempfile.mkdtemp()
    try:
        old = datetime.utcnow() - timedelta(days=60)
        _make_project(store, "p1", status="archived", updated_at=old)
        fpath = os.path.join(tmpdir, "old.zip")
        with open(fpath, "w") as f:
            f.write("x")
        _make_artifact(store, "p1", "a1", "old.zip", file_path=fpath)

        cs = CleanupService(store=store)
        stats = cs.run()
        assert stats["artifacts_deleted"] == 1
        assert not os.path.exists(fpath)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_remove_orphan_dirs():
    store = MemoryStore()
    _make_project(store, "p1", name="my_project")
    tmpdir = tempfile.mkdtemp()
    try:
        # Create build dirs
        known = os.path.join(tmpdir, "my_project")
        orphan = os.path.join(tmpdir, "abandoned")
        os.makedirs(known)
        os.makedirs(orphan)

        cs = CleanupService(store=store, build_root=tmpdir)
        stats = cs.run()
        assert stats["orphan_dirs_removed"] == 1
        assert not os.path.exists(orphan)
        assert os.path.exists(known)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_cleanup_full_run_returns_dict():
    store = MemoryStore()
    cs = CleanupService(store=store)
    stats = cs.run()
    assert isinstance(stats, dict)
    assert "failed_archived" in stats
    assert "artifacts_deleted" in stats
    assert "orphan_dirs_removed" in stats


# ==================================================================
# Transition table coverage
# ==================================================================

def test_all_states_in_transitions():
    """Every state mentioned as a target should also be a key."""
    all_targets = set()
    for targets in _TRANSITIONS.values():
        all_targets |= targets
    for t in all_targets:
        assert t in _TRANSITIONS, f"State '{t}' is a target but not a key"


def test_regenerating_flow():
    store = MemoryStore()
    _make_project(store, "p1", status="completed")
    lc = ProjectLifecycle(store=store)
    lc.transition("p1", "regenerating")
    lc.transition("p1", "in_progress")
    lc.transition("p1", "completed")
    assert store.get_project("p1").status == "completed"


if __name__ == "__main__":
    import traceback

    tests = [v for k, v in list(globals().items()) if k.startswith("test_") and callable(v)]
    passed = failed = 0
    for t in tests:
        try:
            t()
            passed += 1
            print(f"  PASS: {t.__name__}")
        except Exception:
            failed += 1
            print(f"  FAIL: {t.__name__}")
            traceback.print_exc()
    print(f"\n{passed} passed, {failed} failed out of {passed + failed}")
