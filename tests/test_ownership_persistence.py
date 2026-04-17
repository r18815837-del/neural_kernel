"""Tests for ownership fields in persistence layer.

Validates that StoredProject and StoredArtifact correctly handle
owner_client_id / owner_user_id through save, get, and list operations.
"""
from __future__ import annotations

import os
import tempfile
import uuid
from datetime import datetime

import pytest

from persistence.models import StoredProject, StoredArtifact
from persistence.sqlite_store import SQLiteStore


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------


@pytest.fixture()
def store(tmp_path):
    """Fresh SQLiteStore in a temporary directory."""
    db = tmp_path / "test.db"
    return SQLiteStore(db_path=str(db))


def _make_project(
    project_id: str | None = None,
    owner_client_id: str | None = None,
    owner_user_id: str | None = None,
    status: str = "pending",
) -> StoredProject:
    now = datetime.utcnow()
    return StoredProject(
        project_id=project_id or str(uuid.uuid4()),
        user_id="u1",
        session_id="s1",
        raw_text="build me an app",
        project_name="TestApp",
        summary="Test project",
        project_type="application",
        features_json="[]",
        tech_stack_json="{}",
        status=status,
        created_at=now,
        updated_at=now,
        owner_client_id=owner_client_id,
        owner_user_id=owner_user_id,
    )


def _make_artifact(
    project_id: str,
    owner_client_id: str | None = None,
    owner_user_id: str | None = None,
) -> StoredArtifact:
    return StoredArtifact(
        artifact_id=str(uuid.uuid4()),
        project_id=project_id,
        filename="app.zip",
        file_path="/tmp/app.zip",
        file_size_bytes=1024,
        format="zip",
        created_at=datetime.utcnow(),
        owner_client_id=owner_client_id,
        owner_user_id=owner_user_id,
    )


# ------------------------------------------------------------------
# StoredProject model tests
# ------------------------------------------------------------------


class TestStoredProjectOwnership:
    def test_default_owner_fields_are_none(self):
        p = _make_project()
        assert p.owner_client_id is None
        assert p.owner_user_id is None

    def test_owner_fields_set_on_construction(self):
        p = _make_project(owner_client_id="client_1", owner_user_id="user_1")
        assert p.owner_client_id == "client_1"
        assert p.owner_user_id == "user_1"

    def test_to_dict_includes_owner_when_set(self):
        p = _make_project(owner_client_id="c1", owner_user_id="u1")
        d = p.to_dict()
        assert d["owner_client_id"] == "c1"
        assert d["owner_user_id"] == "u1"

    def test_to_dict_omits_owner_when_none(self):
        p = _make_project()
        d = p.to_dict()
        assert "owner_client_id" not in d
        assert "owner_user_id" not in d

    def test_from_dict_round_trip(self):
        p = _make_project(owner_client_id="cx", owner_user_id="ux")
        d = p.to_dict()
        p2 = StoredProject.from_dict(d)
        assert p2.owner_client_id == "cx"
        assert p2.owner_user_id == "ux"

    def test_from_dict_legacy_no_owner(self):
        p = _make_project()
        d = p.to_dict()
        # Legacy dicts won't have owner keys at all
        d.pop("owner_client_id", None)
        d.pop("owner_user_id", None)
        p2 = StoredProject.from_dict(d)
        assert p2.owner_client_id is None
        assert p2.owner_user_id is None


# ------------------------------------------------------------------
# SQLiteStore tests — ownership persistence
# ------------------------------------------------------------------


class TestSQLiteStoreOwnership:
    def test_save_and_get_with_owner(self, store):
        p = _make_project(owner_client_id="c1", owner_user_id="u1")
        store.save_project(p)
        got = store.get_project(p.project_id)
        assert got is not None
        assert got.owner_client_id == "c1"
        assert got.owner_user_id == "u1"

    def test_save_and_get_without_owner(self, store):
        p = _make_project()
        store.save_project(p)
        got = store.get_project(p.project_id)
        assert got is not None
        assert got.owner_client_id is None

    def test_get_project_owner_filter_matches(self, store):
        p = _make_project(owner_client_id="c1")
        store.save_project(p)
        got = store.get_project(p.project_id, owner_client_id="c1")
        assert got is not None
        assert got.project_id == p.project_id

    def test_get_project_owner_filter_no_match(self, store):
        p = _make_project(owner_client_id="c1")
        store.save_project(p)
        got = store.get_project(p.project_id, owner_client_id="c_other")
        assert got is None

    def test_get_project_owner_filter_allows_ownerless(self, store):
        """Ownerless projects are visible to any owner filter."""
        p = _make_project()
        store.save_project(p)
        got = store.get_project(p.project_id, owner_client_id="any")
        assert got is not None

    def test_list_projects_owner_filter(self, store):
        p1 = _make_project(owner_client_id="c1")
        p2 = _make_project(owner_client_id="c2")
        p3 = _make_project()  # ownerless
        store.save_project(p1)
        store.save_project(p2)
        store.save_project(p3)

        result = store.list_projects(owner_client_id="c1")
        ids = {r.project_id for r in result}
        assert p1.project_id in ids
        assert p3.project_id in ids  # ownerless visible
        assert p2.project_id not in ids  # other owner hidden

    def test_list_projects_no_owner_filter_returns_all(self, store):
        p1 = _make_project(owner_client_id="c1")
        p2 = _make_project(owner_client_id="c2")
        store.save_project(p1)
        store.save_project(p2)
        result = store.list_projects()
        assert len(result) == 2

    def test_artifact_owner_fields(self, store):
        p = _make_project(owner_client_id="c1")
        store.save_project(p)
        a = _make_artifact(p.project_id, owner_client_id="c1", owner_user_id="u1")
        store.save_artifact(a)
        got = store.get_artifact(a.artifact_id)
        assert got is not None
        assert got.owner_client_id == "c1"
        assert got.owner_user_id == "u1"

    def test_artifact_owner_default_none(self, store):
        p = _make_project()
        store.save_project(p)
        a = _make_artifact(p.project_id)
        store.save_artifact(a)
        got = store.get_artifact(a.artifact_id)
        assert got is not None
        assert got.owner_client_id is None


# ------------------------------------------------------------------
# StoredArtifact model tests
# ------------------------------------------------------------------


class TestStoredArtifactOwnership:
    def test_to_dict_includes_owner(self):
        a = _make_artifact("p1", owner_client_id="c1", owner_user_id="u1")
        d = a.to_dict()
        assert d["owner_client_id"] == "c1"
        assert d["owner_user_id"] == "u1"

    def test_to_dict_omits_owner_when_none(self):
        a = _make_artifact("p1")
        d = a.to_dict()
        assert "owner_client_id" not in d

    def test_from_dict_round_trip(self):
        a = _make_artifact("p1", owner_client_id="cx")
        d = a.to_dict()
        a2 = StoredArtifact.from_dict(d)
        assert a2.owner_client_id == "cx"
