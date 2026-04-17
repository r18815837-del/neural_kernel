"""Tests for ownership enforcement in client routes.

Uses mocked dependencies to validate that the /api/v1/client/* endpoints
correctly filter and deny based on ownership.
"""
from __future__ import annotations

import json
import sys
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest

# Guard: skip if fastapi is not available
try:
    from fastapi.testclient import TestClient
    from fastapi import FastAPI

    _HAS_FASTAPI = True
except ImportError:
    _HAS_FASTAPI = False

if _HAS_FASTAPI:
    from api.routes.client import router as client_router
    from api.auth.dependencies import require_client_auth
    from api.auth.models import AuthContext, ANONYMOUS
    from api.dependencies import (
        get_contract_service,
        get_project_store,
        get_store,
        get_access_policy,
    )
    from integration.client_contract import ClientContractService
    from persistence.access import OwnershipAccessPolicy
    from persistence.models import StoredProject

pytestmark = pytest.mark.skipif(not _HAS_FASTAPI, reason="fastapi not installed")


# ------------------------------------------------------------------
# Fixtures & helpers
# ------------------------------------------------------------------


def _auth_context(client_id: str = "c1") -> AuthContext:
    return AuthContext(
        authenticated=True,
        auth_type="api_key",
        client_id=client_id,
    )


def _make_stored(
    project_id: str = "p1",
    owner_client_id: str | None = None,
    status: str = "completed",
) -> StoredProject:
    now = datetime.utcnow()
    return StoredProject(
        project_id=project_id,
        user_id="u1",
        session_id="s1",
        raw_text="test",
        project_name="TestApp",
        summary="summary",
        project_type="application",
        features_json="[]",
        tech_stack_json="{}",
        status=status,
        created_at=now,
        updated_at=now,
        owner_client_id=owner_client_id,
    )


@pytest.fixture()
def app_client():
    """Build a minimal FastAPI test client with overridden deps."""
    app = FastAPI()
    app.include_router(client_router)

    # Defaults — tests override as needed
    store_mock = MagicMock()
    store_mock.get_project.return_value = None
    store_mock.list_projects.return_value = []

    project_store: Dict[str, Any] = {}
    contract = ClientContractService(base_url="/api/v1")
    policy = OwnershipAccessPolicy(allow_legacy_ownerless=True)

    app.dependency_overrides[require_client_auth] = lambda: _auth_context("c1")
    app.dependency_overrides[get_store] = lambda: store_mock
    app.dependency_overrides[get_project_store] = lambda: project_store
    app.dependency_overrides[get_contract_service] = lambda: contract
    app.dependency_overrides[get_access_policy] = lambda: policy

    client = TestClient(app)
    return client, store_mock, project_store, policy, app


# ------------------------------------------------------------------
# /status/{project_id} ownership tests
# ------------------------------------------------------------------


class TestClientStatusOwnership:
    def test_own_project_in_memory(self, app_client):
        client, store_mock, project_store, _, _ = app_client
        project_store["p1"] = {
            "status": "completed",
            "message": "done",
            "created_at": datetime.utcnow(),
            "progress": 1.0,
            "artifact_path": None,
            "project_name": "MyApp",
            "features_detected": [],
            "tech_stack": {},
            "agent_details": [],
            "error": None,
            "owner_client_id": "c1",
        }
        resp = client.get("/api/v1/client/status/p1")
        assert resp.status_code == 200

    def test_other_owner_in_memory_returns_404(self, app_client):
        client, _, project_store, _, _ = app_client
        project_store["p1"] = {
            "status": "completed",
            "message": "done",
            "created_at": datetime.utcnow(),
            "progress": 1.0,
            "artifact_path": None,
            "project_name": "MyApp",
            "features_detected": [],
            "tech_stack": {},
            "agent_details": [],
            "error": None,
            "owner_client_id": "c_other",
        }
        resp = client.get("/api/v1/client/status/p1")
        assert resp.status_code == 404

    def test_own_project_in_store(self, app_client):
        client, store_mock, _, _, _ = app_client
        store_mock.get_project.return_value = _make_stored(
            project_id="p2", owner_client_id="c1"
        )
        resp = client.get("/api/v1/client/status/p2")
        assert resp.status_code == 200

    def test_other_owner_in_store_returns_404(self, app_client):
        client, store_mock, _, _, _ = app_client
        store_mock.get_project.return_value = _make_stored(
            project_id="p2", owner_client_id="c_other"
        )
        resp = client.get("/api/v1/client/status/p2")
        assert resp.status_code == 404

    def test_ownerless_project_allowed_by_default(self, app_client):
        client, store_mock, _, _, _ = app_client
        store_mock.get_project.return_value = _make_stored(
            project_id="p3", owner_client_id=None
        )
        resp = client.get("/api/v1/client/status/p3")
        assert resp.status_code == 200

    def test_ownerless_project_denied_when_legacy_off(self, app_client):
        client, store_mock, _, _, app = app_client
        strict_policy = OwnershipAccessPolicy(allow_legacy_ownerless=False)
        app.dependency_overrides[get_access_policy] = lambda: strict_policy
        store_mock.get_project.return_value = _make_stored(
            project_id="p3", owner_client_id=None
        )
        resp = client.get("/api/v1/client/status/p3")
        assert resp.status_code == 404


# ------------------------------------------------------------------
# /projects ownership tests
# ------------------------------------------------------------------


class TestClientListOwnership:
    def test_list_filters_by_owner(self, app_client):
        client, store_mock, _, _, _ = app_client
        store_mock.list_projects.return_value = [
            _make_stored("p1", owner_client_id="c1"),
            _make_stored("p2", owner_client_id="c_other"),
        ]
        resp = client.get("/api/v1/client/projects")
        assert resp.status_code == 200
        data = resp.json()
        project_ids = [p["project_id"] for p in data.get("projects", [])]
        assert "p1" in project_ids
        assert "p2" not in project_ids

    def test_list_includes_ownerless(self, app_client):
        client, store_mock, _, _, _ = app_client
        store_mock.list_projects.return_value = [
            _make_stored("p1", owner_client_id="c1"),
            _make_stored("p3", owner_client_id=None),
        ]
        resp = client.get("/api/v1/client/projects")
        data = resp.json()
        project_ids = [p["project_id"] for p in data.get("projects", [])]
        assert "p1" in project_ids
        assert "p3" in project_ids

    def test_list_passes_owner_to_store(self, app_client):
        client, store_mock, _, _, _ = app_client
        store_mock.list_projects.return_value = []
        client.get("/api/v1/client/projects")
        _, kwargs = store_mock.list_projects.call_args
        assert kwargs.get("owner_client_id") == "c1"


# ------------------------------------------------------------------
# /download/{project_id}/info ownership tests
# ------------------------------------------------------------------


class TestClientDownloadOwnership:
    def test_own_project_download_info(self, app_client, tmp_path):
        client, store_mock, project_store, _, _ = app_client
        artifact = tmp_path / "app.zip"
        artifact.write_bytes(b"fake zip")
        project_store["p1"] = {
            "status": "completed",
            "artifact_path": str(artifact),
            "project_name": "MyApp",
            "features_detected": [],
            "tech_stack": {},
            "completed_at": datetime.utcnow(),
            "owner_client_id": "c1",
        }
        resp = client.get("/api/v1/client/download/p1/info")
        assert resp.status_code == 200

    def test_other_owner_download_returns_404(self, app_client, tmp_path):
        client, _, project_store, _, _ = app_client
        artifact = tmp_path / "app.zip"
        artifact.write_bytes(b"fake zip")
        project_store["p1"] = {
            "status": "completed",
            "artifact_path": str(artifact),
            "project_name": "MyApp",
            "features_detected": [],
            "tech_stack": {},
            "completed_at": datetime.utcnow(),
            "owner_client_id": "c_other",
        }
        resp = client.get("/api/v1/client/download/p1/info")
        assert resp.status_code == 404

    def test_own_stored_project_download_info(self, app_client, tmp_path):
        client, store_mock, _, _, _ = app_client
        artifact = tmp_path / "app.zip"
        artifact.write_bytes(b"fake zip")
        stored = _make_stored("p2", owner_client_id="c1", status="completed")
        stored.artifact_path = str(artifact)
        store_mock.get_project.return_value = stored
        resp = client.get("/api/v1/client/download/p2/info")
        assert resp.status_code == 200

    def test_other_stored_project_download_returns_404(self, app_client, tmp_path):
        client, store_mock, _, _, _ = app_client
        artifact = tmp_path / "app.zip"
        artifact.write_bytes(b"fake zip")
        stored = _make_stored("p2", owner_client_id="c_other", status="completed")
        stored.artifact_path = str(artifact)
        store_mock.get_project.return_value = stored
        resp = client.get("/api/v1/client/download/p2/info")
        assert resp.status_code == 404


# ------------------------------------------------------------------
# Anonymous access (auth disabled)
# ------------------------------------------------------------------


class TestAnonymousAccess:
    def test_anonymous_sees_all_projects(self, app_client):
        client, store_mock, _, _, app = app_client
        app.dependency_overrides[require_client_auth] = lambda: ANONYMOUS
        store_mock.list_projects.return_value = [
            _make_stored("p1", owner_client_id="c1"),
            _make_stored("p2", owner_client_id="c2"),
        ]
        resp = client.get("/api/v1/client/projects")
        data = resp.json()
        assert len(data.get("projects", [])) == 2

    def test_anonymous_sees_any_project_status(self, app_client):
        client, store_mock, _, _, app = app_client
        app.dependency_overrides[require_client_auth] = lambda: ANONYMOUS
        store_mock.get_project.return_value = _make_stored(
            "p1", owner_client_id="c_other"
        )
        resp = client.get("/api/v1/client/status/p1")
        assert resp.status_code == 200
