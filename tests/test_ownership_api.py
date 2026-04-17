"""Integration tests for ownership across the full API stack.

Tests owner binding on project creation and end-to-end ownership
enforcement through the generation → client routes flow.
"""
from __future__ import annotations

import sys
import uuid
from datetime import datetime
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest

try:
    from fastapi.testclient import TestClient
    from fastapi import FastAPI

    _HAS_FASTAPI = True
except ImportError:
    _HAS_FASTAPI = False

if _HAS_FASTAPI:
    from api.auth.models import AuthContext, ANONYMOUS
    from api.auth.dependencies import require_client_auth, optional_auth
    from api.auth.config import AuthConfig
    from api.dependencies import (
        get_assistant_manager,
        get_contract_service,
        get_project_store,
        get_store,
        get_access_policy,
    )
    from api.routes.generation import router as gen_router
    from api.routes.client import router as client_router
    from integration.client_contract import ClientContractService
    from persistence.access import OwnershipAccessPolicy
    from persistence.models import StoredProject

pytestmark = pytest.mark.skipif(not _HAS_FASTAPI, reason="fastapi not installed")


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _auth(client_id: str = "c1") -> AuthContext:
    return AuthContext(authenticated=True, auth_type="api_key", client_id=client_id)


@pytest.fixture()
def full_app():
    """App with both generation and client routers."""
    app = FastAPI()
    app.include_router(gen_router)
    app.include_router(client_router)

    store_mock = MagicMock()
    store_mock.get_project.return_value = None
    store_mock.list_projects.return_value = []
    store_mock.save_project.return_value = "pid"
    store_mock.log_request.return_value = "lid"

    project_store: Dict[str, Any] = {}
    contract = ClientContractService(base_url="/api/v1")
    policy = OwnershipAccessPolicy(allow_legacy_ownerless=True)

    # Mock the assistant manager
    manager_mock = MagicMock()

    app.dependency_overrides[get_store] = lambda: store_mock
    app.dependency_overrides[get_project_store] = lambda: project_store
    app.dependency_overrides[get_contract_service] = lambda: contract
    app.dependency_overrides[get_access_policy] = lambda: policy
    app.dependency_overrides[get_assistant_manager] = lambda: manager_mock

    return app, store_mock, project_store, policy


# ------------------------------------------------------------------
# Owner binding on project creation
# ------------------------------------------------------------------


class TestOwnerBindingOnCreate:
    def test_authenticated_create_sets_owner(self, full_app):
        app, store_mock, project_store, _ = full_app
        app.dependency_overrides[optional_auth] = lambda: _auth("c1")
        app.dependency_overrides[require_client_auth] = lambda: _auth("c1")

        client = TestClient(app)
        resp = client.post(
            "/api/v1/generate",
            json={"text": "Build me a todo app"},
        )
        assert resp.status_code == 200
        data = resp.json()
        pid = data["project_id"]

        # Check in-memory store got owner
        assert project_store[pid]["owner_client_id"] == "c1"

        # Check persistent store got owner in save_project call
        save_call = store_mock.save_project.call_args
        saved_project = save_call[0][0]
        assert saved_project.owner_client_id == "c1"

    def test_anonymous_create_has_no_owner(self, full_app):
        app, store_mock, project_store, _ = full_app
        app.dependency_overrides[optional_auth] = lambda: ANONYMOUS
        app.dependency_overrides[require_client_auth] = lambda: ANONYMOUS

        client = TestClient(app)
        resp = client.post(
            "/api/v1/generate",
            json={"text": "Build me a todo app"},
        )
        assert resp.status_code == 200
        pid = resp.json()["project_id"]

        assert project_store[pid]["owner_client_id"] is None

        saved_project = store_mock.save_project.call_args[0][0]
        assert saved_project.owner_client_id is None


# ------------------------------------------------------------------
# Cross-client isolation
# ------------------------------------------------------------------


class TestCrossClientIsolation:
    def test_client_a_cannot_see_client_b_project(self, full_app):
        app, store_mock, project_store, _ = full_app

        # Project owned by c_other
        now = datetime.utcnow()
        project_store["p_other"] = {
            "status": "completed",
            "message": "done",
            "created_at": now,
            "progress": 1.0,
            "artifact_path": None,
            "project_name": "OtherApp",
            "features_detected": [],
            "tech_stack": {},
            "agent_details": [],
            "error": None,
            "owner_client_id": "c_other",
        }

        # Authenticated as c1
        app.dependency_overrides[require_client_auth] = lambda: _auth("c1")
        client = TestClient(app)

        resp = client.get("/api/v1/client/status/p_other")
        assert resp.status_code == 404

    def test_client_a_can_see_own_project(self, full_app):
        app, store_mock, project_store, _ = full_app

        now = datetime.utcnow()
        project_store["p_mine"] = {
            "status": "completed",
            "message": "done",
            "created_at": now,
            "progress": 1.0,
            "artifact_path": None,
            "project_name": "MyApp",
            "features_detected": [],
            "tech_stack": {},
            "agent_details": [],
            "error": None,
            "owner_client_id": "c1",
        }

        app.dependency_overrides[require_client_auth] = lambda: _auth("c1")
        client = TestClient(app)

        resp = client.get("/api/v1/client/status/p_mine")
        assert resp.status_code == 200


# ------------------------------------------------------------------
# Legacy compatibility
# ------------------------------------------------------------------


class TestLegacyCompatibility:
    def test_ownerless_visible_when_legacy_on(self, full_app):
        app, store_mock, project_store, _ = full_app

        now = datetime.utcnow()
        project_store["p_legacy"] = {
            "status": "completed",
            "message": "done",
            "created_at": now,
            "progress": 1.0,
            "artifact_path": None,
            "project_name": "LegacyApp",
            "features_detected": [],
            "tech_stack": {},
            "agent_details": [],
            "error": None,
            "owner_client_id": None,
        }

        app.dependency_overrides[require_client_auth] = lambda: _auth("c1")
        client = TestClient(app)
        resp = client.get("/api/v1/client/status/p_legacy")
        assert resp.status_code == 200

    def test_ownerless_hidden_when_legacy_off(self, full_app):
        app, _, project_store, _ = full_app
        strict_policy = OwnershipAccessPolicy(allow_legacy_ownerless=False)
        app.dependency_overrides[get_access_policy] = lambda: strict_policy

        now = datetime.utcnow()
        project_store["p_legacy"] = {
            "status": "completed",
            "message": "done",
            "created_at": now,
            "progress": 1.0,
            "artifact_path": None,
            "project_name": "LegacyApp",
            "features_detected": [],
            "tech_stack": {},
            "agent_details": [],
            "error": None,
            "owner_client_id": None,
        }

        app.dependency_overrides[require_client_auth] = lambda: _auth("c1")
        client = TestClient(app)
        resp = client.get("/api/v1/client/status/p_legacy")
        assert resp.status_code == 404


# ------------------------------------------------------------------
# AuthConfig ownership setting
# ------------------------------------------------------------------


class TestAuthConfigOwnership:
    def test_config_default_allows_legacy(self):
        from api.auth.config import AuthConfig
        cfg = AuthConfig()
        assert cfg.allow_legacy_ownerless_access is True

    def test_config_from_env(self, monkeypatch):
        from api.auth.config import AuthConfig, reset_auth_config
        monkeypatch.setenv("NK_ALLOW_LEGACY_OWNERLESS_ACCESS", "false")
        reset_auth_config()
        cfg = AuthConfig.from_env()
        assert cfg.allow_legacy_ownerless_access is False
        reset_auth_config()
