"""Tests for OwnershipAccessPolicy and api/auth/access helpers.

Validates the pure-logic access policy and the FastAPI-level wrappers
that convert policy denials into AccessDeniedError.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import pytest

from persistence.access import OwnershipAccessPolicy
from api.auth.access import (
    AccessDeniedError,
    get_caller_client_id,
    verify_project_access,
    verify_artifact_access,
    filter_projects_by_access,
)
from api.auth.models import AuthContext, ANONYMOUS


# ------------------------------------------------------------------
# OwnershipAccessPolicy — core logic
# ------------------------------------------------------------------


class TestOwnershipAccessPolicy:
    """Pure policy tests — no FastAPI deps."""

    def test_anonymous_caller_always_allowed(self):
        policy = OwnershipAccessPolicy(allow_legacy_ownerless=False)
        assert policy.can_read_project(
            caller_client_id=None, resource_owner_client_id="c1"
        )

    def test_owner_matches(self):
        policy = OwnershipAccessPolicy()
        assert policy.can_read_project(
            caller_client_id="c1", resource_owner_client_id="c1"
        )

    def test_owner_mismatch(self):
        policy = OwnershipAccessPolicy()
        assert not policy.can_read_project(
            caller_client_id="c1", resource_owner_client_id="c2"
        )

    def test_ownerless_with_legacy_allowed(self):
        policy = OwnershipAccessPolicy(allow_legacy_ownerless=True)
        assert policy.can_read_project(
            caller_client_id="c1", resource_owner_client_id=None
        )

    def test_ownerless_with_legacy_denied(self):
        policy = OwnershipAccessPolicy(allow_legacy_ownerless=False)
        assert not policy.can_read_project(
            caller_client_id="c1", resource_owner_client_id=None
        )

    def test_can_list_delegates_to_check(self):
        policy = OwnershipAccessPolicy()
        assert policy.can_list_projects(
            caller_client_id="c1", resource_owner_client_id="c1"
        )
        assert not policy.can_list_projects(
            caller_client_id="c1", resource_owner_client_id="c2"
        )

    def test_can_transition_delegates_to_check(self):
        policy = OwnershipAccessPolicy()
        assert policy.can_transition(
            caller_client_id="c1", resource_owner_client_id="c1"
        )

    def test_can_read_artifact_delegates_to_check(self):
        policy = OwnershipAccessPolicy()
        assert policy.can_read_artifact(
            caller_client_id="c1", resource_owner_client_id="c1"
        )
        assert not policy.can_read_artifact(
            caller_client_id="c1", resource_owner_client_id="c2"
        )

    def test_from_env_default(self, monkeypatch):
        monkeypatch.delenv("NK_ALLOW_LEGACY_OWNERLESS_ACCESS", raising=False)
        p = OwnershipAccessPolicy.from_env()
        assert p.allow_legacy_ownerless is True

    def test_from_env_false(self, monkeypatch):
        monkeypatch.setenv("NK_ALLOW_LEGACY_OWNERLESS_ACCESS", "false")
        p = OwnershipAccessPolicy.from_env()
        assert p.allow_legacy_ownerless is False


# ------------------------------------------------------------------
# get_caller_client_id
# ------------------------------------------------------------------


class TestGetCallerClientId:
    def test_anonymous_returns_none(self):
        assert get_caller_client_id(ANONYMOUS) is None

    def test_authenticated_returns_client_id(self):
        auth = AuthContext(authenticated=True, client_id="c1", auth_type="api_key")
        assert get_caller_client_id(auth) == "c1"

    def test_authenticated_no_client_id_returns_none(self):
        auth = AuthContext(authenticated=True, auth_type="bearer")
        assert get_caller_client_id(auth) is None


# ------------------------------------------------------------------
# verify_project_access
# ------------------------------------------------------------------


class TestVerifyProjectAccess:
    def test_allowed_does_not_raise(self):
        auth = AuthContext(authenticated=True, client_id="c1", auth_type="api_key")
        policy = OwnershipAccessPolicy()
        project = {"owner_client_id": "c1"}
        verify_project_access(project, auth, policy, project_id="p1")

    def test_denied_raises_access_denied(self):
        auth = AuthContext(authenticated=True, client_id="c1", auth_type="api_key")
        policy = OwnershipAccessPolicy()
        project = {"owner_client_id": "c_other"}
        with pytest.raises(AccessDeniedError) as exc_info:
            verify_project_access(project, auth, policy, project_id="p1")
        assert exc_info.value.resource_type == "project"
        assert exc_info.value.resource_id == "p1"

    def test_anonymous_always_allowed(self):
        policy = OwnershipAccessPolicy(allow_legacy_ownerless=False)
        project = {"owner_client_id": "c1"}
        verify_project_access(project, ANONYMOUS, policy, project_id="p1")

    def test_works_with_object_attr(self):
        """Accepts objects with .owner_client_id attribute."""

        @dataclass
        class FakeProject:
            owner_client_id: Optional[str] = None

        auth = AuthContext(authenticated=True, client_id="c1", auth_type="api_key")
        policy = OwnershipAccessPolicy()
        verify_project_access(
            FakeProject(owner_client_id="c1"), auth, policy, project_id="p1"
        )


# ------------------------------------------------------------------
# verify_artifact_access
# ------------------------------------------------------------------


class TestVerifyArtifactAccess:
    def test_allowed(self):
        auth = AuthContext(authenticated=True, client_id="c1", auth_type="api_key")
        policy = OwnershipAccessPolicy()
        artifact = {"owner_client_id": "c1"}
        verify_artifact_access(artifact, auth, policy, artifact_id="a1")

    def test_denied(self):
        auth = AuthContext(authenticated=True, client_id="c1", auth_type="api_key")
        policy = OwnershipAccessPolicy()
        artifact = {"owner_client_id": "c_other"}
        with pytest.raises(AccessDeniedError):
            verify_artifact_access(artifact, auth, policy, artifact_id="a1")


# ------------------------------------------------------------------
# filter_projects_by_access
# ------------------------------------------------------------------


class TestFilterProjectsByAccess:
    def test_filters_out_other_owner(self):
        auth = AuthContext(authenticated=True, client_id="c1", auth_type="api_key")
        policy = OwnershipAccessPolicy()
        projects = [
            {"owner_client_id": "c1", "id": 1},
            {"owner_client_id": "c2", "id": 2},
            {"owner_client_id": None, "id": 3},
        ]
        result = filter_projects_by_access(projects, auth, policy)
        ids = [p["id"] for p in result]
        assert 1 in ids
        assert 3 in ids  # legacy ownerless
        assert 2 not in ids

    def test_anonymous_sees_all(self):
        policy = OwnershipAccessPolicy(allow_legacy_ownerless=False)
        projects = [
            {"owner_client_id": "c1"},
            {"owner_client_id": "c2"},
            {"owner_client_id": None},
        ]
        result = filter_projects_by_access(projects, ANONYMOUS, policy)
        assert len(result) == 3

    def test_legacy_denied_filters_ownerless(self):
        auth = AuthContext(authenticated=True, client_id="c1", auth_type="api_key")
        policy = OwnershipAccessPolicy(allow_legacy_ownerless=False)
        projects = [
            {"owner_client_id": "c1"},
            {"owner_client_id": None},
        ]
        result = filter_projects_by_access(projects, auth, policy)
        assert len(result) == 1
