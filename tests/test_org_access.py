"""Tests for multi-tenant org-aware access policy.

Validates the upgraded OwnershipAccessPolicy with org_id, role,
and the interaction between org membership, client ownership, and roles.
"""
from __future__ import annotations

import pytest

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from persistence.access import OwnershipAccessPolicy
from api.auth.access import (
    AccessDeniedError,
    verify_project_access,
    filter_projects_by_access,
    require_scope,
    require_role,
)
from api.auth.models import (
    AuthContext, ANONYMOUS,
    ROLE_ADMIN, ROLE_OPERATOR, ROLE_CLIENT, ROLE_VIEWER,
    SCOPE_PROJECT_CREATE, SCOPE_PROJECT_READ,
    scopes_for_role,
)


POLICY = OwnershipAccessPolicy(allow_legacy_ownerless=True)
STRICT_POLICY = OwnershipAccessPolicy(allow_legacy_ownerless=False)


def _auth(client_id="c1", org_id=None, role=ROLE_CLIENT, scopes=None):
    return AuthContext(
        authenticated=True,
        auth_type="bearer",
        client_id=client_id,
        org_id=org_id,
        role=role,
        scopes=scopes or scopes_for_role(role),
    )


# ------------------------------------------------------------------
# Roles & scopes model
# ------------------------------------------------------------------


class TestRolesAndScopes:
    def test_admin_has_admin_scope(self):
        s = scopes_for_role(ROLE_ADMIN)
        assert "admin" in s

    def test_operator_has_project_create(self):
        s = scopes_for_role(ROLE_OPERATOR)
        assert SCOPE_PROJECT_CREATE in s

    def test_viewer_cannot_create(self):
        s = scopes_for_role(ROLE_VIEWER)
        assert SCOPE_PROJECT_CREATE not in s
        assert SCOPE_PROJECT_READ in s

    def test_unknown_role_empty_scopes(self):
        assert scopes_for_role("superuser") == []

    def test_auth_context_has_role(self):
        auth = _auth(role=ROLE_ADMIN)
        assert auth.is_admin is True
        assert auth.is_operator is True  # admin is also operator-level
        assert auth.is_at_least_client is True

    def test_viewer_is_not_operator(self):
        auth = _auth(role=ROLE_VIEWER)
        assert auth.is_admin is False
        assert auth.is_operator is False
        assert auth.is_at_least_client is False


# ------------------------------------------------------------------
# Admin role — sees everything
# ------------------------------------------------------------------


class TestAdminAccess:
    def test_admin_sees_any_project(self):
        assert POLICY.can_read_project(
            caller_client_id="c1", caller_org_id="org-a", caller_role=ROLE_ADMIN,
            resource_owner_client_id="c_other", resource_org_id="org-b",
        )

    def test_admin_sees_ownerless(self):
        assert STRICT_POLICY.can_read_project(
            caller_client_id="c1", caller_org_id="org-a", caller_role=ROLE_ADMIN,
            resource_owner_client_id=None, resource_org_id=None,
        )


# ------------------------------------------------------------------
# Org-level isolation
# ------------------------------------------------------------------


class TestOrgIsolation:
    def test_same_org_operator_sees_all_in_org(self):
        assert POLICY.can_read_project(
            caller_client_id="c1", caller_org_id="org-a", caller_role=ROLE_OPERATOR,
            resource_owner_client_id="c_other", resource_org_id="org-a",
        )

    def test_different_org_operator_denied(self):
        assert not POLICY.can_read_project(
            caller_client_id="c1", caller_org_id="org-a", caller_role=ROLE_OPERATOR,
            resource_owner_client_id="c_other", resource_org_id="org-b",
        )

    def test_same_org_viewer_sees_all_in_org(self):
        assert POLICY.can_read_project(
            caller_client_id="c1", caller_org_id="org-a", caller_role=ROLE_VIEWER,
            resource_owner_client_id="c_other", resource_org_id="org-a",
        )

    def test_different_org_viewer_denied(self):
        assert not POLICY.can_read_project(
            caller_client_id="c1", caller_org_id="org-a", caller_role=ROLE_VIEWER,
            resource_owner_client_id="c_other", resource_org_id="org-b",
        )

    def test_same_org_client_sees_own_project(self):
        assert POLICY.can_read_project(
            caller_client_id="c1", caller_org_id="org-a", caller_role=ROLE_CLIENT,
            resource_owner_client_id="c1", resource_org_id="org-a",
        )

    def test_same_org_client_sees_other_in_org(self):
        """Client role members can see other projects in their org."""
        assert POLICY.can_read_project(
            caller_client_id="c1", caller_org_id="org-a", caller_role=ROLE_CLIENT,
            resource_owner_client_id="c_other", resource_org_id="org-a",
        )

    def test_different_org_client_denied(self):
        assert not POLICY.can_read_project(
            caller_client_id="c1", caller_org_id="org-a", caller_role=ROLE_CLIENT,
            resource_owner_client_id="c_other", resource_org_id="org-b",
        )

    def test_no_org_on_resource_ownership_by_client(self):
        """Resource without org but with owner — direct match works."""
        assert POLICY.can_read_project(
            caller_client_id="c1", caller_org_id="org-a", caller_role=ROLE_CLIENT,
            resource_owner_client_id="c1", resource_org_id=None,
        )

    def test_no_org_on_resource_no_owner_legacy(self):
        assert POLICY.can_read_project(
            caller_client_id="c1", caller_org_id="org-a", caller_role=ROLE_CLIENT,
            resource_owner_client_id=None, resource_org_id=None,
        )

    def test_no_org_on_resource_no_owner_strict(self):
        assert not STRICT_POLICY.can_read_project(
            caller_client_id="c1", caller_org_id="org-a", caller_role=ROLE_CLIENT,
            resource_owner_client_id=None, resource_org_id=None,
        )


# ------------------------------------------------------------------
# Client ownership (no org)
# ------------------------------------------------------------------


class TestClientOwnershipNoOrg:
    def test_owner_match(self):
        assert POLICY.can_read_project(
            caller_client_id="c1", caller_org_id=None, caller_role=ROLE_CLIENT,
            resource_owner_client_id="c1", resource_org_id=None,
        )

    def test_owner_mismatch(self):
        assert not POLICY.can_read_project(
            caller_client_id="c1", caller_org_id=None, caller_role=ROLE_CLIENT,
            resource_owner_client_id="c2", resource_org_id=None,
        )


# ------------------------------------------------------------------
# Anonymous
# ------------------------------------------------------------------


class TestAnonymousMultiTenant:
    def test_anonymous_sees_everything(self):
        assert STRICT_POLICY.can_read_project(
            caller_client_id=None, caller_org_id=None, caller_role=None,
            resource_owner_client_id="c1", resource_org_id="org-a",
        )


# ------------------------------------------------------------------
# verify helpers with org
# ------------------------------------------------------------------


class TestVerifyHelpersOrg:
    def test_verify_same_org(self):
        auth = _auth(client_id="c1", org_id="org-a", role=ROLE_CLIENT)
        project = {"owner_client_id": "c1", "org_id": "org-a"}
        verify_project_access(project, auth, POLICY, project_id="p1")

    def test_verify_different_org_denied(self):
        auth = _auth(client_id="c1", org_id="org-a", role=ROLE_CLIENT)
        project = {"owner_client_id": "c_other", "org_id": "org-b"}
        with pytest.raises(AccessDeniedError):
            verify_project_access(project, auth, POLICY, project_id="p1")

    def test_filter_org(self):
        auth = _auth(client_id="c1", org_id="org-a", role=ROLE_OPERATOR)
        projects = [
            {"owner_client_id": "c2", "org_id": "org-a"},  # same org
            {"owner_client_id": "c3", "org_id": "org-b"},  # different org
            {"owner_client_id": None, "org_id": None},      # legacy
        ]
        result = filter_projects_by_access(projects, auth, POLICY)
        assert len(result) == 2  # same org + legacy


# ------------------------------------------------------------------
# Scope / role guards
# ------------------------------------------------------------------


class TestScopeRoleGuards:
    def test_require_scope_passes(self):
        auth = _auth(role=ROLE_CLIENT)
        require_scope(auth, SCOPE_PROJECT_READ)

    def test_require_scope_fails(self):
        auth = _auth(role=ROLE_VIEWER, scopes=["project:read"])
        with pytest.raises(AccessDeniedError):
            require_scope(auth, SCOPE_PROJECT_CREATE)

    def test_require_scope_anonymous(self):
        require_scope(ANONYMOUS, SCOPE_PROJECT_CREATE)  # no-op

    def test_require_role_passes(self):
        auth = _auth(role=ROLE_ADMIN)
        require_role(auth, ROLE_ADMIN, ROLE_OPERATOR)

    def test_require_role_fails(self):
        auth = _auth(role=ROLE_VIEWER)
        with pytest.raises(AccessDeniedError):
            require_role(auth, ROLE_ADMIN, ROLE_OPERATOR)

    def test_require_role_anonymous(self):
        require_role(ANONYMOUS, ROLE_ADMIN)  # no-op
