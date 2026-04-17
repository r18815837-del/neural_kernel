"""Ownership access-control helpers for FastAPI routes.

Thin wrappers around ``OwnershipAccessPolicy`` that raise HTTP 404
(preferred over 403 to avoid leaking existence) when the caller
doesn't own the resource.

All helpers extract caller identity (client_id, org_id, role) from
``AuthContext`` and resource ownership from the resource object.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from persistence.access import OwnershipAccessPolicy
from .models import AuthContext


class AccessDeniedError(Exception):
    """Raised when ownership check fails — route should return 404."""

    def __init__(self, resource_type: str = "resource", resource_id: str = ""):
        self.resource_type = resource_type
        self.resource_id = resource_id
        super().__init__(f"{resource_type} not found: {resource_id}")


# ------------------------------------------------------------------
# Identity extraction
# ------------------------------------------------------------------


def get_caller_client_id(auth: AuthContext) -> Optional[str]:
    """Extract the effective client_id from the auth context.

    Returns None when auth is disabled (anonymous), which causes the
    policy to allow everything.
    """
    if auth.is_anonymous:
        return None
    return auth.client_id


def get_caller_org_id(auth: AuthContext) -> Optional[str]:
    """Extract org_id from auth context, or None if anonymous."""
    if auth.is_anonymous:
        return None
    return auth.org_id


def get_caller_role(auth: AuthContext) -> Optional[str]:
    """Extract role from auth context, or None if anonymous."""
    if auth.is_anonymous:
        return None
    return auth.role


# ------------------------------------------------------------------
# Verification helpers
# ------------------------------------------------------------------


def verify_project_access(
    project: object,
    auth: AuthContext,
    policy: OwnershipAccessPolicy,
    *,
    project_id: str = "",
) -> None:
    """Raise ``AccessDeniedError`` if the caller can't read this project.

    Works with both ``StoredProject`` (has attributes) and
    in-memory dicts (has keys).
    """
    if not policy.can_read_project(
        caller_client_id=get_caller_client_id(auth),
        caller_org_id=get_caller_org_id(auth),
        caller_role=get_caller_role(auth),
        resource_owner_client_id=_extract_field(project, "owner_client_id"),
        resource_org_id=_extract_field(project, "org_id"),
    ):
        raise AccessDeniedError("project", project_id)


def verify_artifact_access(
    artifact: object,
    auth: AuthContext,
    policy: OwnershipAccessPolicy,
    *,
    artifact_id: str = "",
) -> None:
    """Raise ``AccessDeniedError`` if the caller can't read this artifact."""
    if not policy.can_read_artifact(
        caller_client_id=get_caller_client_id(auth),
        caller_org_id=get_caller_org_id(auth),
        caller_role=get_caller_role(auth),
        resource_owner_client_id=_extract_field(artifact, "owner_client_id"),
        resource_org_id=_extract_field(artifact, "org_id"),
    ):
        raise AccessDeniedError("artifact", artifact_id)


def filter_projects_by_access(
    projects: list,
    auth: AuthContext,
    policy: OwnershipAccessPolicy,
) -> list:
    """Return only the projects the caller is allowed to see."""
    caller_cid = get_caller_client_id(auth)
    caller_oid = get_caller_org_id(auth)
    caller_role = get_caller_role(auth)
    return [
        p for p in projects
        if policy.can_list_projects(
            caller_client_id=caller_cid,
            caller_org_id=caller_oid,
            caller_role=caller_role,
            resource_owner_client_id=_extract_field(p, "owner_client_id"),
            resource_org_id=_extract_field(p, "org_id"),
        )
    ]


# ------------------------------------------------------------------
# Scope / role guard helpers
# ------------------------------------------------------------------


def require_scope(auth: AuthContext, scope: str) -> None:
    """Raise AccessDeniedError if the caller lacks a required scope."""
    if auth.is_anonymous:
        return  # auth is off
    if not auth.has_scope(scope):
        raise AccessDeniedError("scope", scope)


def require_role(auth: AuthContext, *roles: str) -> None:
    """Raise AccessDeniedError if the caller's role is not in allowed set."""
    if auth.is_anonymous:
        return  # auth is off
    if auth.role not in roles:
        raise AccessDeniedError("role", auth.role or "none")


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _extract_field(resource: object, field: str) -> Optional[str]:
    """Pull a field from a StoredProject, dict, or similar."""
    if isinstance(resource, dict):
        return resource.get(field)
    return getattr(resource, field, None)
