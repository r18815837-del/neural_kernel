"""Multi-tenant ownership access policy — pure logic, no framework deps.

Access hierarchy (highest to lowest privilege):
  1. Admin role     → sees everything in the system
  2. Operator role  → sees everything in own org
  3. Client role    → sees own projects (by client_id) + org-shared resources
  4. Viewer role    → read-only within own org
  5. Anonymous      → auth is off, sees everything (dev mode)

Resource ownership is checked via three fields:
  - ``org_id``            — tenant isolation boundary
  - ``owner_client_id``   — created-by client
  - ``owner_user_id``     — created-by user (within client/org)

Legacy resources (all ownership fields None) are controlled by
``allow_legacy_ownerless``.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class OwnershipAccessPolicy:
    """Stateless multi-tenant ownership checker.

    Parameters:
        allow_legacy_ownerless: Whether authenticated callers may access
            resources that have no ownership fields set (legacy rows).
    """

    allow_legacy_ownerless: bool = True

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_env(cls) -> OwnershipAccessPolicy:
        """Load policy from environment."""
        raw = os.getenv("NK_ALLOW_LEGACY_OWNERLESS_ACCESS", "true")
        return cls(
            allow_legacy_ownerless=raw.lower() in ("true", "1", "yes"),
        )

    # ------------------------------------------------------------------
    # Project-level checks
    # ------------------------------------------------------------------

    def can_read_project(
        self,
        *,
        caller_client_id: Optional[str] = None,
        caller_org_id: Optional[str] = None,
        caller_role: Optional[str] = None,
        resource_owner_client_id: Optional[str] = None,
        resource_org_id: Optional[str] = None,
    ) -> bool:
        """Can the caller read this project?"""
        return self._check(
            caller_client_id=caller_client_id,
            caller_org_id=caller_org_id,
            caller_role=caller_role,
            resource_owner_client_id=resource_owner_client_id,
            resource_org_id=resource_org_id,
        )

    def can_list_projects(
        self,
        *,
        caller_client_id: Optional[str] = None,
        caller_org_id: Optional[str] = None,
        caller_role: Optional[str] = None,
        resource_owner_client_id: Optional[str] = None,
        resource_org_id: Optional[str] = None,
    ) -> bool:
        """Should this project appear in a list for the caller?"""
        return self._check(
            caller_client_id=caller_client_id,
            caller_org_id=caller_org_id,
            caller_role=caller_role,
            resource_owner_client_id=resource_owner_client_id,
            resource_org_id=resource_org_id,
        )

    def can_transition(
        self,
        *,
        caller_client_id: Optional[str] = None,
        caller_org_id: Optional[str] = None,
        caller_role: Optional[str] = None,
        resource_owner_client_id: Optional[str] = None,
        resource_org_id: Optional[str] = None,
    ) -> bool:
        """Can the caller perform lifecycle transitions?"""
        return self._check(
            caller_client_id=caller_client_id,
            caller_org_id=caller_org_id,
            caller_role=caller_role,
            resource_owner_client_id=resource_owner_client_id,
            resource_org_id=resource_org_id,
        )

    # ------------------------------------------------------------------
    # Artifact-level checks
    # ------------------------------------------------------------------

    def can_read_artifact(
        self,
        *,
        caller_client_id: Optional[str] = None,
        caller_org_id: Optional[str] = None,
        caller_role: Optional[str] = None,
        resource_owner_client_id: Optional[str] = None,
        resource_org_id: Optional[str] = None,
    ) -> bool:
        """Can the caller access this artifact?"""
        return self._check(
            caller_client_id=caller_client_id,
            caller_org_id=caller_org_id,
            caller_role=caller_role,
            resource_owner_client_id=resource_owner_client_id,
            resource_org_id=resource_org_id,
        )

    # ------------------------------------------------------------------
    # Core logic
    # ------------------------------------------------------------------

    def _check(
        self,
        *,
        caller_client_id: Optional[str],
        caller_org_id: Optional[str],
        caller_role: Optional[str],
        resource_owner_client_id: Optional[str],
        resource_org_id: Optional[str],
    ) -> bool:
        """Evaluate access.

        Decision tree:
          1. Anonymous caller (client_id is None) → allowed (auth is off)
          2. Admin role → allowed unconditionally
          3. Resource is fully ownerless → allow_legacy_ownerless
          4. Org match check (if both sides have org_id):
             - Org mismatch → denied
             - Org match + operator/viewer → allowed (org-wide access)
          5. Direct client_id ownership match → allowed
          6. Otherwise → denied
        """
        # 1. Anonymous → auth off
        if caller_client_id is None:
            return True

        # 2. Admin sees everything
        if caller_role == "admin":
            return True

        # 3. Fully ownerless legacy resource
        if resource_owner_client_id is None and resource_org_id is None:
            return self.allow_legacy_ownerless

        # 4. Org-level check
        if resource_org_id is not None and caller_org_id is not None:
            if resource_org_id != caller_org_id:
                return False
            # Same org — operator and viewer get org-wide access
            if caller_role in ("operator", "viewer"):
                return True

        # 5. Direct ownership match
        if resource_owner_client_id is not None:
            if caller_client_id == resource_owner_client_id:
                return True

        # 6. Resource has org but no direct ownership? Allow within org
        #    for any authenticated member of that org.
        if resource_org_id is not None and caller_org_id == resource_org_id:
            return True

        # If resource only has owner_client_id (no org), and it doesn't
        # match the caller, deny.
        if resource_owner_client_id is not None:
            return False

        # Fallback: ownerless on one dimension
        return self.allow_legacy_ownerless
