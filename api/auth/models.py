"""Auth principal and context models."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


# ------------------------------------------------------------------
# Roles / scopes constants
# ------------------------------------------------------------------

# Canonical roles — used in JWT ``role`` claim and scope checks.
ROLE_ADMIN = "admin"          # full platform access
ROLE_OPERATOR = "operator"    # manage all projects in own org
ROLE_CLIENT = "client"        # create / view own projects
ROLE_VIEWER = "viewer"        # read-only access to own org
ROLE_SERVICE = "service"      # machine-to-machine (API key)

ALL_ROLES = frozenset({ROLE_ADMIN, ROLE_OPERATOR, ROLE_CLIENT, ROLE_VIEWER, ROLE_SERVICE})

# Scope strings that routes can check via ``auth.has_scope()``.
SCOPE_PROJECT_CREATE = "project:create"
SCOPE_PROJECT_READ = "project:read"
SCOPE_PROJECT_LIST = "project:list"
SCOPE_ARTIFACT_READ = "artifact:read"
SCOPE_ADMIN = "admin"

# Default scope sets per role — used when JWT omits explicit scopes.
DEFAULT_SCOPES: dict[str, List[str]] = {
    ROLE_ADMIN: [SCOPE_ADMIN, SCOPE_PROJECT_CREATE, SCOPE_PROJECT_READ, SCOPE_PROJECT_LIST, SCOPE_ARTIFACT_READ],
    ROLE_OPERATOR: [SCOPE_PROJECT_CREATE, SCOPE_PROJECT_READ, SCOPE_PROJECT_LIST, SCOPE_ARTIFACT_READ],
    ROLE_CLIENT: [SCOPE_PROJECT_CREATE, SCOPE_PROJECT_READ, SCOPE_PROJECT_LIST, SCOPE_ARTIFACT_READ],
    ROLE_VIEWER: [SCOPE_PROJECT_READ, SCOPE_PROJECT_LIST, SCOPE_ARTIFACT_READ],
    ROLE_SERVICE: [SCOPE_PROJECT_CREATE, SCOPE_PROJECT_READ, SCOPE_PROJECT_LIST, SCOPE_ARTIFACT_READ],
}


def scopes_for_role(role: str) -> List[str]:
    """Return the default scopes for a role, or empty list for unknown."""
    return list(DEFAULT_SCOPES.get(role, []))


# ------------------------------------------------------------------
# AuthContext — the single object that flows through every request
# ------------------------------------------------------------------


@dataclass(frozen=True)
class AuthContext:
    """Immutable authentication context carried through the request.

    Routes receive this via dependency injection and can inspect
    who/what is making the request without coupling to a specific
    auth mechanism.
    """

    authenticated: bool = False
    auth_type: Optional[str] = None  # "api_key" | "bearer" | None
    client_id: Optional[str] = None
    scopes: List[str] = field(default_factory=list)
    raw_subject: Optional[str] = None  # original token/key subject

    # Ownership / multi-tenancy
    user_id: Optional[str] = None
    org_id: Optional[str] = None
    role: Optional[str] = None  # canonical role from ALL_ROLES

    @property
    def is_anonymous(self) -> bool:
        return not self.authenticated

    def has_scope(self, scope: str) -> bool:
        return scope in self.scopes

    def has_role(self, role: str) -> bool:
        return self.role == role

    @property
    def is_admin(self) -> bool:
        return self.role == ROLE_ADMIN

    @property
    def is_operator(self) -> bool:
        return self.role in (ROLE_ADMIN, ROLE_OPERATOR)

    @property
    def is_at_least_client(self) -> bool:
        return self.role in (ROLE_ADMIN, ROLE_OPERATOR, ROLE_CLIENT)


# ------------------------------------------------------------------
# Concrete principal types — thin wrappers for type safety
# ------------------------------------------------------------------


@dataclass(frozen=True)
class ApiKeyPrincipal:
    """Represents a validated API key."""

    client_id: str
    scopes: List[str] = field(default_factory=lambda: ["client"])


@dataclass(frozen=True)
class BearerPrincipal:
    """Represents a validated bearer/JWT token."""

    subject: str          # "sub" claim
    client_id: Optional[str] = None
    user_id: Optional[str] = None
    org_id: Optional[str] = None
    role: Optional[str] = None
    scopes: List[str] = field(default_factory=list)
    issuer: Optional[str] = None
    expires_at: Optional[int] = None  # unix epoch

    def to_auth_context(self) -> AuthContext:
        """Convert principal to an AuthContext for request propagation."""
        effective_scopes = self.scopes or scopes_for_role(self.role or ROLE_CLIENT)
        return AuthContext(
            authenticated=True,
            auth_type="bearer",
            client_id=self.client_id or self.subject,
            user_id=self.user_id,
            org_id=self.org_id,
            role=self.role or ROLE_CLIENT,
            scopes=effective_scopes,
            raw_subject=self.subject,
        )


# Anonymous singleton
ANONYMOUS = AuthContext(authenticated=False)
