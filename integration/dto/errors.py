"""Client-facing error DTOs — stable, retryable-aware."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class ErrorDTO:
    """Universal client-facing error.

    ``code`` is a machine-readable string (e.g. ``"project_not_found"``),
    ``message`` is human-readable, ``retryable`` hints the client.
    """

    code: str
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    retryable: bool = False

    def to_dict(self) -> dict[str, object]:
        d: dict[str, object] = {
            "code": self.code,
            "message": self.message,
            "retryable": self.retryable,
        }
        if self.details:
            d["details"] = self.details
        return d


@dataclass(frozen=True)
class ValidationErrorDTO(ErrorDTO):
    """Input validation error with per-field details."""

    field_errors: List[Dict[str, str]] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        d = super().to_dict()
        if self.field_errors:
            d["field_errors"] = self.field_errors
        return d


@dataclass(frozen=True)
class NotFoundErrorDTO(ErrorDTO):
    """Resource-not-found error."""

    resource_type: str = "project"
    resource_id: str = ""

    def to_dict(self) -> dict[str, object]:
        d = super().to_dict()
        d["resource_type"] = self.resource_type
        d["resource_id"] = self.resource_id
        return d
