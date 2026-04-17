from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class TechStackSpec:
    backend: Optional[str] = None
    frontend: Optional[str] = None
    database: Optional[str] = None
    mobile: Optional[str] = None
    deployment: Optional[str] = None
    testing: List[str] = field(default_factory=list)
    integrations: List[str] = field(default_factory=list)
    metadata: Dict[str, object] = field(default_factory=dict)

    def validate(self) -> None:
        for field_name in ("backend", "frontend", "database", "mobile", "deployment"):
            value = getattr(self, field_name)
            if value is not None and not str(value).strip():
                raise ValueError(f"{field_name} cannot be empty string")

        for item in self.testing:
            if not str(item).strip():
                raise ValueError("testing entries cannot be empty")

        for item in self.integrations:
            if not str(item).strip():
                raise ValueError("integrations entries cannot be empty")