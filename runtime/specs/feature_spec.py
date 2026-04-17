from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class FeatureSpec:
    name: str
    description: str
    priority: str = "medium"
    required: bool = True
    dependencies: List[str] = field(default_factory=list)
    metadata: Dict[str, object] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.name.strip():
            raise ValueError("Feature name cannot be empty")
        if not self.description.strip():
            raise ValueError("Feature description cannot be empty")
        if self.priority not in {"low", "medium", "high", "critical"}:
            raise ValueError("priority must be one of: low, medium, high, critical")