from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .feature_spec import FeatureSpec
from .tech_stack_spec import TechStackSpec


@dataclass
class ProjectSpec:
    project_name: str
    summary: str
    project_type: str = "application"
    features: List[FeatureSpec] = field(default_factory=list)
    tech_stack: Optional[TechStackSpec] = None
    output_format: str = "zip"
    target_platforms: List[str] = field(default_factory=list)
    constraints: List[str] = field(default_factory=list)
    metadata: Dict[str, object] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.project_name.strip():
            raise ValueError("project_name cannot be empty")
        if not self.summary.strip():
            raise ValueError("summary cannot be empty")
        if not self.project_type.strip():
            raise ValueError("project_type cannot be empty")
        if self.output_format not in {"zip", "folder", "repo"}:
            raise ValueError("output_format must be one of: zip, folder, repo")

        for feature in self.features:
            feature.validate()

        if self.tech_stack is not None:
            self.tech_stack.validate()