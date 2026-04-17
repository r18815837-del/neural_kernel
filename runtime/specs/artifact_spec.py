from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class ArtifactSpec:
    artifact_name: str
    artifact_type: str = "project_bundle"
    files: List[str] = field(default_factory=list)
    include_readme: bool = True
    include_tests: bool = True
    include_env_example: bool = True
    packaging: str = "zip"
    metadata: Dict[str, object] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.artifact_name.strip():
            raise ValueError("artifact_name cannot be empty")
        if not self.artifact_type.strip():
            raise ValueError("artifact_type cannot be empty")
        if self.packaging not in {"zip", "folder", "repo"}:
            raise ValueError("packaging must be one of: zip, folder, repo")