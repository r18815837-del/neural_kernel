"""DTO for artifact download metadata."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from .project_status import FeatureItemDTO, TechStackDTO


@dataclass(frozen=True)
class ArtifactMetadataDTO:
    """Everything a client needs before downloading an artifact."""

    project_id: str
    project_name: str
    artifact_name: str
    artifact_size_bytes: int
    packaging_format: str  # zip | folder
    download_url: str
    features: List[FeatureItemDTO] = field(default_factory=list)
    tech_stack: Optional[TechStackDTO] = None
    created_at: str = ""  # ISO 8601
    version: Optional[int] = None

    def to_dict(self) -> dict[str, object]:
        d: dict[str, object] = {
            "project_id": self.project_id,
            "project_name": self.project_name,
            "artifact_name": self.artifact_name,
            "artifact_size_bytes": self.artifact_size_bytes,
            "packaging_format": self.packaging_format,
            "download_url": self.download_url,
            "created_at": self.created_at,
        }
        if self.features:
            d["features"] = [f.to_dict() for f in self.features]
        if self.tech_stack is not None:
            d["tech_stack"] = self.tech_stack.to_dict()
        if self.version is not None:
            d["version"] = self.version
        return d
