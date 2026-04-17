"""API configuration."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class ApiConfig:
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False
    cors_origins: List[str] = field(default_factory=lambda: ["*"])
    output_dir: str = "build"
    rate_limit_rpm: int = 60
