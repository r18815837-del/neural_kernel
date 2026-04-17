from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class GenerationResponse:
    text: str
    token_ids: List[int] = field(default_factory=list)
    finish_reason: str = "completed"
    metadata: Dict[str, object] = field(default_factory=dict)


@dataclass
class ErrorResponse:
    error: str
    message: str
    metadata: Dict[str, object] = field(default_factory=dict)