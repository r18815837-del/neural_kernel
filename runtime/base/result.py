from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class TaskResult:
    success: bool
    message: str
    artifacts: List[str] = field(default_factory=list)
    payload: Dict[str, object] = field(default_factory=dict)
    error: Optional[str] = None