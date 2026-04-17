from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class ExecutionResult:
    task_id: str
    success: bool
    message: str
    outputs: Dict[str, object] = field(default_factory=dict)
    artifacts: List[str] = field(default_factory=list)
    error: Optional[str] = None
    metadata: Dict[str, object] = field(default_factory=dict)