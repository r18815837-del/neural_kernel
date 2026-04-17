from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class ExecutionTask:
    task_id: str
    title: str
    task_type: str
    description: str
    inputs: Dict[str, object] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    metadata: Dict[str, object] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.task_id.strip():
            raise ValueError("task_id cannot be empty")
        if not self.title.strip():
            raise ValueError("title cannot be empty")
        if not self.task_type.strip():
            raise ValueError("task_type cannot be empty")
        if not self.description.strip():
            raise ValueError("description cannot be empty")