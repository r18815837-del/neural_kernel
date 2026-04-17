from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class ToolResult:
    success: bool
    message: str
    outputs: Dict[str, object] = field(default_factory=dict)
    artifacts: List[str] = field(default_factory=list)
    error: Optional[str] = None
    metadata: Dict[str, object] = field(default_factory=dict)


class BaseTool(ABC):
    name: str = "base_tool"
    description: str = "Base execution tool"

    @abstractmethod
    def run(self, **kwargs) -> ToolResult:
        raise NotImplementedError