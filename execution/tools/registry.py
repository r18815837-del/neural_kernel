from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

from .base import BaseTool


@dataclass
class ToolRegistry:
    tools: Dict[str, BaseTool] = field(default_factory=dict)

    def register(self, tool: BaseTool) -> None:
        if not tool.name.strip():
            raise ValueError("Tool name cannot be empty")
        self.tools[tool.name] = tool

    def get(self, name: str) -> BaseTool:
        if name not in self.tools:
            raise KeyError(f"Tool '{name}' is not registered")
        return self.tools[name]

    def has(self, name: str) -> bool:
        return name in self.tools

    def available(self):
        return sorted(self.tools.keys())