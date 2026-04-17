from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class InferenceSession:
    session_id: str
    user_id: Optional[str] = None
    messages: List[Dict[str, str]] = field(default_factory=list)
    metadata: Dict[str, object] = field(default_factory=dict)

    def add_message(self, role: str, content: str) -> None:
        if role not in {"system", "user", "assistant"}:
            raise ValueError(f"Unsupported role: {role}")
        self.messages.append({"role": role, "content": content})

    def clear(self) -> None:
        self.messages.clear()

    def last_message(self) -> Optional[Dict[str, str]]:
        if not self.messages:
            return None
        return self.messages[-1]