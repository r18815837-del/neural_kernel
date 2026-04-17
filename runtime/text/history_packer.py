from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class HistoryPacker:
    max_messages: int = 20

    def pack(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        if self.max_messages <= 0:
            raise ValueError("max_messages must be > 0")
        return messages[-self.max_messages :]

    def pack_with_system(
        self,
        messages: List[Dict[str, str]],
        system_message: Optional[str] = None,
    ) -> List[Dict[str, str]]:
        packed = self.pack(messages)

        if system_message is None:
            return packed

        return [{"role": "system", "content": system_message}] + packed