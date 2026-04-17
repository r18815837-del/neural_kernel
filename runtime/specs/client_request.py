from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class ClientRequest:
    raw_text: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    attachments: List[str] = field(default_factory=list)
    metadata: Dict[str, object] = field(default_factory=dict)