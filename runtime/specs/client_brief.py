from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class ClientBrief:
    title: str
    summary: str
    goals: List[str] = field(default_factory=list)
    constraints: List[str] = field(default_factory=list)
    requested_features: List[str] = field(default_factory=list)
    target_users: List[str] = field(default_factory=list)
    delivery_format: Optional[str] = None
    metadata: Dict[str, object] = field(default_factory=dict)