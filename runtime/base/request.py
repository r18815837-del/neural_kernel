from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class GenerationRequest:
    prompt: str
    max_new_tokens: int = 128
    temperature: float = 1.0
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    do_sample: bool = True
    stop: List[str] = field(default_factory=list)
    metadata: Dict[str, object] = field(default_factory=dict)