from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class GenerationConfig:
    """Configuration for autoregressive text generation."""

    max_new_tokens: int = 128
    min_new_tokens: int = 0

    temperature: float = 1.0
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    do_sample: bool = True

    repetition_penalty: float = 1.0

    stop_token_ids: Optional[List[int]] = field(default_factory=list)
    eos_token_id: Optional[int] = None
    pad_token_id: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "GenerationConfig":
        return cls(**config_dict)

    def copy(self, **updates: Any) -> "GenerationConfig":
        data = self.to_dict()
        data.update(updates)
        return self.__class__(**data)

    def validate(self) -> None:
        if self.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be > 0")
        if self.min_new_tokens < 0:
            raise ValueError("min_new_tokens must be >= 0")
        if self.min_new_tokens > self.max_new_tokens:
            raise ValueError("min_new_tokens cannot exceed max_new_tokens")

        if self.temperature <= 0:
            raise ValueError("temperature must be > 0")

        if self.top_k is not None and self.top_k <= 0:
            raise ValueError("top_k must be > 0 when provided")

        if self.top_p is not None and not (0.0 < self.top_p <= 1.0):
            raise ValueError("top_p must be in range (0.0, 1.0] when provided")

        if self.repetition_penalty <= 0:
            raise ValueError("repetition_penalty must be > 0")

        if self.eos_token_id is not None and self.eos_token_id < 0:
            raise ValueError("eos_token_id must be >= 0 when provided")

        if self.pad_token_id is not None and self.pad_token_id < 0:
            raise ValueError("pad_token_id must be >= 0 when provided")

        if self.stop_token_ids is not None:
            for token_id in self.stop_token_ids:
                if token_id < 0:
                    raise ValueError("stop_token_ids must contain only >= 0 values")