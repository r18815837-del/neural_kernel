from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from models.base.generation_config import GenerationConfig


@dataclass
class DecodingPolicy:
    generation_config: GenerationConfig

    def validate(self) -> None:
        self.generation_config.validate()

    @property
    def max_new_tokens(self) -> int:
        return self.generation_config.max_new_tokens

    @property
    def temperature(self) -> float:
        return self.generation_config.temperature

    @property
    def top_k(self) -> Optional[int]:
        return self.generation_config.top_k

    @property
    def top_p(self) -> Optional[float]:
        return self.generation_config.top_p

    @property
    def do_sample(self) -> bool:
        return self.generation_config.do_sample