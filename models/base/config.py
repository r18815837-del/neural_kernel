from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict


@dataclass
class ModelConfig:
    """Base configuration for a model.

    This is the canonical container for architecture-level settings.
    Specific model families can subclass it and add extra fields.
    """

    model_name: str = "base_model"
    vocab_size: int = 0
    hidden_size: int = 128
    num_layers: int = 1
    num_heads: int = 1
    max_seq_len: int = 128
    dropout: float = 0.0

    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2

    tie_embeddings: bool = False
    use_bias: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "ModelConfig":
        return cls(**config_dict)

    def copy(self, **updates: Any) -> "ModelConfig":
        data = self.to_dict()
        data.update(updates)
        return self.__class__(**data)

    def validate(self) -> None:
        if self.vocab_size < 0:
            raise ValueError("vocab_size must be >= 0")
        if self.hidden_size <= 0:
            raise ValueError("hidden_size must be > 0")
        if self.num_layers <= 0:
            raise ValueError("num_layers must be > 0")
        if self.num_heads <= 0:
            raise ValueError("num_heads must be > 0")
        if self.max_seq_len <= 0:
            raise ValueError("max_seq_len must be > 0")
        if not (0.0 <= self.dropout < 1.0):
            raise ValueError("dropout must be in range [0.0, 1.0)")
        if self.pad_token_id < 0:
            raise ValueError("pad_token_id must be >= 0")
        if self.bos_token_id < 0:
            raise ValueError("bos_token_id must be >= 0")
        if self.eos_token_id < 0:
            raise ValueError("eos_token_id must be >= 0")
        if self.hidden_size % self.num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")