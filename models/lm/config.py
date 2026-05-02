from __future__ import annotations

from dataclasses import dataclass

from models.base.config import ModelConfig


@dataclass
class TokenLMConfig(ModelConfig):
    """Configuration for token-level causal language models."""

    tie_embeddings: bool = True
    causal: bool = True
    use_positional_encoding: bool = True

    def validate(self) -> None:
        super().validate()

        if not isinstance(self.tie_embeddings, bool):
            raise ValueError("tie_embeddings must be a bool")
        if not isinstance(self.causal, bool):
            raise ValueError("causal must be a bool")
        if not isinstance(self.use_positional_encoding, bool):
            raise ValueError("use_positional_encoding must be a bool")