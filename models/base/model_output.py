from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class ModelOutput:
    """Standardized model output container."""

    logits: Any = None
    hidden_states: Any = None
    attentions: Any = None
    loss: Any = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "logits": self.logits,
            "hidden_states": self.hidden_states,
            "attentions": self.attentions,
            "loss": self.loss,
            "metadata": self.metadata,
        }

    def has_logits(self) -> bool:
        return self.logits is not None

    def has_hidden_states(self) -> bool:
        return self.hidden_states is not None

    def has_attentions(self) -> bool:
        return self.attentions is not None

    def has_loss(self) -> bool:
        return self.loss is not None


@dataclass
class CausalLMOutput(ModelOutput):
    """Output container for causal language models."""
    pass


@dataclass
class SequenceClassifierOutput(ModelOutput):
    """Output container for sequence classification models."""
    probabilities: Optional[Any] = None

    def to_dict(self) -> Dict[str, Any]:
        data = super().to_dict()
        data["probabilities"] = self.probabilities
        return data