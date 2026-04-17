from __future__ import annotations

from typing import Any, Optional

from kernel.nn.module import Module
from models.base.generation_config import GenerationConfig
from models.base.model_output import CausalLMOutput
from models.lm.config import TokenLMConfig


class TokenLanguageModel(Module):
    """High-level token language model wrapper.

    This class should own the model-level contract:
    - config-driven construction
    - standardized forward output
    - generation API
    """

    def __init__(self, config: TokenLMConfig):
        super().__init__()
        config.validate()
        self.config = config

        # TODO:
        # здесь нужно собрать модель из твоих существующих блоков:
        # - Embedding
        # - PositionalEncoding
        # - TransformerBlock/Encoder
        # - lm_head
        #
        # Примерно:
        # self.token_embedding = ...
        # self.position_encoding = ...
        # self.encoder = ...
        # self.lm_head = ...

    def forward(
        self,
        input_ids,
        attention_mask=None,
        labels=None,
    ) -> CausalLMOutput:
        # TODO:
        # 1. embeddings
        # 2. positional encoding
        # 3. transformer stack
        # 4. lm head -> logits
        # 5. optional loss if labels provided

        logits = None
        loss = None
        hidden_states = None

        return CausalLMOutput(
            logits=logits,
            hidden_states=hidden_states,
            loss=loss,
            metadata={
                "model_name": self.config.model_name,
                "causal": self.config.causal,
            },
        )

    def generate(
        self,
        input_ids,
        generation_config: Optional[GenerationConfig] = None,
    ):
        if generation_config is None:
            generation_config = GenerationConfig()

        generation_config.validate()

        # TODO:
        # здесь должен использоваться твой текущий generation path
        # из существующего token_lm.py:
        # - greedy
        # - temperature
        # - top_k
        # - top_p
        # - causal mask
        #
        # На этом этапе можно просто прокинуть в существующую реализацию.

        raise NotImplementedError("Hook generate() to existing token_lm generation logic")