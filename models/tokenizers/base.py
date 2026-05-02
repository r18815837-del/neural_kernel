from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List


class BaseTokenizer(ABC):
    """Abstract tokenizer interface."""

    @abstractmethod
    def encode(self, text: str) -> List[int]:
        raise NotImplementedError

    @abstractmethod
    def decode(self, token_ids: List[int]) -> str:
        raise NotImplementedError

    def encode_batch(self, texts: List[str]) -> List[List[int]]:
        return [self.encode(text) for text in texts]

    def decode_batch(self, batch_token_ids: List[List[int]]) -> List[str]:
        return [self.decode(token_ids) for token_ids in batch_token_ids]

    @property
    def vocab_size(self) -> int:
        raise NotImplementedError("Tokenizer must expose vocab_size")

    @property
    def pad_token_id(self) -> int:
        raise NotImplementedError("Tokenizer must expose pad_token_id")

    @property
    def bos_token_id(self) -> int:
        raise NotImplementedError("Tokenizer must expose bos_token_id")

    @property
    def eos_token_id(self) -> int:
        raise NotImplementedError("Tokenizer must expose eos_token_id")