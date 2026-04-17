from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

from .types import TokenizerInfo, ChatMessage


class BaseTokenizer(ABC):
    @abstractmethod
    def info(self) -> TokenizerInfo:
        raise NotImplementedError

    @abstractmethod
    def encode(
        self,
        text: str,
        *,
        add_special_tokens: bool = True,
    ) -> list[int]:
        raise NotImplementedError

    @abstractmethod
    def decode(
        self,
        token_ids: Sequence[int],
        *,
        skip_special_tokens: bool = False,
    ) -> str:
        raise NotImplementedError

    def count_tokens(
        self,
        text: str,
        *,
        add_special_tokens: bool = True,
    ) -> int:
        return len(self.encode(text, add_special_tokens=add_special_tokens))

    def apply_chat_template(
        self,
        messages: Sequence[ChatMessage],
        *,
        add_generation_prompt: bool = True,
        tokenize: bool = False,
    ) -> str | list[int]:
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support chat templates"
        )