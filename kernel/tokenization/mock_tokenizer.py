from __future__ import annotations

from typing import Sequence

from .base import BaseTokenizer
from .types import TokenizerInfo, ChatMessage


class MockTokenizer(BaseTokenizer):
    def __init__(self) -> None:
        self._token_to_id = {
            "<pad>": 0,
            "<bos>": 1,
            "<eos>": 2,
            "<unk>": 3,
        }
        self._id_to_token = {v: k for k, v in self._token_to_id.items()}

    def info(self) -> TokenizerInfo:
        return TokenizerInfo(
            name="mock-tokenizer",
            vocab_size=len(self._token_to_id),
            bos_token_id=1,
            eos_token_id=2,
            pad_token_id=0,
            unk_token_id=3,
            model_max_length=4096,
            supports_chat_template=True,
        )

    def _get_or_create_id(self, token: str) -> int:
        if token not in self._token_to_id:
            token_id = len(self._token_to_id)
            self._token_to_id[token] = token_id
            self._id_to_token[token_id] = token
        return self._token_to_id[token]

    def encode(
        self,
        text: str,
        *,
        add_special_tokens: bool = True,
    ) -> list[int]:
        pieces = text.strip().split() if text.strip() else []
        ids = [self._get_or_create_id(piece) for piece in pieces]

        if add_special_tokens:
            return [1, *ids, 2]
        return ids

    def decode(
        self,
        token_ids: Sequence[int],
        *,
        skip_special_tokens: bool = False,
    ) -> str:
        special_ids = {0, 1, 2}
        tokens: list[str] = []

        for token_id in token_ids:
            if skip_special_tokens and token_id in special_ids:
                continue
            tokens.append(self._id_to_token.get(token_id, "<unk>"))

        return " ".join(tokens)

    def apply_chat_template(
        self,
        messages: Sequence[ChatMessage],
        *,
        add_generation_prompt: bool = True,
        tokenize: bool = False,
    ) -> str | list[int]:
        parts: list[str] = []

        for msg in messages:
            parts.append(f"<{msg.role}> {msg.content}")

        if add_generation_prompt:
            parts.append("<assistant>")

        prompt = "\n".join(parts)

        if tokenize:
            return self.encode(prompt, add_special_tokens=True)
        return prompt