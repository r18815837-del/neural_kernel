from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TokenizerInfo:
    name: str
    vocab_size: int | None = None
    bos_token_id: int | None = None
    eos_token_id: int | None = None
    pad_token_id: int | None = None
    unk_token_id: int | None = None
    model_max_length: int | None = None
    supports_chat_template: bool = False


@dataclass(frozen=True)
class ChatMessage:
    role: str
    content: str