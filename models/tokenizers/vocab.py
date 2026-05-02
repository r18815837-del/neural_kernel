from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass
class Vocabulary:
    token_to_id: Dict[str, int]

    def __post_init__(self) -> None:
        if not self.token_to_id:
            raise ValueError("Vocabulary cannot be empty")

        ids = list(self.token_to_id.values())
        if len(ids) != len(set(ids)):
            raise ValueError("Vocabulary token ids must be unique")

        self.id_to_token: Dict[int, str] = {idx: token for token, idx in self.token_to_id.items()}

    def __len__(self) -> int:
        return len(self.token_to_id)

    def has_token(self, token: str) -> bool:
        return token in self.token_to_id

    def has_id(self, idx: int) -> bool:
        return idx in self.id_to_token

    def token_id(self, token: str) -> int:
        if token not in self.token_to_id:
            raise KeyError(f"Token not found in vocabulary: {token}")
        return self.token_to_id[token]

    def token(self, idx: int) -> str:
        if idx not in self.id_to_token:
            raise KeyError(f"Token id not found in vocabulary: {idx}")
        return self.id_to_token[idx]

    def encode_tokens(self, tokens: List[str]) -> List[int]:
        return [self.token_id(token) for token in tokens]

    def decode_ids(self, ids: List[int]) -> List[str]:
        return [self.token(idx) for idx in ids]