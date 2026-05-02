from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SpecialTokens:
    pad_token: str = "<pad>"
    unk_token: str = "<unk>"
    bos_token: str = "<bos>"
    eos_token: str = "<eos>"
    system_token: str = "<system>"
    user_token: str = "<user>"
    assistant_token: str = "<assistant>"