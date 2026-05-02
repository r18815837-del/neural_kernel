from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class ChatTemplate:
    """Converts structured chat messages into a single prompt string."""

    system_prefix: str = "<system>\n"
    user_prefix: str = "<user>\n"
    assistant_prefix: str = "<assistant>\n"
    separator: str = "\n"
    append_assistant_prefix: bool = True

    def format_message(self, role: str, content: str) -> str:
        if role == "system":
            return f"{self.system_prefix}{content}{self.separator}"
        if role == "user":
            return f"{self.user_prefix}{content}{self.separator}"
        if role == "assistant":
            return f"{self.assistant_prefix}{content}{self.separator}"
        raise ValueError(f"Unsupported role: {role}")

    def format_messages(self, messages: List[Dict[str, str]]) -> str:
        parts: List[str] = []

        for message in messages:
            role = message.get("role")
            content = message.get("content")

            if role is None:
                raise ValueError("Each message must contain 'role'")
            if content is None:
                raise ValueError("Each message must contain 'content'")

            parts.append(self.format_message(role, content))

        if self.append_assistant_prefix:
            parts.append(self.assistant_prefix)

        return "".join(parts)

    def build_prompt(
        self,
        user_message: str,
        system_message: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        messages: List[Dict[str, str]] = []

        if system_message is not None:
            messages.append({"role": "system", "content": system_message})

        if history:
            messages.extend(history)

        messages.append({"role": "user", "content": user_message})
        return self.format_messages(messages)