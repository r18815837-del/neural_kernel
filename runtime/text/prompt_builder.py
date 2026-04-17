from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from models.tokenizers.chat_template import ChatTemplate
from runtime.base.session import InferenceSession


@dataclass
class PromptBuilder:
    chat_template: ChatTemplate

    def build_from_messages(
        self,
        messages: List[Dict[str, str]],
        append_assistant_prefix: bool = True,
    ) -> str:
        template = ChatTemplate(
            system_prefix=self.chat_template.system_prefix,
            user_prefix=self.chat_template.user_prefix,
            assistant_prefix=self.chat_template.assistant_prefix,
            separator=self.chat_template.separator,
            append_assistant_prefix=append_assistant_prefix,
        )
        return template.format_messages(messages)

    def build_from_session(
        self,
        session: InferenceSession,
        system_message: Optional[str] = None,
        new_user_message: Optional[str] = None,
        append_assistant_prefix: bool = True,
    ) -> str:
        messages: List[Dict[str, str]] = []

        if system_message is not None:
            messages.append({"role": "system", "content": system_message})

        messages.extend(session.messages)

        if new_user_message is not None:
            messages.append({"role": "user", "content": new_user_message})

        return self.build_from_messages(
            messages=messages,
            append_assistant_prefix=append_assistant_prefix,
        )