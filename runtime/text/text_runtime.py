from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from models.base.generation_config import GenerationConfig
from runtime.base.request import GenerationRequest
from runtime.base.response import GenerationResponse
from runtime.base.session import InferenceSession
from runtime.text.history_packer import HistoryPacker
from runtime.text.prompt_builder import PromptBuilder


@dataclass
class TextRuntime:
    model: object
    tokenizer: object
    prompt_builder: PromptBuilder
    history_packer: HistoryPacker

    def build_generation_config(
        self,
        request: GenerationRequest,
    ) -> GenerationConfig:
        config = GenerationConfig(
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            top_k=request.top_k,
            top_p=request.top_p,
            do_sample=request.do_sample,
        )
        config.validate()
        return config

    def build_prompt(
        self,
        session: Optional[InferenceSession],
        request: GenerationRequest,
        system_message: Optional[str] = None,
    ) -> str:
        if session is None:
            return request.prompt

        packed_messages = self.history_packer.pack(session.messages)

        temp_session = InferenceSession(
            session_id=session.session_id,
            user_id=session.user_id,
            messages=packed_messages,
            metadata=dict(session.metadata),
        )

        return self.prompt_builder.build_from_session(
            session=temp_session,
            system_message=system_message,
            new_user_message=request.prompt,
            append_assistant_prefix=True,
        )

    def generate(
        self,
        request: GenerationRequest,
        session: Optional[InferenceSession] = None,
        system_message: Optional[str] = None,
    ) -> GenerationResponse:
        prompt = self.build_prompt(
            session=session,
            request=request,
            system_message=system_message,
        )

        input_ids = self.tokenizer.encode(prompt)
        generation_config = self.build_generation_config(request)

        output_ids = self.model.generate(
            input_ids=input_ids,
            generation_config=generation_config,
        )

        if not isinstance(output_ids, list):
            output_ids = list(output_ids)

        generated_text = self.tokenizer.decode(output_ids)

        finish_reason = "completed"
        if len(output_ids) == 0:
            finish_reason = "empty"

        return GenerationResponse(
            text=generated_text,
            token_ids=output_ids,
            finish_reason=finish_reason,
            metadata={
                "prompt": prompt,
                "input_length": len(input_ids),
                "output_length": len(output_ids),
            },
        )