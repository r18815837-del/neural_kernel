"""System prompts and message templates for different agent roles."""

from __future__ import annotations

from llm.prompts.system_prompts import (
    ARCHITECT_PROMPT,
    BACKEND_PROMPT,
    CODE_GENERATOR_PROMPT,
    DOCS_PROMPT,
    PRODUCT_MANAGER_PROMPT,
    QA_PROMPT,
    REQUIREMENT_PARSER_PROMPT,
)
from llm.prompts.templates import (
    build_architect_prompt,
    build_backend_code_prompt,
    build_docs_prompt,
    build_parse_requirements_prompt,
    build_product_manager_prompt,
    build_qa_test_prompt,
)

__all__ = [
    # System prompts
    "REQUIREMENT_PARSER_PROMPT",
    "ARCHITECT_PROMPT",
    "BACKEND_PROMPT",
    "PRODUCT_MANAGER_PROMPT",
    "DOCS_PROMPT",
    "QA_PROMPT",
    "CODE_GENERATOR_PROMPT",
    # Message builders
    "build_parse_requirements_prompt",
    "build_architect_prompt",
    "build_backend_code_prompt",
    "build_docs_prompt",
    "build_product_manager_prompt",
    "build_qa_test_prompt",
]
