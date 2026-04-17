"""LLM provider abstraction layer for neural_kernel.

Provides unified interface to multiple LLM providers:
- OpenAI (primary)
- Anthropic (future)
- Ollama (local/offline)
"""

from __future__ import annotations

from llm.base import BaseLLMClient
from llm.config import LLMConfig
from llm.factory import create_llm_client
from llm.openai_client import OpenAIClient

__all__ = [
    "BaseLLMClient",
    "LLMConfig",
    "OpenAIClient",
    "create_llm_client",
]
