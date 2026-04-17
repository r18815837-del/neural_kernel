"""Factory for creating LLM clients."""

from __future__ import annotations

import logging
from typing import Optional

from llm.base import BaseLLMClient
from llm.config import LLMConfig

logger = logging.getLogger(__name__)


def create_llm_client(config: Optional[LLMConfig] = None) -> Optional[BaseLLMClient]:
    """Factory function to create the appropriate LLM client.

    Creates and returns a client based on the provider specified in config.
    If no config is provided, creates default config from environment variables.
    Returns None if LLM is not available or configured.

    Args:
        config: Optional LLMConfig instance. If None, loads from environment.

    Returns:
        BaseLLMClient subclass instance (OpenAIClient, AnthropicClient, or OllamaClient),
        or None if LLM is not available.

    Examples:
        Create with explicit config:
            config = LLMConfig(provider="openai", model="gpt-4o")
            client = create_llm_client(config)

        Create from environment:
            client = create_llm_client()  # Uses env vars, returns None if no API key

        Create with custom model:
            config = LLMConfig(provider="ollama", model="mistral", base_url="http://localhost:11434")
            client = create_llm_client(config)
    """
    if config is None:
        config = LLMConfig.from_env()
        logger.debug(f"Loaded LLM config from environment: provider={config.provider}, model={config.model}")
    else:
        logger.debug(f"Using provided LLM config: provider={config.provider}, model={config.model}")

    # Validate configuration before creating client
    try:
        config.validate()
    except ValueError as e:
        logger.warning(f"LLM configuration invalid: {e}. LLM features will be disabled.")
        return None

    # Create client based on provider
    try:
        if config.provider == "openai":
            from llm.openai_client import OpenAIClient

            logger.info(f"Creating OpenAI client with model {config.model}")
            return OpenAIClient(config)

        elif config.provider == "anthropic":
            from llm.anthropic_client import AnthropicClient

            logger.info(f"Creating Anthropic client with model {config.model}")
            return AnthropicClient(config)

        elif config.provider == "ollama":
            from llm.ollama_client import OllamaClient

            base_url = config.base_url or "http://localhost:11434"
            logger.info(f"Creating Ollama client with model {config.model} at {base_url}")
            return OllamaClient(config)

        else:
            logger.error(f"Unknown provider: {config.provider}")
            return None

    except Exception as e:
        logger.error(f"Failed to create LLM client: {e}")
        return None
