"""LLM configuration and settings."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _load_dotenv() -> None:
    """Load .env file from project root if python-dotenv is available."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        # python-dotenv not installed — rely on system env vars
        return

    # Walk up from this file to find .env in project root
    current = Path(__file__).resolve().parent
    for _ in range(5):  # max 5 levels up
        env_path = current / ".env"
        if env_path.is_file():
            load_dotenv(env_path, override=False)
            return
        current = current.parent


# Load .env on module import
_load_dotenv()


@dataclass
class LLMConfig:
    """Configuration for LLM clients.

    Supports multiple providers (OpenAI, Anthropic, Ollama) with sensible defaults.
    Configuration can be overridden via environment variables or direct parameters.

    Environment variables:
    - OPENAI_API_KEY: OpenAI API key
    - ANTHROPIC_API_KEY: Anthropic API key
    - NK_LLM_PROVIDER: Provider type ("openai", "anthropic", "ollama")
    - NK_LLM_MODEL: Default model name
    - NK_LLM_FAST_MODEL: Model for fast/simple tasks
    - NK_LLM_SMART_MODEL: Model for complex tasks
    - NK_LLM_TEMPERATURE: Temperature setting (0.0-2.0)
    - NK_LLM_MAX_TOKENS: Maximum tokens in response
    - NK_LLM_TIMEOUT: Request timeout in seconds
    - NK_LLM_RETRY_ATTEMPTS: Number of retry attempts
    """

    # Provider and model selection
    provider: str = "openai"  # "openai", "anthropic", "ollama"
    model: str = "gpt-4o-mini"  # default fast/cheap model

    # API authentication
    api_key: str | None = None  # if None, loads from env var

    # Endpoint configuration
    base_url: str | None = None  # for Ollama or custom endpoints (e.g., http://localhost:11434)

    # Generation parameters
    temperature: float = 0.3  # low for structured output, higher for creative
    max_tokens: int = 4096

    # Request handling
    timeout: int = 60  # seconds
    retry_attempts: int = 3
    retry_delay: float = 1.0  # exponential backoff: delay * (2 ** attempt)

    # Model selection by task complexity
    fast_model: str = "gpt-4o-mini"  # simple tasks: parsing, classification, extraction
    smart_model: str = "gpt-4o"  # complex tasks: architecture, code generation, analysis

    # Internal
    _resolved_api_key: str | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        """Resolve API key from parameter or environment variable."""
        if self.api_key:
            self._resolved_api_key = self.api_key
        elif self.provider == "openai":
            self._resolved_api_key = os.environ.get("OPENAI_API_KEY")
        elif self.provider == "anthropic":
            self._resolved_api_key = os.environ.get("ANTHROPIC_API_KEY")
        # Ollama doesn't require an API key

    @property
    def resolved_api_key(self) -> str | None:
        """Get the resolved API key."""
        return self._resolved_api_key

    def validate(self) -> None:
        """Validate configuration and raise errors if invalid.

        Raises:
            ValueError: If provider is invalid, API key is missing for non-Ollama providers,
                       or other configuration is invalid.
        """
        valid_providers = {"openai", "anthropic", "ollama"}
        if self.provider not in valid_providers:
            raise ValueError(
                f"Invalid provider '{self.provider}'. Must be one of: {valid_providers}"
            )

        if self.provider in ("openai", "anthropic") and not self._resolved_api_key:
            env_var = "OPENAI_API_KEY" if self.provider == "openai" else "ANTHROPIC_API_KEY"
            raise ValueError(
                f"API key required for {self.provider}. "
                f"Set {env_var} environment variable or pass api_key parameter."
            )

        if not 0.0 <= self.temperature <= 2.0:
            raise ValueError(f"Temperature must be between 0.0 and 2.0, got {self.temperature}")

        if self.max_tokens < 1:
            raise ValueError(f"max_tokens must be >= 1, got {self.max_tokens}")

        if self.timeout < 1:
            raise ValueError(f"timeout must be >= 1, got {self.timeout}")

        if self.retry_attempts < 0:
            raise ValueError(f"retry_attempts must be >= 0, got {self.retry_attempts}")

        if self.retry_delay < 0:
            raise ValueError(f"retry_delay must be >= 0, got {self.retry_delay}")

    @classmethod
    def from_env(cls) -> LLMConfig:
        """Create LLMConfig from environment variables with sensible defaults.

        Reads from environment variables:
        - NK_LLM_PROVIDER: Provider type (default: "openai")
        - NK_LLM_MODEL: Model name (default: "gpt-4o-mini")
        - NK_LLM_FAST_MODEL: Fast model (default: "gpt-4o-mini")
        - NK_LLM_SMART_MODEL: Smart model (default: "gpt-4o")
        - NK_LLM_TEMPERATURE: Temperature (default: 0.3)
        - NK_LLM_MAX_TOKENS: Max tokens (default: 4096)
        - NK_LLM_TIMEOUT: Timeout in seconds (default: 60)
        - NK_LLM_RETRY_ATTEMPTS: Retry attempts (default: 3)
        - OPENAI_API_KEY: OpenAI API key
        - ANTHROPIC_API_KEY: Anthropic API key

        Returns:
            LLMConfig: Configuration loaded from environment.
        """
        return cls(
            provider=os.environ.get("NK_LLM_PROVIDER", "openai"),
            model=os.environ.get("NK_LLM_MODEL", "gpt-4o-mini"),
            fast_model=os.environ.get("NK_LLM_FAST_MODEL", "gpt-4o-mini"),
            smart_model=os.environ.get("NK_LLM_SMART_MODEL", "gpt-4o"),
            temperature=float(os.environ.get("NK_LLM_TEMPERATURE", "0.3")),
            max_tokens=int(os.environ.get("NK_LLM_MAX_TOKENS", "4096")),
            timeout=int(os.environ.get("NK_LLM_TIMEOUT", "60")),
            retry_attempts=int(os.environ.get("NK_LLM_RETRY_ATTEMPTS", "3")),
        )
