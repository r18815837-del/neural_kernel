"""Abstract base class for LLM clients."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import Any, Optional


class BaseLLMClient(ABC):
    """Abstract interface for LLM clients.

    Defines the contract that all LLM provider implementations must follow.
    Supports both synchronous and asynchronous operations.
    """

    @abstractmethod
    async def complete(
        self,
        messages: list[dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> str:
        """Generate a text completion asynchronously.

        Args:
            messages: List of message dicts with 'role' and 'content' keys.
                     E.g., [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]
            temperature: Override default temperature for this request.
            max_tokens: Override default max_tokens for this request.
            **kwargs: Provider-specific options.

        Returns:
            Generated text response.

        Raises:
            ValueError: If message format is invalid.
            RuntimeError: If API call fails.
        """
        ...

    @abstractmethod
    def complete_sync(
        self,
        messages: list[dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> str:
        """Generate a text completion synchronously.

        Args:
            messages: List of message dicts with 'role' and 'content' keys.
            temperature: Override default temperature for this request.
            max_tokens: Override default max_tokens for this request.
            **kwargs: Provider-specific options.

        Returns:
            Generated text response.

        Raises:
            ValueError: If message format is invalid.
            RuntimeError: If API call fails.
        """
        ...

    @abstractmethod
    async def complete_json(
        self,
        messages: list[dict[str, str]],
        schema: Optional[dict[str, Any]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Generate a JSON response asynchronously.

        Uses native JSON response format if available (e.g., OpenAI's response_format).

        Args:
            messages: List of message dicts with 'role' and 'content' keys.
                     Last user message should instruct the model to respond with JSON.
            schema: Optional JSON schema defining expected structure.
                   Implementation-specific; some providers may use it for validation.
            temperature: Override default temperature for this request.
            max_tokens: Override default max_tokens for this request.
            **kwargs: Provider-specific options.

        Returns:
            Parsed JSON response as dict.

        Raises:
            ValueError: If response is not valid JSON.
            RuntimeError: If API call fails.
        """
        ...

    @abstractmethod
    def complete_json_sync(
        self,
        messages: list[dict[str, str]],
        schema: Optional[dict[str, Any]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Generate a JSON response synchronously.

        Uses native JSON response format if available (e.g., OpenAI's response_format).

        Args:
            messages: List of message dicts with 'role' and 'content' keys.
                     Last user message should instruct the model to respond with JSON.
            schema: Optional JSON schema defining expected structure.
                   Implementation-specific; some providers may use it for validation.
            temperature: Override default temperature for this request.
            max_tokens: Override default max_tokens for this request.
            **kwargs: Provider-specific options.

        Returns:
            Parsed JSON response as dict.

        Raises:
            ValueError: If response is not valid JSON.
            RuntimeError: If API call fails.
        """
        ...

    def _validate_messages(self, messages: list[dict[str, str]]) -> None:
        """Validate message format.

        Args:
            messages: List of message dicts to validate.

        Raises:
            ValueError: If messages are invalid.
        """
        if not messages:
            raise ValueError("messages cannot be empty")
        if not isinstance(messages, list):
            raise ValueError("messages must be a list")

        for i, msg in enumerate(messages):
            if not isinstance(msg, dict):
                raise ValueError(f"Message {i} is not a dict: {type(msg)}")
            if "role" not in msg:
                raise ValueError(f"Message {i} missing 'role' key")
            if "content" not in msg:
                raise ValueError(f"Message {i} missing 'content' key")
            if msg["role"] not in ("system", "user", "assistant"):
                raise ValueError(f"Message {i} has invalid role: {msg['role']}")

    def _parse_json_response(self, text: str) -> dict[str, Any]:
        """Parse JSON from response text, with basic error handling.

        Handles cases where LLM wraps JSON in markdown code blocks.

        Args:
            text: Response text potentially containing JSON.

        Returns:
            Parsed JSON as dict.

        Raises:
            ValueError: If text is not valid JSON.
        """
        text = text.strip()

        # Remove markdown code block wrappers if present
        if text.startswith("```json"):
            text = text[7:]
        elif text.startswith("```"):
            text = text[3:]

        if text.endswith("```"):
            text = text[:-3]

        text = text.strip()

        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in response: {e}\n\nResponse text: {text[:200]}") from e
