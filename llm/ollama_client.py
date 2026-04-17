"""Ollama local LLM client implementation."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any, Optional

from llm.base import BaseLLMClient
from llm.config import LLMConfig

logger = logging.getLogger(__name__)


class OllamaClient(BaseLLMClient):
    """Ollama local LLM client using HTTP API.

    Connects to Ollama instance (default: http://localhost:11434).
    Useful for offline development, free inference, and private data.

    Requires Ollama to be running. See: https://ollama.ai

    Args:
        config: LLMConfig instance with provider="ollama".

    Raises:
        ValueError: If config is invalid.
        ImportError: If httpx package is not installed.
    """

    def __init__(self, config: LLMConfig):
        """Initialize Ollama client.

        Args:
            config: LLMConfig instance with provider="ollama".

        Raises:
            ImportError: If httpx package is not installed.
        """
        config.validate()
        self.config = config
        self.base_url = config.base_url or "http://localhost:11434"

        # Import httpx here to allow optional dependency
        try:
            import httpx
        except ImportError:
            raise ImportError("httpx package required. Install with: pip install httpx")

        self.httpx = httpx
        self.client = httpx.Client(timeout=self.config.timeout)
        self.async_client = httpx.AsyncClient(timeout=self.config.timeout)

    async def complete(
        self,
        messages: list[dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> str:
        """Generate a text completion asynchronously via Ollama API.

        Args:
            messages: List of message dicts with 'role' and 'content'.
            temperature: Override config temperature.
            max_tokens: Override config max_tokens.
            **kwargs: Additional Ollama parameters.

        Returns:
            Generated text response.

        Raises:
            ValueError: If message format is invalid.
            RuntimeError: If Ollama API call fails.
        """
        self._validate_messages(messages)

        temp = temperature if temperature is not None else self.config.temperature
        max_tok = max_tokens if max_tokens is not None else self.config.max_tokens

        for attempt in range(self.config.retry_attempts):
            try:
                logger.debug(
                    f"Ollama completion (attempt {attempt + 1}/{self.config.retry_attempts}): "
                    f"model={self.config.model}, temp={temp}, max_tokens={max_tok}"
                )

                url = f"{self.base_url}/api/chat"
                payload = {
                    "model": self.config.model,
                    "messages": messages,
                    "stream": False,
                    "options": {
                        "temperature": temp,
                        "num_predict": max_tok,
                    },
                }
                payload.update(kwargs)

                response = await self.async_client.post(url, json=payload)
                response.raise_for_status()

                data = response.json()
                text = data.get("message", {}).get("content", "")
                if not text:
                    raise RuntimeError("Empty response from Ollama API")

                logger.debug(f"Ollama response received ({len(text)} chars)")

                return text

            except Exception as e:
                # Determine if error is retryable
                error_msg = str(e).lower()
                is_timeout = "timeout" in error_msg or "timed out" in error_msg
                is_connection = "connection" in error_msg or "refused" in error_msg
                is_retriable = is_timeout or is_connection

                if not is_retriable or attempt == self.config.retry_attempts - 1:
                    # Not retryable or last attempt
                    logger.error(f"Ollama API error: {e}")
                    raise RuntimeError(f"Ollama API error: {e}") from e

                # Exponential backoff
                delay = self.config.retry_delay * (2 ** attempt)
                logger.warning(
                    f"Ollama API error (retrying in {delay:.1f}s): {e}"
                )
                await asyncio.sleep(delay)

        raise RuntimeError("Max retries exceeded")

    def complete_sync(
        self,
        messages: list[dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> str:
        """Generate a text completion synchronously via Ollama API.

        Args:
            messages: List of message dicts with 'role' and 'content'.
            temperature: Override config temperature.
            max_tokens: Override config max_tokens.
            **kwargs: Additional Ollama parameters.

        Returns:
            Generated text response.

        Raises:
            ValueError: If message format is invalid.
            RuntimeError: If Ollama API call fails.
        """
        self._validate_messages(messages)

        temp = temperature if temperature is not None else self.config.temperature
        max_tok = max_tokens if max_tokens is not None else self.config.max_tokens

        for attempt in range(self.config.retry_attempts):
            try:
                logger.debug(
                    f"Ollama completion sync (attempt {attempt + 1}/{self.config.retry_attempts}): "
                    f"model={self.config.model}, temp={temp}, max_tokens={max_tok}"
                )

                url = f"{self.base_url}/api/chat"
                payload = {
                    "model": self.config.model,
                    "messages": messages,
                    "stream": False,
                    "options": {
                        "temperature": temp,
                        "num_predict": max_tok,
                    },
                }
                payload.update(kwargs)

                response = self.client.post(url, json=payload)
                response.raise_for_status()

                data = response.json()
                text = data.get("message", {}).get("content", "")
                if not text:
                    raise RuntimeError("Empty response from Ollama API")

                logger.debug(f"Ollama response received ({len(text)} chars)")

                return text

            except Exception as e:
                # Determine if error is retryable
                error_msg = str(e).lower()
                is_timeout = "timeout" in error_msg or "timed out" in error_msg
                is_connection = "connection" in error_msg or "refused" in error_msg
                is_retriable = is_timeout or is_connection

                if not is_retriable or attempt == self.config.retry_attempts - 1:
                    # Not retryable or last attempt
                    logger.error(f"Ollama API error: {e}")
                    raise RuntimeError(f"Ollama API error: {e}") from e

                # Exponential backoff
                delay = self.config.retry_delay * (2 ** attempt)
                logger.warning(
                    f"Ollama API error (retrying in {delay:.1f}s): {e}"
                )
                time.sleep(delay)

        raise RuntimeError("Max retries exceeded")

    async def complete_json(
        self,
        messages: list[dict[str, str]],
        schema: Optional[dict[str, Any]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Generate a JSON response asynchronously.

        Note: Ollama doesn't have built-in JSON mode, so we parse JSON from text response.

        Args:
            messages: List of message dicts. User message should mention JSON format.
            schema: Optional JSON schema (for documentation).
            temperature: Override config temperature.
            max_tokens: Override config max_tokens.
            **kwargs: Additional Ollama parameters.

        Returns:
            Parsed JSON response as dict.

        Raises:
            ValueError: If response is not valid JSON.
            RuntimeError: If API call fails.
        """
        # Get text response and parse JSON from it
        text = await self.complete(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )
        return self._parse_json_response(text)

    def complete_json_sync(
        self,
        messages: list[dict[str, str]],
        schema: Optional[dict[str, Any]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Generate a JSON response synchronously.

        Note: Ollama doesn't have built-in JSON mode, so we parse JSON from text response.

        Args:
            messages: List of message dicts. User message should mention JSON format.
            schema: Optional JSON schema (for documentation).
            temperature: Override config temperature.
            max_tokens: Override config max_tokens.
            **kwargs: Additional Ollama parameters.

        Returns:
            Parsed JSON response as dict.

        Raises:
            ValueError: If response is not valid JSON.
            RuntimeError: If API call fails.
        """
        # Get text response and parse JSON from it
        text = self.complete_sync(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )
        return self._parse_json_response(text)
