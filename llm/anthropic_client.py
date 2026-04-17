"""Anthropic (Claude) LLM client implementation."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any, Optional

from llm.base import BaseLLMClient
from llm.config import LLMConfig

logger = logging.getLogger(__name__)


class AnthropicClient(BaseLLMClient):
    """Anthropic API client with async/sync support and JSON handling.

    Uses the anthropic Python SDK for API interactions.
    Implements exponential backoff for retries on rate limits and transient errors.

    Args:
        config: LLMConfig instance with Anthropic settings.

    Raises:
        ValueError: If config is invalid or API key is missing.
    """

    def __init__(self, config: LLMConfig):
        """Initialize Anthropic client.

        Args:
            config: LLMConfig instance with provider="anthropic".

        Raises:
            ValueError: If Anthropic API key is not available.
            ImportError: If anthropic package is not installed.
        """
        config.validate()
        if not config.resolved_api_key:
            raise ValueError("Anthropic API key required but not found")

        self.config = config
        self.api_key = config.resolved_api_key

        # Import anthropic here to allow optional dependency
        try:
            import anthropic
        except ImportError:
            raise ImportError("anthropic package required. Install with: pip install anthropic")

        self.client = anthropic.Anthropic(api_key=self.api_key)
        self.async_client = anthropic.AsyncAnthropic(api_key=self.api_key)

    async def complete(
        self,
        messages: list[dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> str:
        """Generate a text completion asynchronously.

        Args:
            messages: List of message dicts with 'role' and 'content'.
            temperature: Override config temperature.
            max_tokens: Override config max_tokens.
            **kwargs: Additional Anthropic parameters.

        Returns:
            Generated text response.

        Raises:
            ValueError: If message format is invalid.
            RuntimeError: If API call fails after all retries.
        """
        self._validate_messages(messages)

        temp = temperature if temperature is not None else self.config.temperature
        max_tok = max_tokens if max_tokens is not None else self.config.max_tokens

        for attempt in range(self.config.retry_attempts):
            try:
                logger.debug(
                    f"Anthropic completion (attempt {attempt + 1}/{self.config.retry_attempts}): "
                    f"model={self.config.model}, temp={temp}, max_tokens={max_tok}"
                )

                response = await self.async_client.messages.create(
                    model=self.config.model,
                    messages=messages,
                    temperature=temp,
                    max_tokens=max_tok,
                    timeout=self.config.timeout,
                    **kwargs,
                )

                # Anthropic returns list of content blocks
                text_blocks = [block.text for block in response.content if hasattr(block, "text")]
                if not text_blocks:
                    raise RuntimeError("No text content in Anthropic response")

                text = text_blocks[0]

                # Log token usage
                if hasattr(response, "usage"):
                    logger.debug(
                        f"Anthropic tokens: input={response.usage.input_tokens}, "
                        f"output={response.usage.output_tokens}"
                    )

                return text

            except Exception as e:
                # Determine if error is retryable
                error_msg = str(e).lower()
                is_rate_limit = "rate_limit" in error_msg or "429" in error_msg
                is_timeout = "timeout" in error_msg or "timed out" in error_msg
                is_retriable = is_rate_limit or is_timeout or "500" in error_msg

                if not is_retriable or attempt == self.config.retry_attempts - 1:
                    # Not retryable or last attempt
                    logger.error(f"Anthropic API error: {e}")
                    raise RuntimeError(f"Anthropic API error: {e}") from e

                # Exponential backoff
                delay = self.config.retry_delay * (2 ** attempt)
                logger.warning(
                    f"Anthropic API error (retrying in {delay:.1f}s): {e}"
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
        """Generate a text completion synchronously.

        Args:
            messages: List of message dicts with 'role' and 'content'.
            temperature: Override config temperature.
            max_tokens: Override config max_tokens.
            **kwargs: Additional Anthropic parameters.

        Returns:
            Generated text response.

        Raises:
            ValueError: If message format is invalid.
            RuntimeError: If API call fails after all retries.
        """
        self._validate_messages(messages)

        temp = temperature if temperature is not None else self.config.temperature
        max_tok = max_tokens if max_tokens is not None else self.config.max_tokens

        for attempt in range(self.config.retry_attempts):
            try:
                logger.debug(
                    f"Anthropic completion sync (attempt {attempt + 1}/{self.config.retry_attempts}): "
                    f"model={self.config.model}, temp={temp}, max_tokens={max_tok}"
                )

                response = self.client.messages.create(
                    model=self.config.model,
                    messages=messages,
                    temperature=temp,
                    max_tokens=max_tok,
                    timeout=self.config.timeout,
                    **kwargs,
                )

                # Anthropic returns list of content blocks
                text_blocks = [block.text for block in response.content if hasattr(block, "text")]
                if not text_blocks:
                    raise RuntimeError("No text content in Anthropic response")

                text = text_blocks[0]

                # Log token usage
                if hasattr(response, "usage"):
                    logger.debug(
                        f"Anthropic tokens: input={response.usage.input_tokens}, "
                        f"output={response.usage.output_tokens}"
                    )

                return text

            except Exception as e:
                # Determine if error is retryable
                error_msg = str(e).lower()
                is_rate_limit = "rate_limit" in error_msg or "429" in error_msg
                is_timeout = "timeout" in error_msg or "timed out" in error_msg
                is_retriable = is_rate_limit or is_timeout or "500" in error_msg

                if not is_retriable or attempt == self.config.retry_attempts - 1:
                    # Not retryable or last attempt
                    logger.error(f"Anthropic API error: {e}")
                    raise RuntimeError(f"Anthropic API error: {e}") from e

                # Exponential backoff
                delay = self.config.retry_delay * (2 ** attempt)
                logger.warning(
                    f"Anthropic API error (retrying in {delay:.1f}s): {e}"
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

        Note: Anthropic doesn't have built-in JSON mode, so we parse JSON from text response.

        Args:
            messages: List of message dicts. User message should mention JSON format.
            schema: Optional JSON schema (for documentation).
            temperature: Override config temperature.
            max_tokens: Override config max_tokens.
            **kwargs: Additional Anthropic parameters.

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

        Note: Anthropic doesn't have built-in JSON mode, so we parse JSON from text response.

        Args:
            messages: List of message dicts. User message should mention JSON format.
            schema: Optional JSON schema (for documentation).
            temperature: Override config temperature.
            max_tokens: Override config max_tokens.
            **kwargs: Additional Anthropic parameters.

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
