"""OpenAI LLM client implementation."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any, Optional

from llm.base import BaseLLMClient
from llm.config import LLMConfig

logger = logging.getLogger(__name__)


class OpenAIClient(BaseLLMClient):
    """OpenAI API client with async/sync support, retry logic, and JSON handling.

    Uses the openai Python SDK for all API interactions.
    Implements exponential backoff for retries on rate limits and transient errors.

    Args:
        config: LLMConfig instance with OpenAI settings.

    Raises:
        ValueError: If config is invalid or API key is missing.
    """

    def __init__(self, config: LLMConfig):
        """Initialize OpenAI client.

        Args:
            config: LLMConfig instance.

        Raises:
            ValueError: If OpenAI API key is not available.
        """
        config.validate()
        if not config.resolved_api_key:
            raise ValueError("OpenAI API key required but not found")

        self.config = config
        self.api_key = config.resolved_api_key

        # Import openai here to allow optional dependency
        try:
            import openai
        except ImportError:
            raise ImportError("openai package required. Install with: pip install openai")

        self.client = openai.OpenAI(api_key=self.api_key)
        self.async_client = openai.AsyncOpenAI(api_key=self.api_key)

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
            **kwargs: Additional OpenAI parameters (e.g., top_p, frequency_penalty).

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
                    f"OpenAI completion (attempt {attempt + 1}/{self.config.retry_attempts}): "
                    f"model={self.config.model}, temp={temp}, max_tokens={max_tok}"
                )

                response = await self.async_client.chat.completions.create(
                    model=self.config.model,
                    messages=messages,
                    temperature=temp,
                    max_tokens=max_tok,
                    timeout=self.config.timeout,
                    **kwargs,
                )

                text = response.choices[0].message.content
                if text is None:
                    raise RuntimeError("Empty response from OpenAI API")

                # Log token usage
                if hasattr(response, "usage") and response.usage:
                    logger.debug(
                        f"OpenAI tokens: prompt={response.usage.prompt_tokens}, "
                        f"completion={response.usage.completion_tokens}, "
                        f"total={response.usage.total_tokens}"
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
                    logger.error(f"OpenAI API error: {e}")
                    raise RuntimeError(f"OpenAI API error: {e}") from e

                # Exponential backoff
                delay = self.config.retry_delay * (2 ** attempt)
                logger.warning(
                    f"OpenAI API error (retrying in {delay:.1f}s): {e}"
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
            **kwargs: Additional OpenAI parameters.

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
                    f"OpenAI completion sync (attempt {attempt + 1}/{self.config.retry_attempts}): "
                    f"model={self.config.model}, temp={temp}, max_tokens={max_tok}"
                )

                response = self.client.chat.completions.create(
                    model=self.config.model,
                    messages=messages,
                    temperature=temp,
                    max_tokens=max_tok,
                    timeout=self.config.timeout,
                    **kwargs,
                )

                text = response.choices[0].message.content
                if text is None:
                    raise RuntimeError("Empty response from OpenAI API")

                # Log token usage
                if hasattr(response, "usage") and response.usage:
                    logger.debug(
                        f"OpenAI tokens: prompt={response.usage.prompt_tokens}, "
                        f"completion={response.usage.completion_tokens}, "
                        f"total={response.usage.total_tokens}"
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
                    logger.error(f"OpenAI API error: {e}")
                    raise RuntimeError(f"OpenAI API error: {e}") from e

                # Exponential backoff
                delay = self.config.retry_delay * (2 ** attempt)
                logger.warning(
                    f"OpenAI API error (retrying in {delay:.1f}s): {e}"
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
        """Generate a JSON response asynchronously using OpenAI's JSON mode.

        Uses response_format={"type": "json_object"} for guaranteed JSON output.

        Args:
            messages: List of message dicts. User message should mention JSON format.
            schema: Optional JSON schema (for documentation; validation is provider-specific).
            temperature: Override config temperature.
            max_tokens: Override config max_tokens.
            **kwargs: Additional OpenAI parameters.

        Returns:
            Parsed JSON response as dict.

        Raises:
            ValueError: If response is not valid JSON.
            RuntimeError: If API call fails.
        """
        self._validate_messages(messages)

        # Ensure last message mentions JSON format (best practice for JSON mode)
        if messages and messages[-1]["role"] == "user":
            if "json" not in messages[-1]["content"].lower():
                logger.debug("Note: User message should mention JSON format for best results")

        temp = temperature if temperature is not None else self.config.temperature
        max_tok = max_tokens if max_tokens is not None else self.config.max_tokens

        for attempt in range(self.config.retry_attempts):
            try:
                logger.debug(
                    f"OpenAI JSON completion (attempt {attempt + 1}/{self.config.retry_attempts}): "
                    f"model={self.config.model}"
                )

                response = await self.async_client.chat.completions.create(
                    model=self.config.model,
                    messages=messages,
                    response_format={"type": "json_object"},
                    temperature=temp,
                    max_tokens=max_tok,
                    timeout=self.config.timeout,
                    **kwargs,
                )

                text = response.choices[0].message.content
                if text is None:
                    raise RuntimeError("Empty response from OpenAI API")

                # Log token usage
                if hasattr(response, "usage") and response.usage:
                    logger.debug(
                        f"OpenAI tokens: prompt={response.usage.prompt_tokens}, "
                        f"completion={response.usage.completion_tokens}, "
                        f"total={response.usage.total_tokens}"
                    )

                return self._parse_json_response(text)

            except ValueError as e:
                # JSON parsing error - not retryable
                logger.error(f"JSON parsing error: {e}")
                raise
            except Exception as e:
                # Determine if error is retryable
                error_msg = str(e).lower()
                is_rate_limit = "rate_limit" in error_msg or "429" in error_msg
                is_timeout = "timeout" in error_msg or "timed out" in error_msg
                is_retriable = is_rate_limit or is_timeout or "500" in error_msg

                if not is_retriable or attempt == self.config.retry_attempts - 1:
                    # Not retryable or last attempt
                    logger.error(f"OpenAI API error: {e}")
                    raise RuntimeError(f"OpenAI API error: {e}") from e

                # Exponential backoff
                delay = self.config.retry_delay * (2 ** attempt)
                logger.warning(
                    f"OpenAI API error (retrying in {delay:.1f}s): {e}"
                )
                await asyncio.sleep(delay)

        raise RuntimeError("Max retries exceeded")

    def complete_json_sync(
        self,
        messages: list[dict[str, str]],
        schema: Optional[dict[str, Any]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Generate a JSON response synchronously using OpenAI's JSON mode.

        Uses response_format={"type": "json_object"} for guaranteed JSON output.

        Args:
            messages: List of message dicts. User message should mention JSON format.
            schema: Optional JSON schema (for documentation; validation is provider-specific).
            temperature: Override config temperature.
            max_tokens: Override config max_tokens.
            **kwargs: Additional OpenAI parameters.

        Returns:
            Parsed JSON response as dict.

        Raises:
            ValueError: If response is not valid JSON.
            RuntimeError: If API call fails.
        """
        self._validate_messages(messages)

        # Ensure last message mentions JSON format (best practice for JSON mode)
        if messages and messages[-1]["role"] == "user":
            if "json" not in messages[-1]["content"].lower():
                logger.debug("Note: User message should mention JSON format for best results")

        temp = temperature if temperature is not None else self.config.temperature
        max_tok = max_tokens if max_tokens is not None else self.config.max_tokens

        for attempt in range(self.config.retry_attempts):
            try:
                logger.debug(
                    f"OpenAI JSON completion sync (attempt {attempt + 1}/{self.config.retry_attempts}): "
                    f"model={self.config.model}"
                )

                response = self.client.chat.completions.create(
                    model=self.config.model,
                    messages=messages,
                    response_format={"type": "json_object"},
                    temperature=temp,
                    max_tokens=max_tok,
                    timeout=self.config.timeout,
                    **kwargs,
                )

                text = response.choices[0].message.content
                if text is None:
                    raise RuntimeError("Empty response from OpenAI API")

                # Log token usage
                if hasattr(response, "usage") and response.usage:
                    logger.debug(
                        f"OpenAI tokens: prompt={response.usage.prompt_tokens}, "
                        f"completion={response.usage.completion_tokens}, "
                        f"total={response.usage.total_tokens}"
                    )

                return self._parse_json_response(text)

            except ValueError as e:
                # JSON parsing error - not retryable
                logger.error(f"JSON parsing error: {e}")
                raise
            except Exception as e:
                # Determine if error is retryable
                error_msg = str(e).lower()
                is_rate_limit = "rate_limit" in error_msg or "429" in error_msg
                is_timeout = "timeout" in error_msg or "timed out" in error_msg
                is_retriable = is_rate_limit or is_timeout or "500" in error_msg

                if not is_retriable or attempt == self.config.retry_attempts - 1:
                    # Not retryable or last attempt
                    logger.error(f"OpenAI API error: {e}")
                    raise RuntimeError(f"OpenAI API error: {e}") from e

                # Exponential backoff
                delay = self.config.retry_delay * (2 ** attempt)
                logger.warning(
                    f"OpenAI API error (retrying in {delay:.1f}s): {e}"
                )
                time.sleep(delay)

        raise RuntimeError("Max retries exceeded")
