#!/usr/bin/env python3
"""Example usage of the LLM provider abstraction layer.

This file demonstrates how to use the llm package for various tasks.
Run with: python examples/llm_example.py

Note: Requires OPENAI_API_KEY to be set for real API calls.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def example_1_basic_config():
    """Example 1: Basic configuration."""
    logger.info("=" * 60)
    logger.info("Example 1: Basic Configuration")
    logger.info("=" * 60)

    from llm import LLMConfig

    # Create config from environment
    config = LLMConfig.from_env()
    logger.info(f"Config loaded: provider={config.provider}, model={config.model}")

    # Or create with explicit values
    config = LLMConfig(
        provider="openai",
        model="gpt-4o-mini",
        temperature=0.3,
        max_tokens=1024,
    )
    logger.info(f"Custom config: {config}")

    # Validate before use
    try:
        config.validate()
        logger.info("Configuration is valid")
    except ValueError as e:
        logger.error(f"Invalid config: {e}")

    print()


def example_2_create_client():
    """Example 2: Creating LLM clients."""
    logger.info("=" * 60)
    logger.info("Example 2: Creating LLM Clients")
    logger.info("=" * 60)

    from llm import LLMConfig, create_llm_client

    # Create with default config (from env)
    try:
        client = create_llm_client()
        logger.info(f"Created client: {type(client).__name__}")
    except Exception as e:
        logger.warning(f"Cannot create OpenAI client (no API key): {e}")

    # Create Ollama client (no API key needed, but requires Ollama running)
    try:
        config = LLMConfig(provider="ollama", model="mistral")
        client = create_llm_client(config)
        logger.info(f"Created Ollama client: {type(client).__name__}")
    except Exception as e:
        logger.warning(f"Cannot create Ollama client (not running?): {e}")

    print()


def example_3_message_validation():
    """Example 3: Message format validation."""
    logger.info("=" * 60)
    logger.info("Example 3: Message Validation")
    logger.info("=" * 60)

    from llm.base import BaseLLMClient

    # Create a minimal test implementation for validation testing
    class TestClient(BaseLLMClient):
        async def complete(self, messages, **kwargs):
            return ""

        def complete_sync(self, messages, **kwargs):
            return ""

        async def complete_json(self, messages, **kwargs):
            return {}

        def complete_json_sync(self, messages, **kwargs):
            return {}

    # Valid messages
    valid_messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "What is 2+2?"},
    ]

    client = TestClient()
    try:
        client._validate_messages(valid_messages)
        logger.info("Valid messages passed validation")
    except ValueError as e:
        logger.error(f"Validation failed: {e}")

    # Invalid messages - empty
    try:
        client._validate_messages([])
    except ValueError as e:
        logger.info(f"Empty list rejected: {e}")

    # Invalid messages - missing role
    try:
        client._validate_messages([{"content": "test"}])
    except ValueError as e:
        logger.info(f"Missing role rejected: {e}")

    # Invalid messages - bad role
    try:
        client._validate_messages([{"role": "invalid", "content": "test"}])
    except ValueError as e:
        logger.info(f"Invalid role rejected: {e}")

    print()


def example_4_json_parsing():
    """Example 4: JSON response parsing."""
    logger.info("=" * 60)
    logger.info("Example 4: JSON Parsing")
    logger.info("=" * 60)

    from llm.base import BaseLLMClient

    # Create a minimal test implementation
    class TestClient(BaseLLMClient):
        async def complete(self, messages, **kwargs):
            return ""

        def complete_sync(self, messages, **kwargs):
            return ""

        async def complete_json(self, messages, **kwargs):
            return {}

        def complete_json_sync(self, messages, **kwargs):
            return {}

    client = TestClient()

    # Parse plain JSON
    json_text = '{"name": "Alice", "age": 30}'
    result = client._parse_json_response(json_text)
    logger.info(f"Parsed JSON: {result}")

    # Parse markdown-wrapped JSON
    wrapped = '```json\n{"status": "success", "data": [1, 2, 3]}\n```'
    result = client._parse_json_response(wrapped)
    logger.info(f"Parsed wrapped JSON: {result}")

    # Invalid JSON
    try:
        client._parse_json_response("{invalid json}")
    except ValueError as e:
        logger.info(f"Invalid JSON rejected: {str(e)[:50]}...")

    print()


def example_5_prompts():
    """Example 5: Using prompt builders."""
    logger.info("=" * 60)
    logger.info("Example 5: Prompt Builders")
    logger.info("=" * 60)

    from llm.prompts import (
        build_parse_requirements_prompt,
        build_architect_prompt,
        build_backend_code_prompt,
        build_docs_prompt,
        build_qa_test_prompt,
    )

    # Parse requirements
    messages = build_parse_requirements_prompt(
        "Build a todo app in Python with FastAPI and PostgreSQL"
    )
    logger.info(f"Requirements prompt: {len(messages)} messages")
    logger.info(f"  - System role: {messages[0]['role']}")
    logger.info(f"  - User role: {messages[1]['role']}")

    # Architect
    spec = {
        "project_name": "TodoApp",
        "features": [{"name": "Create Todo"}, {"name": "List Todos"}],
        "tech_stack": {"backend": "FastAPI", "database": "PostgreSQL"},
    }
    messages = build_architect_prompt(spec)
    logger.info(f"Architect prompt: {len(messages)} messages")

    # Backend code
    messages = build_backend_code_prompt("User Authentication", spec)
    logger.info(f"Backend code prompt: {len(messages)} messages")

    # Docs
    messages = build_docs_prompt(spec, spec["features"])
    logger.info(f"Docs prompt: {len(messages)} messages")

    # QA
    messages = build_qa_test_prompt(spec, spec["features"])
    logger.info(f"QA prompt: {len(messages)} messages")

    print()


def example_6_message_structure():
    """Example 6: Building conversation messages."""
    logger.info("=" * 60)
    logger.info("Example 6: Message Structure")
    logger.info("=" * 60)

    # Basic conversation
    messages = [
        {"role": "system", "content": "You are a helpful Python expert."},
        {"role": "user", "content": "How do I read a file in Python?"},
    ]

    logger.info("Basic conversation:")
    for msg in messages:
        logger.info(f"  {msg['role']}: {msg['content'][:50]}...")

    # Multi-turn conversation
    messages.append(
        {
            "role": "assistant",
            "content": "Use open() function: with open('file.txt') as f: content = f.read()",
        }
    )
    messages.append(
        {
            "role": "user",
            "content": "How do I handle errors?",
        }
    )

    logger.info(f"\nMulti-turn conversation: {len(messages)} messages")

    print()


def example_7_config_env_vars():
    """Example 7: Loading config from environment variables."""
    logger.info("=" * 60)
    logger.info("Example 7: Environment Variables")
    logger.info("=" * 60)

    import os

    # Show env var options
    env_vars = [
        "OPENAI_API_KEY",
        "NK_LLM_PROVIDER",
        "NK_LLM_MODEL",
        "NK_LLM_FAST_MODEL",
        "NK_LLM_SMART_MODEL",
        "NK_LLM_TEMPERATURE",
        "NK_LLM_MAX_TOKENS",
        "NK_LLM_TIMEOUT",
        "NK_LLM_RETRY_ATTEMPTS",
    ]

    logger.info("Supported environment variables:")
    for var in env_vars:
        value = os.environ.get(var, "(not set)")
        logger.info(f"  {var}: {value}")

    print()


def example_8_provider_comparison():
    """Example 8: Comparing different providers."""
    logger.info("=" * 60)
    logger.info("Example 8: Provider Comparison")
    logger.info("=" * 60)

    from llm import LLMConfig

    providers = {
        "openai": {
            "config": LLMConfig(provider="openai", api_key="sk-test"),
            "pros": ["Native JSON mode", "Fast", "Reliable"],
            "cons": ["Costs money", "Requires API key"],
        },
        "anthropic": {
            "config": LLMConfig(provider="anthropic", api_key="sk-ant-test"),
            "pros": ["Strong reasoning", "Large context", "Reliable"],
            "cons": ["Costs money", "Requires API key"],
        },
        "ollama": {
            "config": LLMConfig(provider="ollama", model="mistral"),
            "pros": ["Free", "Private", "Offline", "No API key"],
            "cons": ["Slower", "Requires Ollama running", "Limited models"],
        },
    }

    for name, info in providers.items():
        logger.info(f"\n{name.upper()}:")
        logger.info(f"  Pros: {', '.join(info['pros'])}")
        logger.info(f"  Cons: {', '.join(info['cons'])}")

    print()


def example_9_error_handling():
    """Example 9: Error handling patterns."""
    logger.info("=" * 60)
    logger.info("Example 9: Error Handling")
    logger.info("=" * 60)

    from llm import LLMConfig

    # Missing API key
    logger.info("Scenario 1: Missing API key")
    try:
        config = LLMConfig(provider="openai")
        config.validate()
    except ValueError as e:
        logger.info(f"  Error caught: {e}")

    # Invalid provider
    logger.info("\nScenario 2: Invalid provider")
    try:
        config = LLMConfig(provider="unknown")
        config.validate()
    except ValueError as e:
        logger.info(f"  Error caught: {e}")

    # Invalid temperature
    logger.info("\nScenario 3: Invalid temperature")
    try:
        config = LLMConfig(temperature=5.0)  # Out of range
        config.validate()
    except ValueError as e:
        logger.info(f"  Error caught: {e}")

    # Invalid message format
    logger.info("\nScenario 4: Invalid message format")
    from llm.base import BaseLLMClient

    class TestClient(BaseLLMClient):
        async def complete(self, messages, **kwargs):
            return ""

        def complete_sync(self, messages, **kwargs):
            return ""

        async def complete_json(self, messages, **kwargs):
            return {}

        def complete_json_sync(self, messages, **kwargs):
            return {}

    client = TestClient()
    try:
        client._validate_messages([{"content": "missing role"}])
    except ValueError as e:
        logger.info(f"  Error caught: {e}")

    print()


def example_10_real_world_workflow():
    """Example 10: Real-world workflow simulation."""
    logger.info("=" * 60)
    logger.info("Example 10: Real-World Workflow")
    logger.info("=" * 60)

    from llm import LLMConfig, create_llm_client
    from llm.prompts import build_parse_requirements_prompt

    # Step 1: Parse requirements
    logger.info("\nStep 1: Parse user requirements")
    user_request = """
    I need a simple blog application with:
    - User authentication (login/signup)
    - Create and edit blog posts
    - Comment on posts
    - Search functionality
    Built with Python FastAPI backend and React frontend
    """

    messages = build_parse_requirements_prompt(user_request)
    logger.info(f"  Built {len(messages)} prompt messages")
    logger.info(f"  System prompt length: {len(messages[0]['content'])} chars")
    logger.info(f"  User message length: {len(messages[1]['content'])} chars")

    # Step 2: Show what would be sent to LLM
    logger.info("\nStep 2: Would send to LLM:")
    logger.info(f"  Role: {messages[0]['role']}")
    logger.info(f"  Content preview: {messages[0]['content'][:100]}...")

    logger.info(f"\n  Role: {messages[1]['role']}")
    logger.info(f"  Content preview: {messages[1]['content'][:100]}...")

    # Step 3: Simulate response
    logger.info("\nStep 3: LLM would return structured JSON with:")
    logger.info("  - Project name and description")
    logger.info("  - Functional requirements (FR-001, FR-002, ...)")
    logger.info("  - Non-functional requirements (NFR-001, ...)")
    logger.info("  - Technical constraints")
    logger.info("  - Dependencies")
    logger.info("  - Any ambiguities to clarify")

    # Step 4: Show next steps
    logger.info("\nStep 4: Next steps would be:")
    logger.info("  1. Design architecture (build_architect_prompt)")
    logger.info("  2. Generate backend code (build_backend_code_prompt)")
    logger.info("  3. Create documentation (build_docs_prompt)")
    logger.info("  4. Plan QA testing (build_qa_test_prompt)")

    print()


def main():
    """Run all examples."""
    logger.info("\n\n")
    logger.info("LLM ABSTRACTION LAYER - EXAMPLES")
    logger.info("=" * 60)
    logger.info("")

    # Run examples
    example_1_basic_config()
    example_2_create_client()
    example_3_message_validation()
    example_4_json_parsing()
    example_5_prompts()
    example_6_message_structure()
    example_7_config_env_vars()
    example_8_provider_comparison()
    example_9_error_handling()
    example_10_real_world_workflow()

    logger.info("=" * 60)
    logger.info("All examples finished")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()