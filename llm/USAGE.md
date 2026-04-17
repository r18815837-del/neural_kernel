# LLM Provider Abstraction Layer - Usage Guide

## Overview

The `llm` package provides a unified interface for working with multiple LLM providers:
- **OpenAI** (primary, with JSON mode support)
- **Anthropic** (Claude, coming soon)
- **Ollama** (local models for development)

All clients support both **synchronous and asynchronous** operations.

## Quick Start

### Basic Text Generation

```python
from llm import create_llm_client, LLMConfig

# Create client from config
config = LLMConfig(provider="openai", api_key="sk-...")
client = create_llm_client(config)

# Synchronous text generation
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is 2+2?"}
]
response = client.complete_sync(messages)
print(response)

# Asynchronous text generation
import asyncio
async def main():
    response = await client.complete(messages)
    print(response)

asyncio.run(main())
```

### JSON Mode (OpenAI)

```python
from llm import create_llm_client

client = create_llm_client()  # Loads from env vars

messages = [
    {"role": "system", "content": "You are a JSON expert. Return valid JSON only."},
    {"role": "user", "content": "Extract key information as JSON: ..."}
]

# Synchronous JSON response
result = client.complete_json_sync(messages)
print(result)  # Already parsed dict

# Asynchronous JSON response
import asyncio
async def main():
    result = await client.complete_json(messages)
    print(result)

asyncio.run(main())
```

## Configuration

### Environment Variables

```bash
# Required
export OPENAI_API_KEY="sk-..."

# Optional (with defaults)
export NK_LLM_PROVIDER="openai"              # Default: openai
export NK_LLM_MODEL="gpt-4o-mini"           # Default: gpt-4o-mini
export NK_LLM_FAST_MODEL="gpt-4o-mini"      # For simple tasks
export NK_LLM_SMART_MODEL="gpt-4o"          # For complex tasks
export NK_LLM_TEMPERATURE="0.3"             # Default: 0.3
export NK_LLM_MAX_TOKENS="4096"             # Default: 4096
export NK_LLM_TIMEOUT="60"                  # Default: 60
export NK_LLM_RETRY_ATTEMPTS="3"            # Default: 3
```

### Load Config from Environment

```python
from llm import LLMConfig

# Automatically loads from env vars
config = LLMConfig.from_env()
```

### Create Custom Config

```python
from llm import LLMConfig

config = LLMConfig(
    provider="openai",
    model="gpt-4o",
    temperature=0.7,
    max_tokens=2048,
    api_key="sk-..."
)

config.validate()  # Raises error if invalid
```

## Providers

### OpenAI

```python
from llm import LLMConfig, create_llm_client

config = LLMConfig(provider="openai", api_key="sk-...")
client = create_llm_client(config)

# Uses gpt-4o-mini by default (fast, cheap)
# Override with:
config = LLMConfig(provider="openai", model="gpt-4o")
```

Features:
- Native JSON mode (`response_format={"type": "json_object"}`)
- Exponential backoff on rate limits
- Token usage logging
- Both sync and async

### Anthropic (Claude)

```python
from llm import LLMConfig, create_llm_client

config = LLMConfig(provider="anthropic", api_key="sk-ant-...")
client = create_llm_client(config)
```

Features:
- Content block handling (text, images, etc.)
- Works with latest Claude models
- Exponential backoff on rate limits
- Both sync and async

### Ollama (Local)

```python
from llm import LLMConfig, create_llm_client

# Requires Ollama running at localhost:11434
config = LLMConfig(provider="ollama", model="mistral")
client = create_llm_client(config)

# Or custom Ollama endpoint
config = LLMConfig(
    provider="ollama",
    model="neural-chat",
    base_url="http://ollama.example.com:11434"
)
```

Features:
- Free, offline, private
- No API key needed
- Great for development
- Connection retry with backoff

## Prompts & Templates

The `llm.prompts` module provides system prompts and message builders for different roles:

### Requirement Parser

```python
from llm.prompts import build_parse_requirements_prompt
from llm import create_llm_client

client = create_llm_client()

raw_request = "Build a todo app with user authentication and due dates..."
messages = build_parse_requirements_prompt(raw_request)

result = client.complete_json_sync(messages)
print(result)  # Structured requirements
```

Returns structured requirements with:
- Functional requirements
- Non-functional requirements
- Technical constraints
- Dependencies
- Ambiguities

### Architect

```python
from llm.prompts import build_architect_prompt

project = {
    "project_name": "TodoApp",
    "features": [{"name": "Create Todo"}, {"name": "List Todos"}],
    "tech_stack": {"backend": "FastAPI", "database": "PostgreSQL"}
}

messages = build_architect_prompt(project)
result = client.complete_json_sync(messages)
print(result)  # Architecture design
```

Returns architecture with:
- Architecture pattern
- Components and data flow
- Technology recommendations
- Deployment strategy
- Security design

### Backend Code Generator

```python
from llm.prompts import build_backend_code_prompt

project = {
    "project_name": "TodoApp",
    "tech_stack": {"backend": "FastAPI", "database": "PostgreSQL"}
}

messages = build_backend_code_prompt("User Authentication", project)
result = client.complete_json_sync(messages)
print(result)  # Generated code files
```

### Documentation Generator

```python
from llm.prompts import build_docs_prompt

messages = build_docs_prompt(project_spec, features)
result = client.complete_json_sync(messages)
print(result)  # README, API docs, guides
```

### QA Test Plan

```python
from llm.prompts import build_qa_test_prompt

messages = build_qa_test_prompt(project_spec, features)
result = client.complete_json_sync(messages)
print(result)  # Test cases, test strategy
```

## Error Handling

All clients implement exponential backoff retry logic:

```python
from llm import create_llm_client, LLMConfig

config = LLMConfig(
    provider="openai",
    retry_attempts=3,      # Retry 3 times
    retry_delay=1.0        # Start with 1 second delay
)
client = create_llm_client(config)

try:
    response = client.complete_sync(messages)
except ValueError as e:
    print(f"Invalid message format: {e}")
except RuntimeError as e:
    print(f"API error after retries: {e}")
```

Retryable errors (auto-retried with backoff):
- Rate limits (429)
- Timeouts
- 5xx server errors

Non-retryable errors (raised immediately):
- Invalid API key
- Invalid message format
- 4xx client errors (except 429)

## Message Format

All clients use OpenAI-style message format:

```python
messages = [
    {
        "role": "system",
        "content": "You are an expert Python developer..."
    },
    {
        "role": "user",
        "content": "Write a function that..."
    },
    {
        "role": "assistant",
        "content": "Here's the function..."
    },
    {
        "role": "user",
        "content": "Can you add error handling?"
    }
]

response = client.complete_sync(messages)
```

Valid roles: `system`, `user`, `assistant`

## Advanced Features

### Override Parameters Per Request

```python
client.complete_sync(
    messages,
    temperature=0.9,      # Override config
    max_tokens=1000       # Override config
)

client.complete_json_sync(
    messages,
    schema={"type": "object", "properties": {...}}  # Optional schema hint
)
```

### Logging

Enable logging to see retry attempts, token usage, etc.:

```python
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("llm.openai_client")

# Now you'll see debug messages for all API calls
```

### Using Different Models for Different Tasks

```python
from llm import LLMConfig, create_llm_client

config = LLMConfig()

# Fast model for simple tasks
fast_config = LLMConfig(model=config.fast_model)
fast_client = create_llm_client(fast_config)

# Smart model for complex tasks
smart_config = LLMConfig(model=config.smart_model)
smart_client = create_llm_client(smart_config)
```

## Testing

```python
from llm import LLMConfig
import os

# Test with mock config (no real API calls)
config = LLMConfig(provider="ollama", model="test")
config.validate()  # Should pass

# Test message validation
from llm.base import BaseLLMClient
client = BaseLLMClient()  # Just for testing
try:
    client._validate_messages([])  # Empty messages
except ValueError as e:
    print(f"Expected error: {e}")

# Test JSON parsing
json_text = '{"name": "test", "value": 42}'
result = client._parse_json_response(json_text)
assert result == {"name": "test", "value": 42}

# Test markdown-wrapped JSON
wrapped = "```json\n{\"key\": \"value\"}\n```"
result = client._parse_json_response(wrapped)
assert result == {"key": "value"}
```

## API Reference

### LLMConfig

```python
@dataclass
class LLMConfig:
    provider: str = "openai"
    model: str = "gpt-4o-mini"
    api_key: str | None = None
    base_url: str | None = None
    temperature: float = 0.3
    max_tokens: int = 4096
    timeout: int = 60
    retry_attempts: int = 3
    retry_delay: float = 1.0
    fast_model: str = "gpt-4o-mini"
    smart_model: str = "gpt-4o"
    
    def validate(self) -> None: ...
    
    @classmethod
    def from_env(cls) -> LLMConfig: ...
```

### BaseLLMClient Interface

```python
class BaseLLMClient(ABC):
    async def complete(
        self,
        messages: list[dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str: ...
    
    def complete_sync(
        self,
        messages: list[dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str: ...
    
    async def complete_json(
        self,
        messages: list[dict[str, str]],
        schema: Optional[dict] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> dict: ...
    
    def complete_json_sync(
        self,
        messages: list[dict[str, str]],
        schema: Optional[dict] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> dict: ...
```

### create_llm_client

```python
def create_llm_client(config: Optional[LLMConfig] = None) -> BaseLLMClient:
    """Factory function to create appropriate LLM client."""
```

## Best Practices

1. **Always validate config** before creating client:
   ```python
   config.validate()
   ```

2. **Use environment variables** for API keys:
   ```python
   export OPENAI_API_KEY="sk-..."
   config = LLMConfig.from_env()
   ```

3. **Reuse client instances** across requests (they're thread-safe):
   ```python
   client = create_llm_client()
   # Use same client for multiple requests
   ```

4. **Use appropriate models** for task complexity:
   ```python
   # For parsing, classification, extraction → fast_model
   # For architecture, code generation → smart_model
   ```

5. **Always handle RuntimeError** from API calls:
   ```python
   try:
       response = client.complete_sync(messages)
   except RuntimeError as e:
       # Handle API error after retries exhausted
       pass
   ```

6. **Use JSON mode for structured output**:
   ```python
   # Guarantees valid JSON from OpenAI
   result = client.complete_json_sync(messages)
   ```

7. **Log important operations**:
   ```python
   import logging
   logging.basicConfig(level=logging.INFO)
   # Info level shows model selection, errors
   # Debug level shows retry attempts, token usage
   ```

## Troubleshooting

**ImportError: openai package required**
```bash
pip install openai
```

**API key not found**
```bash
export OPENAI_API_KEY="sk-..."
```

**Connection refused (Ollama)**
```bash
# Start Ollama first
ollama serve
```

**Rate limit errors**
- Increase `retry_attempts`
- Increase `retry_delay`
- Wait before retrying

**Invalid JSON in response (Anthropic/Ollama)**
- Ensure prompt mentions JSON format
- Use explicit schema in system prompt
- Increase `max_tokens` for complex structures

## Contributing

Add new providers by:
1. Create `new_provider_client.py` inheriting `BaseLLMClient`
2. Implement all 4 abstract methods
3. Update `factory.py` to handle new provider
4. Add tests

See `openai_client.py` for reference implementation.
