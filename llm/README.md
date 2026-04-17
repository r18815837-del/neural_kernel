# LLM Provider Abstraction Layer

Production-ready unified interface for multiple LLM providers in neural_kernel.

## Quick Start

```python
from llm import create_llm_client
from llm.prompts import build_parse_requirements_prompt

# Create client (loads config from env)
client = create_llm_client()

# Parse requirements
messages = build_parse_requirements_prompt("Build a todo app in FastAPI")
result = client.complete_json_sync(messages)
print(result)  # Structured requirements
```

## Providers

- **OpenAI** - Primary, production-ready with JSON mode
- **Anthropic** - Claude models
- **Ollama** - Local/offline models

## Configuration

```bash
export OPENAI_API_KEY="sk-..."
export NK_LLM_PROVIDER="openai"
export NK_LLM_MODEL="gpt-4o-mini"
```

## Module Structure

```
llm/
├── __init__.py              # Package exports
├── config.py                # Configuration & validation
├── base.py                  # Abstract interface
├── openai_client.py         # OpenAI implementation
├── anthropic_client.py      # Anthropic implementation
├── ollama_client.py         # Ollama implementation
├── factory.py               # Client factory
├── prompts/
│   ├── __init__.py
│   ├── system_prompts.py    # System prompts for each role
│   └── templates.py         # Message builders
├── USAGE.md                 # Complete usage guide
└── README.md                # This file
```

## Documentation

- **[USAGE.md](USAGE.md)** - Complete usage guide with examples
- **[../LLM_LAYER_SUMMARY.md](../LLM_LAYER_SUMMARY.md)** - Architecture & implementation details
- **[../examples/llm_example.py](../examples/llm_example.py)** - Working examples

## Key Features

- ✓ Unified interface across all providers
- ✓ Async and sync support
- ✓ Native JSON mode (OpenAI)
- ✓ Exponential backoff retries
- ✓ Rate limit handling
- ✓ Token usage logging
- ✓ Configuration validation
- ✓ Message validation
- ✓ Comprehensive error handling
- ✓ Type hints throughout

## API Overview

### Create Client

```python
from llm import LLMConfig, create_llm_client

# From environment
client = create_llm_client()

# Or with config
config = LLMConfig(provider="openai", api_key="sk-...")
client = create_llm_client(config)
```

### Text Generation

```python
messages = [
    {"role": "system", "content": "You are helpful."},
    {"role": "user", "content": "What is 2+2?"}
]

# Sync
response = client.complete_sync(messages)

# Async
response = await client.complete(messages)
```

### JSON Responses

```python
# Sync (OpenAI guarantees valid JSON)
result = client.complete_json_sync(messages)

# Async
result = await client.complete_json(messages)
```

### Prompts

```python
from llm.prompts import (
    build_parse_requirements_prompt,
    build_architect_prompt,
    build_backend_code_prompt,
    build_docs_prompt,
    build_qa_test_prompt,
)

# Use any builder
messages = build_parse_requirements_prompt(raw_text)
result = client.complete_json_sync(messages)
```

## Error Handling

```python
try:
    response = client.complete_sync(messages)
except ValueError as e:
    print(f"Invalid message format: {e}")
except RuntimeError as e:
    print(f"API error after retries: {e}")
```

## Testing

```bash
# Run examples
python examples/llm_example.py

# Test imports
python -c "from llm import create_llm_client; print('OK')"
```

## Environment Variables

```bash
# Required
OPENAI_API_KEY="sk-..."

# Optional (with defaults)
NK_LLM_PROVIDER="openai"
NK_LLM_MODEL="gpt-4o-mini"
NK_LLM_FAST_MODEL="gpt-4o-mini"
NK_LLM_SMART_MODEL="gpt-4o"
NK_LLM_TEMPERATURE="0.3"
NK_LLM_MAX_TOKENS="4096"
NK_LLM_TIMEOUT="60"
NK_LLM_RETRY_ATTEMPTS="3"
```

## Dependencies

```bash
# OpenAI support
pip install openai

# Anthropic support
pip install anthropic

# Ollama support
pip install httpx
```

## Next Steps

1. Set `OPENAI_API_KEY` environment variable
2. Import and use `create_llm_client()`
3. See [USAGE.md](USAGE.md) for detailed examples

## Support

- See [USAGE.md](USAGE.md) for comprehensive guide
- See [../examples/llm_example.py](../examples/llm_example.py) for working examples
- Check error messages for specific troubleshooting
