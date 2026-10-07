# AI Model Picker

A unified Python library for AI model provider selection and configuration. Provides a shared foundation for applications that need to:

- Select AI providers (OpenAI, Anthropic, Google, Mistral, etc.)
- Choose models from those providers
- Manage API keys with environment variable fallback
- Persist configuration across sessions

## Installation

```bash
pip install ai-model-picker
```

Or install from source:

```bash
pip install -e /path/to/ai_model_picker
```

## Quick Start

```python
from ai_model_picker import (
    setup_wizard,
    select_provider_and_model,
    get_api_key_with_fallback,
    get_model_api_id,
)

# Run interactive setup
config = setup_wizard(app_name="my-app")

# Or select provider/model programmatically
provider, model = select_provider_and_model()

# Get API key (checks config, then env var)
api_key = get_api_key_with_fallback(provider, app_name="my-app")

# Convert display name to API ID
model_id = get_model_api_id(model, provider)
```

## Features

### Supported Providers

- **OpenAI** - GPT-4o, GPT-5.x, o3
- **Anthropic** - Claude 3.x, 4.x
- **Google** - Gemini 2.x, 3.x
- **Mistral** - Mistral Large, Devstral
- **Cohere** - Command R/R+
- **Meta** - Llama 4
- **DeepSeek** - R1, V3.x
- **xAI** - Grok 4.x
- **Alibaba** - Qwen3

### Configuration

Configuration is stored in platform-appropriate locations:

- **Linux**: `~/.config/{app_name}/config.json`
- **macOS**: `~/Library/Application Support/{app_name}/config.json`
- **Windows**: `%APPDATA%/{app_name}/config.json`

### API Key Management

```python
from ai_model_picker import (
    set_api_key,
    get_api_key,
    get_api_key_with_fallback,
    get_all_api_keys,
)

# Store a key
set_api_key("openai", "sk-...", app_name="my-app")

# Get key from config only
key = get_api_key("openai", app_name="my-app")

# Get key with env var fallback (checks OPENAI_API_KEY)
key = get_api_key_with_fallback("openai", app_name="my-app")

# Get all stored keys
keys = get_all_api_keys(app_name="my-app")
```

### Interactive Selection

```python
from ai_model_picker import select_provider, select_model

# Select provider interactively
provider = select_provider()

# Select model for that provider
model = select_model(provider)
```

### Model ID Mapping

Display names are mapped to API IDs automatically:

```python
from ai_model_picker import get_model_api_id

# "Claude 4.5 Sonnet" -> "claude-sonnet-4-5-20250514"
api_id = get_model_api_id("Claude 4.5 Sonnet", "anthropic")
```

### AI Client

Make AI API calls with a unified interface:

```python
from ai_model_picker import call_ai, call_ai_simple, AIResponse

# Simple usage - returns just the text
response = call_ai_simple(
    prompt="Write a hello world function in Python",
    provider="openai",
    model="GPT-4o mini",
    app_name="my-app",
)

# Full response with metadata
result: AIResponse = call_ai(
    prompt="Explain recursion",
    provider="anthropic",
    model="Claude 4.5 Sonnet",
    system_prompt="You are a helpful programming tutor.",
    temperature=0.3,
    max_tokens=1000,
)
print(result.content)
print(result.usage)  # Token usage stats
```

### Provider-Specific Quirks Handled

The unified client handles provider differences internally:

| Provider | Quirk | How It's Handled |
|----------|-------|------------------|
| **Google** | Returns content as list of parts | Automatically joined into string |
| **Cohere** | Doesn't support timeout parameter | Timeout omitted |
| **Anthropic** | System prompt is separate parameter | Handled transparently |
| **DeepSeek, xAI, Meta** | OpenAI-compatible APIs | Correct base URLs configured |
| **Alibaba** | DashScope API differences | Response structure normalized |

### Check Provider Availability

```python
from ai_model_picker import check_provider_available

available, error = check_provider_available("anthropic")
if not available:
    print(f"Anthropic not available: {error}")
```

## Integration

For applications using this library, specify a custom `app_name` to isolate configuration:

```python
from ai_model_picker import setup_wizard, load_config

# Each app has its own config file
config = setup_wizard(app_name="ai-auto-commit")
config = load_config(app_name="ai-rules-generator")
```

## Optional Dependencies

Install provider SDKs as needed:

```bash
# Core providers
pip install openai anthropic

# Additional providers
pip install google-generativeai  # Google Gemini
pip install mistralai            # Mistral
pip install cohere               # Cohere
pip install dashscope            # Alibaba Qwen

# DeepSeek, xAI, Meta use OpenAI-compatible APIs (just need openai package)
```

## License

MIT
