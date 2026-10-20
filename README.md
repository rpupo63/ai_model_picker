# AI Model Picker

A unified Python **library** (not a token-passing microservice) for AI model provider selection and configuration. Provides a shared foundation for applications that need to:

- Select AI providers (OpenAI, Anthropic, Google, Mistral, etc.)
- Choose models from those providers
- Persist a secrets-free **preference handoff** (`provider`, `model_id`, `instructions`) for sibling services
- Manage API keys locally with environment variable fallback (keys stay with the app / vault / gateway)

## Architecture

```
App / CLI ──► ai-model-picker (library)
                ├─ catalog (OpenRouter + cache)
                ├─ local config + API keys
                └─ preference.json ──handoff──► other service / gateway
                                              (no API keys in payload)
```

**Do share across apps:** live catalog sync, provider→model mapping, display→API IDs, preference payload.

**Do not use this package as:** a microservice that stores or forwards upstream LLM API keys to other services. Pass `ModelPreference` instead; keep secrets at the inference edge.

### Consumer API surface (CLI apps)

Use these exports only; pass a distinct `app_name` per app (e.g. `ai_auto_commit`, `ai-rules-generator`):

| API | Role |
|-----|------|
| `get_available_providers` / `get_provider_models` / `get_model_api_id` | Live catalog |
| `select_provider` / `select_model` / `select_preference` | Interactive UX |
| `get_api_key_with_fallback` / `set_api_key` | Local secrets only |
| `load_preference` / `save_preference` / `to_handoff_dict` | Secrets-free selection + instructions |
| `call_ai` / `call_with_preference` | Optional unified caller |

Do not hardcode provider allowlists in consuming apps — derive them from `get_available_providers()`.

For local development of sibling CLIs:

```bash
pip install -e ../model_picker
# or rebuild AUR packages that vendor the sibling wheel
```

## Installation

```bash
pip install "ai-model-picker>=0.2.0"
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

### Preference handoff (for other services)

```python
from ai_model_picker import (
    build_preference,
    save_preference,
    load_preference,
    to_handoff_dict,
    handoff_json,
    select_preference,
)

# Programmatic
pref = build_preference(
    "anthropic",
    "Claude Sonnet 5",
    instructions="Prefer concise, code-first answers.",
    app_name="my-app",
)
save_preference(pref, app_name="my-app")

# Secrets-free dict/JSON for a sibling service or gateway
payload = to_handoff_dict(pref)
# {"provider": "anthropic", "model": "...", "model_id": "claude-sonnet-5",
#  "instructions": "...", "app_name": "my-app"}

# Interactive pick + optional instructions
pref = select_preference(app_name="my-app")
print(handoff_json(pref))
```

Preference files live next to config: `~/.config/{app_name}/preference.json`.
They never contain API keys.

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
- **Moonshot (Kimi)** - Kimi K2/K3
- **Z.ai (GLM)** - GLM 5.x
- **MiniMax** - M2/M3
- **Perplexity** - Sonar
- **NVIDIA** - Nemotron
- **ByteDance Seed** - Seed 1.6/2.0
- **Tencent (Hunyuan)** - Hy3 / Hunyuan
- **Xiaomi (MiMo)** - MiMo V2.5
- **Amazon (Nova)** - Nova Pro/Lite/Micro
- **StepFun** - Step 3.x

Model catalogs are refreshed from [OpenRouter's unified models API](https://openrouter.ai/api/v1/models) (`GET /api/v1/models`), so deprecated models drop out automatically. Results are cached under `~/.config/{app_name}/openrouter_models_cache.json` (default TTL: 6 hours). If the network is unavailable, the library falls back to the bundled `provider_models.json`.

```python
from ai_model_picker import refresh_provider_models, get_providers_source, get_provider_models

# Force a fresh pull from OpenRouter
refresh_provider_models()
print(get_providers_source())  # "openrouter"
print(get_provider_models("anthropic")[:5])
```

Environment knobs:

| Variable | Effect |
|----------|--------|
| `AI_MODEL_PICKER_OFFLINE=1` | Skip network; use cache / bundled JSON |
| `AI_MODEL_PICKER_CACHE_TTL` | Cache lifetime in seconds (default `21600`) |
| `AI_MODEL_PICKER_OPENROUTER_URL` | Override the models endpoint URL |

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
from ai_model_picker import setup_wizard, load_config, load_preference, to_handoff_dict

# Each app has its own config + preference files
config = setup_wizard(app_name="ai-auto-commit")
config = load_config(app_name="ai-rules-generator")

# Sibling service receives preference only (not API keys)
handoff = to_handoff_dict(load_preference(app_name="ai-rules-generator"))
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

# OpenAI-compatible providers (DeepSeek, xAI, Meta, Moonshot, Z.ai, MiniMax,
# Perplexity, NVIDIA, ByteDance, Tencent, Xiaomi, Amazon, StepFun):
pip install openai
```

Override any OpenAI-compatible base URL with `{PROVIDER}_BASE_URL`
(e.g. `ZAI_BASE_URL`, `MINIMAX_BASE_URL`, `MOONSHOT_BASE_URL`).

## License

MIT
