"""
Unified AI client for multiple providers.

Provides a consistent interface for calling AI models across different providers,
handling provider-specific quirks internally.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Callable

from .config import (
    get_available_providers,
    get_api_key_with_fallback,
    get_model_api_id,
    get_provider_env_var,
)


@dataclass
class AIResponse:
    """Standardized response from AI providers."""
    content: str
    model: str
    provider: str
    usage: Optional[Dict[str, int]] = None  # prompt_tokens, completion_tokens, total_tokens
    raw_response: Optional[Any] = None


@dataclass
class AIClientConfig:
    """Configuration for AI client."""
    temperature: float = 0.3
    max_tokens: int = 4000
    timeout: int = 60
    system_prompt: Optional[str] = None


# Provider-specific quirks and handlers
_PROVIDER_QUIRKS: Dict[str, Dict[str, Any]] = {
    "openai": {
        "supports_timeout": True,
        "supports_system_in_messages": True,
        "response_path": "choices[0].message.content",
    },
    "anthropic": {
        "supports_timeout": True,
        "supports_system_in_messages": False,  # Uses separate system parameter
        "response_path": "content[0].text",
    },
    "google": {
        "supports_timeout": True,
        "supports_system_in_messages": True,
        "response_content_is_list": True,  # May return list of parts
        "response_path": "text",
    },
    "mistral": {
        "supports_timeout": True,
        "supports_system_in_messages": True,
        "response_path": "choices[0].message.content",
    },
    "cohere": {
        "supports_timeout": False,  # Cohere SDK doesn't support timeout
        "supports_system_in_messages": True,
        "response_path": "text",
    },
    "deepseek": {
        "supports_timeout": True,
        "supports_system_in_messages": True,
        "response_path": "choices[0].message.content",
        "base_url": "https://api.deepseek.com",
    },
    "xai": {
        "supports_timeout": True,
        "supports_system_in_messages": True,
        "response_path": "choices[0].message.content",
        "base_url": "https://api.x.ai/v1",
    },
    "meta": {
        "supports_timeout": True,
        "supports_system_in_messages": True,
        "response_path": "choices[0].message.content",
        # Meta models typically accessed via other APIs (together.ai, etc.)
    },
    "alibaba": {
        "supports_timeout": True,
        "supports_system_in_messages": True,
        "response_path": "output.text",
        # Uses DashScope API
    },
}


def _extract_content_from_response(response: Any, provider: str) -> str:
    """Extract text content from provider-specific response format."""
    quirks = _PROVIDER_QUIRKS.get(provider, {})

    try:
        # Handle Google's list-of-parts response
        if quirks.get("response_content_is_list"):
            if hasattr(response, "text"):
                return response.text
            if hasattr(response, "content"):
                content = response.content
                if isinstance(content, list):
                    text_parts = []
                    for part in content:
                        if isinstance(part, str):
                            text_parts.append(part)
                        elif hasattr(part, "text"):
                            text_parts.append(part.text)
                        elif isinstance(part, dict) and "text" in part:
                            text_parts.append(part["text"])
                    return "".join(text_parts)
                return str(content)

        # Standard OpenAI-style response
        if hasattr(response, "choices") and response.choices:
            message = response.choices[0].message
            if hasattr(message, "content"):
                return message.content or ""

        # Anthropic-style response
        if hasattr(response, "content") and isinstance(response.content, list):
            if response.content and hasattr(response.content[0], "text"):
                return response.content[0].text

        # Cohere-style response
        if hasattr(response, "text"):
            return response.text

        # Alibaba DashScope style
        if hasattr(response, "output") and hasattr(response.output, "text"):
            return response.output.text

        # Fallback
        if hasattr(response, "content"):
            return str(response.content)

        return str(response)

    except Exception as e:
        raise RuntimeError(f"Failed to extract content from {provider} response: {e}")


def _call_openai(
    prompt: str,
    model: str,
    api_key: str,
    config: AIClientConfig,
) -> AIResponse:
    """Call OpenAI API."""
    try:
        import openai
    except ImportError:
        raise ImportError("openai package not installed. Install with: pip install openai")

    client = openai.OpenAI(api_key=api_key)

    messages = []
    if config.system_prompt:
        messages.append({"role": "system", "content": config.system_prompt})
    messages.append({"role": "user", "content": prompt})

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        timeout=config.timeout,
    )

    content = response.choices[0].message.content or ""
    usage = None
    if response.usage:
        usage = {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
        }

    return AIResponse(
        content=content.strip(),
        model=model,
        provider="openai",
        usage=usage,
        raw_response=response,
    )


def _call_anthropic(
    prompt: str,
    model: str,
    api_key: str,
    config: AIClientConfig,
) -> AIResponse:
    """Call Anthropic API."""
    try:
        import anthropic
    except ImportError:
        raise ImportError("anthropic package not installed. Install with: pip install anthropic")

    client = anthropic.Anthropic(api_key=api_key)

    kwargs = {
        "model": model,
        "max_tokens": config.max_tokens,
        "temperature": config.temperature,
        "messages": [{"role": "user", "content": prompt}],
    }

    # Anthropic uses separate system parameter
    if config.system_prompt:
        kwargs["system"] = config.system_prompt

    message = client.messages.create(**kwargs)

    content = message.content[0].text if message.content else ""
    usage = None
    if hasattr(message, "usage"):
        usage = {
            "prompt_tokens": message.usage.input_tokens,
            "completion_tokens": message.usage.output_tokens,
            "total_tokens": message.usage.input_tokens + message.usage.output_tokens,
        }

    return AIResponse(
        content=content.strip(),
        model=model,
        provider="anthropic",
        usage=usage,
        raw_response=message,
    )


def _call_google(
    prompt: str,
    model: str,
    api_key: str,
    config: AIClientConfig,
) -> AIResponse:
    """Call Google Gemini API."""
    try:
        import google.generativeai as genai
    except ImportError:
        raise ImportError("google-generativeai package not installed. Install with: pip install google-generativeai")

    genai.configure(api_key=api_key)

    generation_config = genai.GenerationConfig(
        temperature=config.temperature,
        max_output_tokens=config.max_tokens,
    )

    gemini_model = genai.GenerativeModel(
        model_name=model,
        generation_config=generation_config,
        system_instruction=config.system_prompt if config.system_prompt else None,
    )

    response = gemini_model.generate_content(prompt)

    # Handle Google's potentially complex response
    content = _extract_content_from_response(response, "google")

    usage = None
    if hasattr(response, "usage_metadata"):
        um = response.usage_metadata
        usage = {
            "prompt_tokens": getattr(um, "prompt_token_count", 0),
            "completion_tokens": getattr(um, "candidates_token_count", 0),
            "total_tokens": getattr(um, "total_token_count", 0),
        }

    return AIResponse(
        content=content.strip(),
        model=model,
        provider="google",
        usage=usage,
        raw_response=response,
    )


def _call_mistral(
    prompt: str,
    model: str,
    api_key: str,
    config: AIClientConfig,
) -> AIResponse:
    """Call Mistral API."""
    try:
        from mistralai import Mistral
    except ImportError:
        raise ImportError("mistralai package not installed. Install with: pip install mistralai")

    client = Mistral(api_key=api_key)

    messages = []
    if config.system_prompt:
        messages.append({"role": "system", "content": config.system_prompt})
    messages.append({"role": "user", "content": prompt})

    response = client.chat.complete(
        model=model,
        messages=messages,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
    )

    content = response.choices[0].message.content or ""
    usage = None
    if response.usage:
        usage = {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
        }

    return AIResponse(
        content=content.strip(),
        model=model,
        provider="mistral",
        usage=usage,
        raw_response=response,
    )


def _call_cohere(
    prompt: str,
    model: str,
    api_key: str,
    config: AIClientConfig,
) -> AIResponse:
    """Call Cohere API."""
    try:
        import cohere
    except ImportError:
        raise ImportError("cohere package not installed. Install with: pip install cohere")

    client = cohere.ClientV2(api_key=api_key)

    messages = []
    if config.system_prompt:
        messages.append({"role": "system", "content": config.system_prompt})
    messages.append({"role": "user", "content": prompt})

    # Cohere doesn't support timeout in SDK
    response = client.chat(
        model=model,
        messages=messages,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
    )

    content = ""
    if response.message and response.message.content:
        content = response.message.content[0].text if response.message.content else ""

    usage = None
    if response.usage:
        usage = {
            "prompt_tokens": getattr(response.usage, "tokens", {}).get("input_tokens", 0),
            "completion_tokens": getattr(response.usage, "tokens", {}).get("output_tokens", 0),
            "total_tokens": 0,
        }
        if usage["prompt_tokens"] and usage["completion_tokens"]:
            usage["total_tokens"] = usage["prompt_tokens"] + usage["completion_tokens"]

    return AIResponse(
        content=content.strip(),
        model=model,
        provider="cohere",
        usage=usage,
        raw_response=response,
    )


def _call_deepseek(
    prompt: str,
    model: str,
    api_key: str,
    config: AIClientConfig,
) -> AIResponse:
    """Call DeepSeek API (OpenAI-compatible)."""
    try:
        import openai
    except ImportError:
        raise ImportError("openai package not installed. Install with: pip install openai")

    client = openai.OpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com",
    )

    messages = []
    if config.system_prompt:
        messages.append({"role": "system", "content": config.system_prompt})
    messages.append({"role": "user", "content": prompt})

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        timeout=config.timeout,
    )

    content = response.choices[0].message.content or ""
    usage = None
    if response.usage:
        usage = {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
        }

    return AIResponse(
        content=content.strip(),
        model=model,
        provider="deepseek",
        usage=usage,
        raw_response=response,
    )


def _call_xai(
    prompt: str,
    model: str,
    api_key: str,
    config: AIClientConfig,
) -> AIResponse:
    """Call xAI (Grok) API (OpenAI-compatible)."""
    try:
        import openai
    except ImportError:
        raise ImportError("openai package not installed. Install with: pip install openai")

    client = openai.OpenAI(
        api_key=api_key,
        base_url="https://api.x.ai/v1",
    )

    messages = []
    if config.system_prompt:
        messages.append({"role": "system", "content": config.system_prompt})
    messages.append({"role": "user", "content": prompt})

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        timeout=config.timeout,
    )

    content = response.choices[0].message.content or ""
    usage = None
    if response.usage:
        usage = {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
        }

    return AIResponse(
        content=content.strip(),
        model=model,
        provider="xai",
        usage=usage,
        raw_response=response,
    )


def _call_alibaba(
    prompt: str,
    model: str,
    api_key: str,
    config: AIClientConfig,
) -> AIResponse:
    """Call Alibaba DashScope API."""
    try:
        from dashscope import Generation
    except ImportError:
        raise ImportError("dashscope package not installed. Install with: pip install dashscope")

    import dashscope
    dashscope.api_key = api_key

    messages = []
    if config.system_prompt:
        messages.append({"role": "system", "content": config.system_prompt})
    messages.append({"role": "user", "content": prompt})

    response = Generation.call(
        model=model,
        messages=messages,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        result_format="message",
    )

    content = ""
    if response.output and response.output.choices:
        content = response.output.choices[0].message.content or ""

    usage = None
    if response.usage:
        usage = {
            "prompt_tokens": response.usage.input_tokens,
            "completion_tokens": response.usage.output_tokens,
            "total_tokens": response.usage.total_tokens,
        }

    return AIResponse(
        content=content.strip(),
        model=model,
        provider="alibaba",
        usage=usage,
        raw_response=response,
    )


def _call_meta(
    prompt: str,
    model: str,
    api_key: str,
    config: AIClientConfig,
) -> AIResponse:
    """
    Call Meta Llama API.

    Note: Meta models are typically accessed via third-party APIs like
    Together.ai, Replicate, or cloud providers. This implementation
    uses the Together.ai API as it's the most common.
    """
    try:
        import openai
    except ImportError:
        raise ImportError("openai package not installed. Install with: pip install openai")

    # Use Together.ai as default endpoint for Meta models
    client = openai.OpenAI(
        api_key=api_key,
        base_url="https://api.together.xyz/v1",
    )

    messages = []
    if config.system_prompt:
        messages.append({"role": "system", "content": config.system_prompt})
    messages.append({"role": "user", "content": prompt})

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        timeout=config.timeout,
    )

    content = response.choices[0].message.content or ""
    usage = None
    if response.usage:
        usage = {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
        }

    return AIResponse(
        content=content.strip(),
        model=model,
        provider="meta",
        usage=usage,
        raw_response=response,
    )


# Provider handler registry
_PROVIDER_HANDLERS: Dict[str, Callable] = {
    "openai": _call_openai,
    "anthropic": _call_anthropic,
    "google": _call_google,
    "mistral": _call_mistral,
    "cohere": _call_cohere,
    "deepseek": _call_deepseek,
    "xai": _call_xai,
    "alibaba": _call_alibaba,
    "meta": _call_meta,
}


def call_ai(
    prompt: str,
    provider: str = "openai",
    model: str = "gpt-4o-mini",
    api_key: Optional[str] = None,
    app_name: str = "ai-model-picker",
    temperature: float = 0.3,
    max_tokens: int = 4000,
    timeout: int = 60,
    system_prompt: Optional[str] = None,
) -> Optional[AIResponse]:
    """
    Call an AI provider with a prompt and return the response.

    This is the main entry point for making AI API calls. It handles:
    - Model name to API ID resolution
    - API key retrieval (config file -> environment variable)
    - Provider-specific API calls
    - Response normalization

    Parameters
    ----------
    prompt : str
        The prompt to send to the AI.
    provider : str
        The provider key (openai, anthropic, google, etc.).
    model : str
        The model name (display name or API ID).
    api_key : str | None
        Optional explicit API key. If not provided, will be loaded from
        config file or environment variable.
    app_name : str
        Application name for config file lookup.
    temperature : float
        Sampling temperature (0.0 to 1.0).
    max_tokens : int
        Maximum tokens in response.
    timeout : int
        Request timeout in seconds.
    system_prompt : str | None
        Optional system prompt to set context.

    Returns
    -------
    AIResponse | None
        The AI response, or None if provider is "none".

    Raises
    ------
    ValueError
        If provider is unknown.
    RuntimeError
        If API key is missing or API call fails.
    ImportError
        If required SDK is not installed.
    """
    # Handle "none" provider
    if provider == "none":
        return None

    # Check if provider is supported
    if provider not in _PROVIDER_HANDLERS:
        raise ValueError(
            f"Unknown provider: {provider}. "
            f"Supported providers: {', '.join(_PROVIDER_HANDLERS.keys())}"
        )

    # Resolve model display name to API ID
    api_model = get_model_api_id(model, provider, app_name)

    # Get API key
    key = api_key or get_api_key_with_fallback(provider, app_name)
    if not key:
        env_var = get_provider_env_var(provider) or f"{provider.upper()}_API_KEY"
        raise RuntimeError(
            f"{provider.title()} API key not found. "
            f"Set {env_var} environment variable or run setup."
        )

    # Create config
    config = AIClientConfig(
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
        system_prompt=system_prompt,
    )

    # Call provider
    handler = _PROVIDER_HANDLERS[provider]
    return handler(prompt, api_model, key, config)


def call_ai_simple(
    prompt: str,
    provider: str = "openai",
    model: str = "gpt-4o-mini",
    api_key: Optional[str] = None,
    app_name: str = "ai-model-picker",
    system_prompt: Optional[str] = None,
) -> Optional[str]:
    """
    Simple wrapper that returns just the response text.

    This is a convenience function for when you just need the text content.

    Parameters
    ----------
    prompt : str
        The prompt to send to the AI.
    provider : str
        The provider key.
    model : str
        The model name.
    api_key : str | None
        Optional explicit API key.
    app_name : str
        Application name for config lookup.
    system_prompt : str | None
        Optional system prompt.

    Returns
    -------
    str | None
        The response text, or None if provider is "none" or call fails.
    """
    try:
        response = call_ai(
            prompt=prompt,
            provider=provider,
            model=model,
            api_key=api_key,
            app_name=app_name,
            system_prompt=system_prompt,
        )
        return response.content if response else None
    except Exception as e:
        print(f"Error calling {provider} API: {e}", file=sys.stderr)
        return None


def get_supported_providers() -> List[str]:
    """Get list of providers with implemented handlers."""
    return list(_PROVIDER_HANDLERS.keys())


def check_provider_available(provider: str) -> tuple[bool, Optional[str]]:
    """
    Check if a provider's SDK is installed.

    Returns
    -------
    tuple[bool, str | None]
        (is_available, error_message)
    """
    if provider == "none":
        return True, None

    if provider not in _PROVIDER_HANDLERS:
        return False, f"Unknown provider: {provider}"

    try:
        if provider == "openai":
            import openai
        elif provider == "anthropic":
            import anthropic
        elif provider == "google":
            import google.generativeai
        elif provider == "mistral":
            import mistralai
        elif provider == "cohere":
            import cohere
        elif provider in ("deepseek", "xai", "meta"):
            import openai  # Uses OpenAI-compatible API
        elif provider == "alibaba":
            import dashscope
        return True, None
    except ImportError as e:
        return False, str(e)
