"""
Configuration management for AI Model Picker.

Handles loading/saving user preferences, API keys, and provider/model definitions.
Provider and model lists are loaded from provider_models.json for easy updates.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional, List

from .types import Provider, ProviderInfo, UserConfig, SUPPORTED_PROVIDERS

# Default providers if provider_models.json is missing
_DEFAULT_PROVIDERS: Dict[str, Dict[str, Any]] = {
    "openai": {
        "name": "OpenAI",
        "type": "closed",
        "env_var": "OPENAI_API_KEY",
        "models": ["GPT-4o mini", "GPT-4o", "GPT-4 Turbo"],
        "model_api_ids": {
            "GPT-4o mini": "gpt-4o-mini",
            "GPT-4o": "gpt-4o",
            "GPT-4 Turbo": "gpt-4-turbo",
        },
    },
    "anthropic": {
        "name": "Anthropic",
        "type": "closed",
        "env_var": "ANTHROPIC_API_KEY",
        "models": ["Claude 3.5 Sonnet", "Claude 3 Opus", "Claude 3 Haiku"],
        "model_api_ids": {
            "Claude 3.5 Sonnet": "claude-3-5-sonnet-20241022",
            "Claude 3 Opus": "claude-3-opus-20240229",
            "Claude 3 Haiku": "claude-3-haiku-20240307",
        },
    },
    "none": {
        "name": "None (Template-based only)",
        "type": None,
        "env_var": None,
        "models": ["template"],
        "model_api_ids": {"template": "template"},
    },
}

# Cache for loaded providers
_providers_cache: Optional[Dict[str, Dict[str, Any]]] = None


def _get_provider_models_path() -> Path:
    """Path to provider_models.json (next to this module)."""
    return Path(__file__).parent / "provider_models.json"


def get_config_dir(app_name: str = "ai-model-picker") -> Path:
    """
    Get the platform-appropriate config directory.

    Parameters
    ----------
    app_name : str
        The application name for the config directory.

    Returns
    -------
    Path
        The configuration directory path.
    """
    if sys.platform == "win32":
        appdata = os.getenv("APPDATA")
        if appdata:
            config_dir = Path(appdata) / app_name
        else:
            config_dir = Path.home() / "AppData" / "Roaming" / app_name
    elif sys.platform == "darwin":
        config_dir = Path.home() / "Library" / "Application Support" / app_name
    else:
        xdg_config = os.getenv("XDG_CONFIG_HOME")
        if xdg_config:
            config_dir = Path(xdg_config) / app_name
        else:
            config_dir = Path.home() / ".config" / app_name

    config_dir.mkdir(parents=True, exist_ok=True)
    return config_dir


def get_config_path(app_name: str = "ai-model-picker") -> Path:
    """
    Get the path to the user config file.

    Parameters
    ----------
    app_name : str
        The application name for the config directory.

    Returns
    -------
    Path
        The configuration file path.
    """
    return get_config_dir(app_name) / "config.json"


def load_config(app_name: str = "ai-model-picker") -> UserConfig:
    """
    Load user configuration from file.

    Parameters
    ----------
    app_name : str
        The application name for the config directory.

    Returns
    -------
    UserConfig
        The loaded configuration, or defaults if not found.
    """
    config_path = get_config_path(app_name)

    if not config_path.exists():
        return UserConfig()

    try:
        with open(config_path, "r") as f:
            data = json.load(f)
        return UserConfig.from_dict(data)
    except (json.JSONDecodeError, IOError) as e:
        print(f"Warning: Failed to load config: {e}")
        return UserConfig()


def save_config(config: UserConfig, app_name: str = "ai-model-picker") -> None:
    """
    Save user configuration to file.

    Parameters
    ----------
    config : UserConfig
        The configuration to save.
    app_name : str
        The application name for the config directory.
    """
    config_path = get_config_path(app_name)

    try:
        with open(config_path, "w") as f:
            json.dump(config.to_dict(), f, indent=2)
    except IOError as e:
        raise RuntimeError(f"Failed to save config: {e}")


def reset_config(app_name: str = "ai-model-picker") -> None:
    """
    Reset configuration to defaults by deleting the config file.

    Parameters
    ----------
    app_name : str
        The application name for the config directory.
    """
    config_path = get_config_path(app_name)
    if config_path.exists():
        config_path.unlink()


def get_available_providers() -> Dict[str, Dict[str, Any]]:
    """
    Get available AI providers and their models.

    Loads from provider_models.json when present. Falls back to built-in
    defaults if the file is missing or invalid.

    Returns
    -------
    Dict[str, Dict[str, Any]]
        Dictionary mapping provider keys to their configuration.
    """
    global _providers_cache

    if _providers_cache is not None:
        return _providers_cache

    path = _get_provider_models_path()
    if path.exists():
        try:
            with open(path, "r") as f:
                data = json.load(f)
            if isinstance(data, dict) and data:
                _providers_cache = data
                return data
        except (json.JSONDecodeError, OSError):
            pass

    _providers_cache = dict(_DEFAULT_PROVIDERS)
    return _providers_cache


def get_provider_info(provider: str) -> Optional[ProviderInfo]:
    """
    Get detailed information about a provider.

    Parameters
    ----------
    provider : str
        The provider key.

    Returns
    -------
    ProviderInfo | None
        Provider information, or None if not found.
    """
    providers = get_available_providers()
    data = providers.get(provider)
    if data:
        return ProviderInfo.from_dict(data)
    return None


def get_provider_display_name(provider: str) -> str:
    """
    Get display name for a provider.

    Parameters
    ----------
    provider : str
        The provider key.

    Returns
    -------
    str
        The human-readable provider name.
    """
    providers = get_available_providers()
    return providers.get(provider, {}).get("name", provider.title())


def get_provider_models(provider: str) -> List[str]:
    """
    Get available models for a provider.

    Parameters
    ----------
    provider : str
        The provider key.

    Returns
    -------
    List[str]
        List of model display names for the provider.
    """
    providers = get_available_providers()
    return providers.get(provider, {}).get("models", [])


def get_provider_env_var(provider: str) -> Optional[str]:
    """
    Get the environment variable name for a provider's API key.

    Parameters
    ----------
    provider : str
        The provider key.

    Returns
    -------
    str | None
        The environment variable name, or None if not applicable.
    """
    providers = get_available_providers()
    return providers.get(provider, {}).get("env_var")


def get_default_provider(app_name: str = "ai-model-picker") -> str:
    """
    Get the default provider.

    Parameters
    ----------
    app_name : str
        The application name for the config directory.

    Returns
    -------
    str
        The default provider key.
    """
    config = load_config(app_name)
    return config.provider


def get_default_model(app_name: str = "ai-model-picker") -> str:
    """
    Get the default model.

    Parameters
    ----------
    app_name : str
        The application name for the config directory.

    Returns
    -------
    str
        The default model name.
    """
    config = load_config(app_name)
    return config.model


def set_default_provider(provider: str, app_name: str = "ai-model-picker") -> None:
    """
    Set the default provider.

    Parameters
    ----------
    provider : str
        The provider key to set as default.
    app_name : str
        The application name for the config directory.
    """
    config = load_config(app_name)
    config.provider = provider
    save_config(config, app_name)


def set_default_model(model: str, app_name: str = "ai-model-picker") -> None:
    """
    Set the default model.

    Parameters
    ----------
    model : str
        The model name to set as default.
    app_name : str
        The application name for the config directory.
    """
    config = load_config(app_name)
    config.model = model
    save_config(config, app_name)


def get_api_key(provider: str, app_name: str = "ai-model-picker") -> Optional[str]:
    """
    Get the stored API key for a provider from config file.

    Parameters
    ----------
    provider : str
        The provider key.
    app_name : str
        The application name for the config directory.

    Returns
    -------
    str | None
        The API key if found, None otherwise.
    """
    config = load_config(app_name)
    return config.api_keys.get(provider)


def set_api_key(provider: str, api_key: str, app_name: str = "ai-model-picker") -> None:
    """
    Set the API key for a provider.

    Parameters
    ----------
    provider : str
        The provider key.
    api_key : str
        The API key to store.
    app_name : str
        The application name for the config directory.
    """
    config = load_config(app_name)
    config.api_keys[provider] = api_key.strip()
    save_config(config, app_name)


def remove_api_key(provider: str, app_name: str = "ai-model-picker") -> None:
    """
    Remove the stored API key for a provider.

    Parameters
    ----------
    provider : str
        The provider key.
    app_name : str
        The application name for the config directory.
    """
    config = load_config(app_name)
    if provider in config.api_keys:
        del config.api_keys[provider]
        save_config(config, app_name)


def get_all_api_keys(app_name: str = "ai-model-picker") -> Dict[str, str]:
    """
    Get all stored API keys.

    Parameters
    ----------
    app_name : str
        The application name for the config directory.

    Returns
    -------
    Dict[str, str]
        Dictionary mapping providers to their API keys.
    """
    config = load_config(app_name)
    return config.api_keys.copy()


def get_api_key_with_fallback(provider: str, app_name: str = "ai-model-picker") -> Optional[str]:
    """
    Get API key for a provider, falling back to environment variable.

    Checks in this order:
    1. Config file
    2. Environment variable

    Parameters
    ----------
    provider : str
        The provider key.
    app_name : str
        The application name for the config directory.

    Returns
    -------
    str | None
        The API key if found from either source, None otherwise.
    """
    # First check config file
    key = get_api_key(provider, app_name)
    if key:
        return key

    # Fall back to environment variable
    env_var = get_provider_env_var(provider)
    if env_var:
        return os.getenv(env_var)

    return None


def get_model_api_id(
    model_display_name: str,
    provider: Optional[str] = None,
    app_name: str = "ai-model-picker"
) -> str:
    """
    Get the API ID for a model display name.

    First checks user config for custom mappings, then falls back to
    provider_models.json mappings.

    Parameters
    ----------
    model_display_name : str
        The model display name.
    provider : str | None
        Optional provider to narrow down the search.
    app_name : str
        The application name for the config directory.

    Returns
    -------
    str
        The API model ID (returns display name if no mapping found).
    """
    # Check user config for custom mapping
    config = load_config(app_name)
    if model_display_name in config.model_api_ids:
        return config.model_api_ids[model_display_name]

    # Check provider_models.json mappings
    providers = get_available_providers()

    if provider:
        # Search specific provider
        provider_data = providers.get(provider, {})
        model_api_ids = provider_data.get("model_api_ids", {})
        if model_display_name in model_api_ids:
            return model_api_ids[model_display_name]
    else:
        # Search all providers
        for provider_data in providers.values():
            model_api_ids = provider_data.get("model_api_ids", {})
            if model_display_name in model_api_ids:
                return model_api_ids[model_display_name]

    # Return as-is if no mapping found (might already be an API ID)
    return model_display_name


def register_model_api_id(
    model_display_name: str,
    api_id: str,
    app_name: str = "ai-model-picker"
) -> None:
    """
    Register a custom model display name to API ID mapping.

    Parameters
    ----------
    model_display_name : str
        The model display name.
    api_id : str
        The API model ID.
    app_name : str
        The application name for the config directory.
    """
    config = load_config(app_name)
    config.model_api_ids[model_display_name] = api_id
    save_config(config, app_name)
