"""
AI Model Picker - Unified AI model provider selection and configuration.

A shared library for selecting AI providers, models, and managing API keys
across multiple applications.
"""

from .config import (
    # Config path and loading
    get_config_path,
    load_config,
    save_config,
    reset_config,
    # Provider/model getters
    get_available_providers,
    get_provider_display_name,
    get_provider_models,
    get_provider_env_var,
    get_default_provider,
    get_default_model,
    set_default_provider,
    set_default_model,
    # API key management
    get_api_key,
    set_api_key,
    remove_api_key,
    get_all_api_keys,
    get_api_key_with_fallback,
    # Model ID mapping
    get_model_api_id,
    register_model_api_id,
)

from .selector import (
    select_provider,
    select_model,
    select_provider_and_model,
)

from .setup import (
    setup_wizard,
    configure_api_keys,
    display_config,
)

from .types import (
    Provider,
    ProviderInfo,
    UserConfig,
    SUPPORTED_PROVIDERS,
)

from .client import (
    call_ai,
    call_ai_simple,
    AIResponse,
    AIClientConfig,
    get_supported_providers,
    check_provider_available,
)

__version__ = "0.1.0"

__all__ = [
    # Version
    "__version__",
    # Types
    "Provider",
    "ProviderInfo",
    "UserConfig",
    "SUPPORTED_PROVIDERS",
    # Config
    "get_config_path",
    "load_config",
    "save_config",
    "reset_config",
    "get_available_providers",
    "get_provider_display_name",
    "get_provider_models",
    "get_provider_env_var",
    "get_default_provider",
    "get_default_model",
    "set_default_provider",
    "set_default_model",
    # API keys
    "get_api_key",
    "set_api_key",
    "remove_api_key",
    "get_all_api_keys",
    "get_api_key_with_fallback",
    # Model ID mapping
    "get_model_api_id",
    "register_model_api_id",
    # Selector
    "select_provider",
    "select_model",
    "select_provider_and_model",
    # Setup
    "setup_wizard",
    "configure_api_keys",
    "display_config",
    # AI Client
    "call_ai",
    "call_ai_simple",
    "AIResponse",
    "AIClientConfig",
    "get_supported_providers",
    "check_provider_available",
]
