"""
AI Model Picker - Unified AI model provider selection and configuration.

A shared library for selecting AI providers/models, managing local API keys,
and producing a secrets-free preference payload for other services.

This package is library-first: it is not a token-passing microservice.
Cross-service handoff uses ModelPreference (provider, model_id, instructions),
never upstream API keys.
"""

from .config import (
    # Config path and loading
    get_config_path,
    load_config,
    save_config,
    reset_config,
    # Provider/model getters
    get_available_providers,
    get_providers_source,
    refresh_provider_models,
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

from .preference import (
    build_preference,
    preference_from_config,
    load_preference,
    save_preference,
    to_handoff_dict,
    handoff_json,
    select_preference,
    get_preference_path,
    call_with_preference,
)

from .types import (
    Provider,
    ProviderInfo,
    UserConfig,
    ModelPreference,
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

__version__ = "0.2.0"

__all__ = [
    # Version
    "__version__",
    # Types
    "Provider",
    "ProviderInfo",
    "UserConfig",
    "ModelPreference",
    "SUPPORTED_PROVIDERS",
    # Config
    "get_config_path",
    "load_config",
    "save_config",
    "reset_config",
    "get_available_providers",
    "get_providers_source",
    "refresh_provider_models",
    "get_provider_display_name",
    "get_provider_models",
    "get_provider_env_var",
    "get_default_provider",
    "get_default_model",
    "set_default_provider",
    "set_default_model",
    # API keys (local only — not for cross-service handoff)
    "get_api_key",
    "set_api_key",
    "remove_api_key",
    "get_all_api_keys",
    "get_api_key_with_fallback",
    # Model ID mapping
    "get_model_api_id",
    "register_model_api_id",
    # Preference handoff (secrets-free)
    "build_preference",
    "preference_from_config",
    "load_preference",
    "save_preference",
    "to_handoff_dict",
    "handoff_json",
    "select_preference",
    "get_preference_path",
    "call_with_preference",
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
