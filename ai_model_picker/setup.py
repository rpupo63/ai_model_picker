"""
Interactive setup wizard for AI Model Picker.

Provides guided configuration for selecting providers, models, and API keys.
"""

from __future__ import annotations

import sys
from typing import Optional, List, Callable

from .config import (
    get_config_path,
    load_config,
    save_config,
    get_available_providers,
    get_provider_display_name,
    get_provider_env_var,
    get_provider_models,
    get_api_key,
    set_api_key,
    get_all_api_keys,
)
from .selector import select_provider, select_model
from .types import UserConfig


def mask_api_key(key: str) -> str:
    """Mask API key for display."""
    if not key or len(key) < 8:
        return "****"
    return f"{key[:8]}...{key[-4:]}"


def display_config(
    app_name: str = "ai-model-picker",
    show_keys: bool = False
) -> None:
    """
    Display current configuration.

    Parameters
    ----------
    app_name : str
        The application name for the config directory.
    show_keys : bool
        Whether to show full API keys (default: masked).
    """
    config = load_config(app_name)
    config_path = get_config_path(app_name)

    print()
    print("=" * 60)
    print("Current Configuration")
    print("=" * 60)
    print()
    print(f"Provider: {get_provider_display_name(config.provider)}")
    print(f"Model: {config.model}")
    print()

    # API Keys
    print("API Keys:")
    providers = get_available_providers()
    has_keys = False

    for provider_key in providers:
        if provider_key == "none":
            continue

        key = config.api_keys.get(provider_key)
        display_name = get_provider_display_name(provider_key)
        env_var = get_provider_env_var(provider_key)

        if key:
            has_keys = True
            if show_keys:
                print(f"  {display_name}: {key}")
            else:
                print(f"  {display_name}: {mask_api_key(key)}")
        else:
            env_hint = f" (using ${env_var})" if env_var else ""
            print(f"  {display_name}: Not set{env_hint}")

    print()
    print(f"Config file: {config_path}")
    print("=" * 60)
    print()


def configure_api_keys(
    app_name: str = "ai-model-picker",
    providers: Optional[List[str]] = None,
    require_at_least_one: bool = True
) -> List[str]:
    """
    Interactively configure API keys for providers.

    Parameters
    ----------
    app_name : str
        The application name for the config directory.
    providers : List[str] | None
        Specific providers to configure. If None, configures all providers.
    require_at_least_one : bool
        Whether at least one provider must be configured.

    Returns
    -------
    List[str]
        List of provider keys that were configured.
    """
    all_providers = get_available_providers()

    if providers is None:
        providers = [p for p in all_providers if p != "none"]

    configured = []

    for provider_key in sorted(providers):
        if provider_key == "none":
            continue

        display_name = get_provider_display_name(provider_key)
        env_var = get_provider_env_var(provider_key)

        print(f"\n{display_name}:")
        print("-" * 40)

        # Show existing key if any
        existing_key = get_api_key(provider_key, app_name)
        if existing_key:
            print(f"Current key: {mask_api_key(existing_key)}")
            response = input("Update this key? (y/N): ").strip().lower()
            if response not in ["y", "yes"]:
                configured.append(provider_key)
                continue

        # Show env var hint
        if env_var:
            print(f"(or set ${env_var} environment variable)")

        # Prompt for key
        api_key = input(f"Enter {display_name} API key (or press Enter to skip): ").strip()

        if api_key:
            set_api_key(provider_key, api_key, app_name)
            configured.append(provider_key)
            print(f"  {display_name} API key saved")
        else:
            print(f"  Skipped {display_name}")

    if require_at_least_one and not configured:
        print("\nError: You must configure at least one API provider.")
        sys.exit(1)

    return configured


def setup_wizard(
    app_name: str = "ai-model-picker",
    title: str = "AI Model Picker - Setup Wizard",
    configure_keys: bool = True,
    on_complete: Optional[Callable[[UserConfig], None]] = None
) -> UserConfig:
    """
    Run the interactive setup wizard.

    Parameters
    ----------
    app_name : str
        The application name for the config directory.
    title : str
        The title to display at the top.
    configure_keys : bool
        Whether to include API key configuration step.
    on_complete : Callable | None
        Optional callback to run after setup completes.

    Returns
    -------
    UserConfig
        The configured user settings.
    """
    print("=" * 70)
    print(f"  {title}")
    print("=" * 70)
    print()
    print("Welcome! This wizard will help you configure your AI settings.")
    print()
    print("You can change these settings later using your app's config commands.")
    print()

    # Check if already configured
    existing_keys = get_all_api_keys(app_name)
    if existing_keys:
        print("You already have API keys configured for:")
        for provider in existing_keys:
            print(f"  - {get_provider_display_name(provider)}")
        print()
        response = input("Do you want to reconfigure? (y/N): ").strip().lower()
        if response not in ["y", "yes"]:
            print("\nSetup cancelled. Your existing configuration is unchanged.")
            return load_config(app_name)
        print()

    config = load_config(app_name)

    # Step 1: API Key Setup (optional)
    if configure_keys:
        print("-" * 70)
        print("Step 1: Configure AI Provider API Keys")
        print("-" * 70)
        print()
        print("You can configure multiple providers. At least one is required.")

        configured_providers = configure_api_keys(
            app_name=app_name,
            require_at_least_one=True
        )

        print()
        print("=" * 70)
        display_names = [get_provider_display_name(p) for p in configured_providers]
        print(f"  API keys configured for: {', '.join(display_names)}")
        print("=" * 70)

    # Step 2: Select Default Provider
    print()
    print("-" * 70)
    print("Step 2: Choose Default AI Provider")
    print("-" * 70)
    print()
    print(f"Current default: {get_provider_display_name(config.provider)}")
    print()

    provider = select_provider("Select Default Provider")
    if provider:
        config.provider = provider
    else:
        print(f"  Keeping current default: {config.provider}")

    # Step 3: Select Default Model
    print()
    print("-" * 70)
    print("Step 3: Choose Default Model")
    print("-" * 70)
    print()
    print(f"Current default: {config.model}")
    print()

    model = select_model(config.provider, "Select Default Model")
    if model:
        config.model = model
    else:
        print(f"  Keeping current default: {config.model}")

    # Save configuration
    save_config(config, app_name)

    # Summary
    print()
    print("=" * 70)
    print("  Setup Complete!")
    print("=" * 70)
    print()
    print(f"Configuration saved to: {get_config_path(app_name)}")
    print()
    print(f"Default provider: {get_provider_display_name(config.provider)}")
    print(f"Default model: {config.model}")
    print()

    if on_complete:
        on_complete(config)

    return config
