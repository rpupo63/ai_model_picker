"""
Model preference handoff for consuming apps and sibling services.

This library is intentionally *not* a token broker. The canonical payload
passed to another service is a ModelPreference (provider, model_id,
instructions, optional knobs). Upstream API keys stay with the caller,
a vault, or an LLM gateway — never inside the handoff dict.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

from .config import (
    get_config_dir,
    get_model_api_id,
    load_config,
    save_config,
)
from .selector import select_model, select_provider
from .types import ModelPreference, UserConfig

# Keys that must never appear in a handoff payload
_SECRET_HANDOFF_KEYS = frozenset(
    {
        "api_key",
        "api_keys",
        "token",
        "access_token",
        "refresh_token",
        "secret",
        "password",
        "authorization",
        "auth",
        "bearer",
        "credentials",
    }
)


def get_preference_path(app_name: str = "ai-model-picker") -> Path:
    """Path to the persisted preference file (secrets-free)."""
    return get_config_dir(app_name) / "preference.json"


def build_preference(
    provider: str,
    model: str,
    *,
    instructions: str = "",
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    app_name: str = "ai-model-picker",
) -> ModelPreference:
    """
    Build a ModelPreference from a provider + model display name.

    Resolves the native API model id. Does not read or embed API keys.
    """
    model_id = get_model_api_id(model, provider, app_name)
    return ModelPreference(
        provider=provider,
        model=model,
        model_id=model_id,
        instructions=instructions or "",
        temperature=temperature,
        max_tokens=max_tokens,
        app_name=app_name,
    )


def preference_from_config(
    config: UserConfig,
    app_name: str = "ai-model-picker",
) -> ModelPreference:
    """Derive a handoff preference from a UserConfig (ignores api_keys)."""
    return build_preference(
        provider=config.provider,
        model=config.model,
        instructions=config.instructions,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        app_name=app_name,
    )


def load_preference(app_name: str = "ai-model-picker") -> ModelPreference:
    """
    Load the saved preference for an app.

    Prefers ``preference.json`` when present; otherwise derives from
    ``config.json`` (still without exposing API keys).
    """
    path = get_preference_path(app_name)
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
            if isinstance(data, dict):
                pref = ModelPreference.from_dict(data)
                # Re-resolve model_id in case catalogs moved
                if pref.provider and pref.model:
                    pref.model_id = get_model_api_id(
                        pref.model, pref.provider, app_name
                    )
                pref.app_name = pref.app_name or app_name
                return pref
        except (json.JSONDecodeError, OSError, TypeError, ValueError):
            pass

    return preference_from_config(load_config(app_name), app_name)


def save_preference(
    preference: ModelPreference,
    app_name: str = "ai-model-picker",
    *,
    sync_config: bool = True,
) -> Path:
    """
    Persist a preference to ``preference.json`` (never writes API keys).

    When ``sync_config`` is True, also updates provider/model/instructions
    fields on ``config.json`` without touching ``api_keys``.
    """
    target_app = preference.app_name or app_name
    # Ensure model_id is current
    preference.model_id = get_model_api_id(
        preference.model, preference.provider, target_app
    )
    preference.app_name = target_app

    path = get_preference_path(target_app)
    payload = preference.to_handoff_dict()
    _assert_no_secrets(payload)

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    if sync_config:
        config = load_config(target_app)
        config.provider = preference.provider
        config.model = preference.model
        config.instructions = preference.instructions
        config.temperature = preference.temperature
        config.max_tokens = preference.max_tokens
        save_config(config, target_app)

    return path


def to_handoff_dict(preference: ModelPreference) -> Dict[str, Any]:
    """
    Serialize a preference for another service.

    Guarantees no API keys or credential fields are included.
    """
    payload = preference.to_handoff_dict()
    _assert_no_secrets(payload)
    return payload


def handoff_json(preference: ModelPreference, *, indent: Optional[int] = 2) -> str:
    """JSON string suitable for posting to a sibling service."""
    return json.dumps(to_handoff_dict(preference), indent=indent)


def _assert_no_secrets(payload: Dict[str, Any]) -> None:
    lowered = {str(key).lower() for key in payload}
    overlap = lowered & _SECRET_HANDOFF_KEYS
    if overlap:
        raise ValueError(
            f"Handoff payload must not include secret fields: {sorted(overlap)}"
        )


def call_with_preference(
    prompt: str,
    preference: Optional[ModelPreference] = None,
    *,
    app_name: str = "ai-model-picker",
    api_key: Optional[str] = None,
    system_prompt: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    timeout: int = 60,
):
    """
    Call the configured model using a ModelPreference.

    API keys are resolved locally (config / env), never taken from the
    preference handoff payload.
    """
    from .client import call_ai

    pref = preference or load_preference(app_name)
    # Preference instructions are the default system prompt; explicit
    # system_prompt wins when provided.
    effective_system = (
        system_prompt if system_prompt is not None else (pref.instructions or None)
    )
    effective_temp = (
        temperature if temperature is not None else (pref.temperature if pref.temperature is not None else 0.3)
    )
    effective_max = (
        max_tokens if max_tokens is not None else (pref.max_tokens if pref.max_tokens is not None else 4000)
    )

    return call_ai(
        prompt=prompt,
        provider=pref.provider,
        model=pref.model,
        api_key=api_key,
        app_name=app_name,
        temperature=effective_temp,
        max_tokens=effective_max,
        timeout=timeout,
        system_prompt=effective_system,
    )


def select_preference(
    *,
    app_name: str = "ai-model-picker",
    prompt_instructions: bool = True,
    default_instructions: str = "",
    save: bool = True,
) -> Optional[ModelPreference]:
    """
    Interactively pick provider/model (and optional instructions).

    Returns a secrets-free ModelPreference for handoff to another service.
    """
    existing = load_preference(app_name)

    provider = select_provider("Select AI Provider")
    if not provider:
        return None

    model = select_model(provider)
    if not model:
        return None

    instructions = default_instructions or existing.instructions
    if prompt_instructions:
        try:
            from InquirerPy import inquirer

            instructions = inquirer.text(
                message="Additional instructions for the model (optional)",
                default=instructions or "",
                mandatory=False,
            ).execute()
            if instructions is None:
                instructions = ""
        except KeyboardInterrupt:
            return None

    preference = build_preference(
        provider=provider,
        model=model,
        instructions=instructions or "",
        temperature=existing.temperature,
        max_tokens=existing.max_tokens,
        app_name=app_name,
    )

    if save:
        save_preference(preference, app_name)

    return preference
