"""Tests for secrets-free preference handoff."""

from __future__ import annotations

import json
from unittest import mock

import pytest

from ai_model_picker.preference import (
    build_preference,
    handoff_json,
    load_preference,
    save_preference,
    to_handoff_dict,
    _assert_no_secrets,
)
from ai_model_picker.types import ModelPreference, UserConfig


def test_handoff_dict_excludes_secrets():
    pref = ModelPreference(
        provider="openai",
        model="GPT-4o mini",
        model_id="gpt-4o-mini",
        instructions="Be brief.",
        temperature=0.2,
        max_tokens=1024,
        app_name="test-app",
    )
    payload = to_handoff_dict(pref)
    assert payload["provider"] == "openai"
    assert payload["model_id"] == "gpt-4o-mini"
    assert payload["instructions"] == "Be brief."
    assert "api_key" not in payload
    assert "api_keys" not in payload
    assert "token" not in payload


def test_assert_no_secrets_rejects_credential_fields():
    with pytest.raises(ValueError, match="secret fields"):
        _assert_no_secrets({"provider": "openai", "api_key": "sk-test"})


def test_save_and_load_preference_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))

    with mock.patch(
        "ai_model_picker.preference.get_model_api_id",
        return_value="claude-sonnet-5",
    ):
        pref = build_preference(
            "anthropic",
            "Claude Sonnet 5",
            instructions="Code first.",
            app_name="handoff-demo",
        )
        path = save_preference(pref, app_name="handoff-demo")
        assert path.exists()
        raw = json.loads(path.read_text())
        assert "api_key" not in raw
        assert raw["model_id"] == "claude-sonnet-5"

        loaded = load_preference("handoff-demo")
        assert loaded.provider == "anthropic"
        assert loaded.instructions == "Code first."
        assert loaded.model_id == "claude-sonnet-5"


def test_preference_from_config_ignores_api_keys():
    from ai_model_picker.preference import preference_from_config

    config = UserConfig(
        provider="xai",
        model="Grok 4.5",
        api_keys={"xai": "should-not-leak"},
        instructions="Use tools sparingly.",
    )
    with mock.patch(
        "ai_model_picker.preference.get_model_api_id",
        return_value="grok-4.5",
    ):
        pref = preference_from_config(config, app_name="cli")
        payload = json.loads(handoff_json(pref))
    assert "should-not-leak" not in json.dumps(payload)
    assert payload["model_id"] == "grok-4.5"
    assert payload["instructions"] == "Use tools sparingly."
