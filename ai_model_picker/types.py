"""Type definitions for AI Model Picker."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Literal, Optional, Dict, Any, List

# Supported provider identifiers
Provider = Literal[
    "openai",
    "anthropic",
    "google",
    "mistral",
    "cohere",
    "meta",
    "deepseek",
    "xai",
    "alibaba",
    "moonshot",
    "zai",
    "minimax",
    "perplexity",
    "nvidia",
    "bytedance",
    "tencent",
    "xiaomi",
    "amazon",
    "stepfun",
    "none",
]

SUPPORTED_PROVIDERS: List[Provider] = [
    "openai",
    "anthropic",
    "google",
    "mistral",
    "cohere",
    "meta",
    "deepseek",
    "xai",
    "alibaba",
    "moonshot",
    "zai",
    "minimax",
    "perplexity",
    "nvidia",
    "bytedance",
    "tencent",
    "xiaomi",
    "amazon",
    "stepfun",
    "none",
]


@dataclass
class ProviderInfo:
    """Information about an AI provider."""
    name: str
    type: Optional[str]  # "closed", "open", or None
    env_var: Optional[str]
    models: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProviderInfo":
        return cls(
            name=data.get("name", ""),
            type=data.get("type"),
            env_var=data.get("env_var"),
            models=data.get("models", []),
        )


@dataclass
class ModelPreference:
    """
    Canonical handoff payload for another app or service.

    Contains model selection and optional instructions only.
    Never includes API keys, bearer tokens, or other secrets —
    those stay with the caller, a vault, or an LLM gateway.
    """

    provider: str = "openai"
    model: str = "gpt-4o-mini"  # display name as chosen in the picker
    model_id: str = "gpt-4o-mini"  # native provider API id
    instructions: str = ""
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    app_name: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_handoff_dict(self) -> Dict[str, Any]:
        """Serialize fields intended for cross-service handoff (no secrets)."""
        payload: Dict[str, Any] = {
            "provider": self.provider,
            "model": self.model,
            "model_id": self.model_id,
            "instructions": self.instructions or "",
        }
        if self.temperature is not None:
            payload["temperature"] = self.temperature
        if self.max_tokens is not None:
            payload["max_tokens"] = self.max_tokens
        if self.app_name:
            payload["app_name"] = self.app_name
        return payload

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelPreference":
        model = data.get("model", "gpt-4o-mini")
        return cls(
            provider=data.get("provider", "openai"),
            model=model,
            model_id=data.get("model_id", model),
            instructions=data.get("instructions") or "",
            temperature=data.get("temperature"),
            max_tokens=data.get("max_tokens"),
            app_name=data.get("app_name"),
        )


@dataclass
class UserConfig:
    """User configuration for AI model selection (may include local API keys)."""
    provider: str = "openai"
    model: str = "gpt-4o-mini"
    api_keys: Dict[str, str] = field(default_factory=dict)
    model_api_ids: Dict[str, str] = field(default_factory=dict)  # display_name -> api_id
    instructions: str = ""
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UserConfig":
        return cls(
            provider=data.get("provider", "openai"),
            model=data.get("model", "gpt-4o-mini"),
            api_keys=data.get("api_keys", {}),
            model_api_ids=data.get("model_api_ids", {}),
            instructions=data.get("instructions") or "",
            temperature=data.get("temperature"),
            max_tokens=data.get("max_tokens"),
        )
