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
class UserConfig:
    """User configuration for AI model selection."""
    provider: str = "openai"
    model: str = "gpt-4o-mini"
    api_keys: Dict[str, str] = field(default_factory=dict)
    model_api_ids: Dict[str, str] = field(default_factory=dict)  # display_name -> api_id

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UserConfig":
        return cls(
            provider=data.get("provider", "openai"),
            model=data.get("model", "gpt-4o-mini"),
            api_keys=data.get("api_keys", {}),
            model_api_ids=data.get("model_api_ids", {}),
        )
