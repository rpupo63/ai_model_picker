"""
Fetch live model catalogs from OpenRouter's unified models API.

OpenRouter aggregates active models across major providers into one
standardized list (https://openrouter.ai/api/v1/models). Deprecated
upstream models disappear from that list automatically.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"
DEFAULT_CACHE_TTL_SECONDS = 6 * 60 * 60  # 6 hours
REQUEST_TIMEOUT_SECONDS = 15

# OpenRouter id prefix -> local provider key
OPENROUTER_PREFIX_TO_PROVIDER: Dict[str, str] = {
    "openai": "openai",
    "anthropic": "anthropic",
    "google": "google",
    "mistralai": "mistral",
    "cohere": "cohere",
    "meta-llama": "meta",
    "meta": "meta",
    "deepseek": "deepseek",
    "x-ai": "xai",
    "qwen": "alibaba",
    "moonshotai": "moonshot",
    "z-ai": "zai",
    "minimax": "minimax",
    "perplexity": "perplexity",
    "nvidia": "nvidia",
    "bytedance-seed": "bytedance",
    "tencent": "tencent",
    "xiaomi": "xiaomi",
    "amazon": "amazon",
    "stepfun": "stepfun",
}

# Used to strip "Provider: " from OpenRouter display names
_PROVIDER_NAME_PREFIXES: Dict[str, Tuple[str, ...]] = {
    "openai": ("OpenAI: ",),
    "anthropic": ("Anthropic: ",),
    "google": ("Google: ",),
    "mistral": ("Mistral: ",),
    "cohere": ("Cohere: ",),
    "meta": ("Meta: ",),
    "deepseek": ("DeepSeek: ",),
    "xai": ("xAI: ", "XAI: "),
    "alibaba": ("Qwen: ", "Alibaba: "),
    "moonshot": ("MoonshotAI: ", "Moonshot: ", "Kimi: "),
    "zai": ("Z.ai: ", "Z.AI: ", "Zhipu: ", "GLM: "),
    "minimax": ("MiniMax: ",),
    "perplexity": ("Perplexity: ",),
    "nvidia": ("NVIDIA: ", "Nvidia: "),
    "bytedance": ("ByteDance Seed: ", "ByteDance: ", "Seed: "),
    "tencent": ("Tencent: ", "Hunyuan: "),
    "xiaomi": ("Xiaomi: ", "MiMo: "),
    "amazon": ("Amazon: ", "AWS: "),
    "stepfun": ("StepFun: ", "Step: "),
}


def _cache_ttl_seconds() -> int:
    raw = os.getenv("AI_MODEL_PICKER_CACHE_TTL")
    if raw:
        try:
            return max(0, int(raw))
        except ValueError:
            pass
    return DEFAULT_CACHE_TTL_SECONDS


def _models_url() -> str:
    return os.getenv("AI_MODEL_PICKER_OPENROUTER_URL", OPENROUTER_MODELS_URL)


def _offline_mode() -> bool:
    return os.getenv("AI_MODEL_PICKER_OFFLINE", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _is_chat_model(model: Dict[str, Any]) -> bool:
    """Keep models that accept text and produce text."""
    architecture = model.get("architecture") or {}
    input_modalities = architecture.get("input_modalities") or []
    output_modalities = architecture.get("output_modalities") or []

    if input_modalities and "text" not in input_modalities:
        return False
    if output_modalities and "text" not in output_modalities:
        return False

    return True


def _is_expired(model: Dict[str, Any]) -> bool:
    expiration = model.get("expiration_date")
    if not expiration:
        return False
    try:
        # OpenRouter uses unix seconds or ISO; treat numeric as unix.
        if isinstance(expiration, (int, float)):
            return float(expiration) <= time.time()
        # ISO date / datetime — compare date prefix if parse fails fully
        from datetime import datetime, timezone

        normalized = str(expiration).replace("Z", "+00:00")
        try:
            expires_at = datetime.fromisoformat(normalized)
        except ValueError:
            expires_at = datetime.strptime(str(expiration)[:10], "%Y-%m-%d").replace(
                tzinfo=timezone.utc
            )
        if expires_at.tzinfo is None:
            expires_at = expires_at.replace(tzinfo=timezone.utc)
        return expires_at.timestamp() <= time.time()
    except (TypeError, ValueError):
        return False


def _native_api_id(openrouter_id: str) -> str:
    """Convert openai/gpt-4o[:variant] -> gpt-4o for direct provider APIs."""
    slug = openrouter_id.split("/", 1)[-1]
    # OpenRouter routing/variant suffixes are not native provider model IDs
    return slug.split(":", 1)[0]


def _display_name(model: Dict[str, Any], provider_key: str) -> str:
    name = (model.get("name") or model.get("id") or "").strip()
    for prefix in _PROVIDER_NAME_PREFIXES.get(provider_key, ()):
        if name.startswith(prefix):
            name = name[len(prefix) :].strip()
            break
    return name or model.get("id", "unknown")


def fetch_openrouter_models(
    url: Optional[str] = None,
    timeout: int = REQUEST_TIMEOUT_SECONDS,
) -> List[Dict[str, Any]]:
    """
    Fetch the raw model list from OpenRouter.

    Returns
    -------
    List[Dict[str, Any]]
        The `data` array from the OpenRouter response.

    Raises
    ------
    RuntimeError
        If the request fails or the payload is invalid.
    """
    request_url = url or _models_url()
    request = urllib.request.Request(
        request_url,
        headers={
            "Accept": "application/json",
            "User-Agent": "ai-model-picker/0.1",
        },
        method="GET",
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"OpenRouter models HTTP {exc.code}: {exc.reason}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"OpenRouter models request failed: {exc.reason}") from exc
    except (TimeoutError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise RuntimeError(f"OpenRouter models response invalid: {exc}") from exc

    if isinstance(payload, dict):
        models = payload.get("data")
    elif isinstance(payload, list):
        models = payload
    else:
        models = None

    if not isinstance(models, list):
        raise RuntimeError("OpenRouter models response missing data array")

    return models


def group_models_by_provider(
    openrouter_models: List[Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """
    Group OpenRouter models into local provider catalogs.

    Returns
    -------
    Dict[str, Dict[str, Any]]
        Mapping of provider key -> {"models": [...], "model_api_ids": {...},
        "model_metadata": {...}} sorted newest-first.
    """
    grouped: Dict[str, List[Dict[str, Any]]] = {}

    for model in openrouter_models:
        model_id = model.get("id")
        if not isinstance(model_id, str) or "/" not in model_id:
            continue
        if model_id.startswith("~"):
            continue
        # Skip OpenRouter-only variants (e.g. :free) for direct-provider use
        if ":" in model_id.split("/", 1)[1]:
            continue
        if _is_expired(model) or not _is_chat_model(model):
            continue

        prefix, _ = model_id.split("/", 1)
        provider_key = OPENROUTER_PREFIX_TO_PROVIDER.get(prefix)
        if not provider_key:
            continue

        grouped.setdefault(provider_key, []).append(model)

    catalogs: Dict[str, Dict[str, Any]] = {}
    for provider_key, models in grouped.items():
        models.sort(key=lambda m: m.get("created") or 0, reverse=True)

        display_names: List[str] = []
        model_api_ids: Dict[str, str] = {}
        model_metadata: Dict[str, Dict[str, Any]] = {}
        used_names: Dict[str, int] = {}

        for model in models:
            model_id = model["id"]
            api_id = _native_api_id(model_id)
            base_name = _display_name(model, provider_key)

            # Disambiguate duplicate display names
            count = used_names.get(base_name, 0)
            used_names[base_name] = count + 1
            display = base_name if count == 0 else f"{base_name} ({api_id})"

            display_names.append(display)
            model_api_ids[display] = api_id
            model_metadata[display] = {
                "openrouter_id": model_id,
                "context_length": model.get("context_length"),
                "pricing": model.get("pricing"),
                "description": model.get("description"),
            }

        catalogs[provider_key] = {
            "models": display_names,
            "model_api_ids": model_api_ids,
            "model_metadata": model_metadata,
        }

    return catalogs


def load_cache(cache_path) -> Optional[Dict[str, Any]]:
    """Load a previously written OpenRouter cache file, if valid."""
    try:
        with open(cache_path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None

    if not isinstance(data, dict) or "providers" not in data:
        return None
    return data


def cache_is_fresh(cache: Dict[str, Any], ttl_seconds: Optional[int] = None) -> bool:
    """Return True if cache fetched_at is within the TTL window."""
    ttl = _cache_ttl_seconds() if ttl_seconds is None else ttl_seconds
    if ttl <= 0:
        return False
    fetched_at = cache.get("fetched_at")
    if not isinstance(fetched_at, (int, float)):
        return False
    return (time.time() - float(fetched_at)) < ttl


def save_cache(cache_path, providers: Dict[str, Dict[str, Any]]) -> None:
    """Persist grouped provider catalogs to disk."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "source": "openrouter",
        "url": _models_url(),
        "fetched_at": time.time(),
        "providers": providers,
    }
    tmp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    tmp_path.replace(cache_path)


def get_live_provider_catalogs(
    cache_path,
    *,
    force_refresh: bool = False,
) -> Tuple[Optional[Dict[str, Dict[str, Any]]], Optional[str]]:
    """
    Resolve live provider catalogs from cache or OpenRouter.

    Returns
    -------
    (catalogs, error)
        catalogs is provider_key -> model catalog, or None on total failure.
        error is a warning string when falling back / failing, else None.
    """
    if not force_refresh and not _offline_mode():
        cache = load_cache(cache_path)
        if cache and cache_is_fresh(cache):
            providers = cache.get("providers")
            if isinstance(providers, dict) and providers:
                return providers, None

    if _offline_mode():
        cache = load_cache(cache_path)
        if cache and isinstance(cache.get("providers"), dict) and cache["providers"]:
            return cache["providers"], None
        return None, "offline mode enabled and no OpenRouter cache available"

    try:
        raw_models = fetch_openrouter_models()
        catalogs = group_models_by_provider(raw_models)
        if not catalogs:
            return None, "OpenRouter returned no mappable models"
        save_cache(cache_path, catalogs)
        return catalogs, None
    except RuntimeError as exc:
        cache = load_cache(cache_path)
        if cache and isinstance(cache.get("providers"), dict) and cache["providers"]:
            return cache["providers"], f"using stale OpenRouter cache ({exc})"
        return None, str(exc)
