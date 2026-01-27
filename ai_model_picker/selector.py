"""
Interactive selection interface for AI providers and models.

Provides searchable dropdowns for selecting providers and models
in a terminal interface.
"""

from __future__ import annotations

import sys
from typing import List, Optional, Tuple

from .config import (
    get_available_providers,
    get_provider_display_name,
    get_provider_models,
)


def _filter_options(options: List[str], query: str) -> List[str]:
    """Filter options by substring match (case-insensitive)."""
    if not query:
        return options
    query_lower = query.lower()
    return [opt for opt in options if query_lower in opt.lower()]


def _display_filtered_options(
    options: List[str],
    query: str = "",
    title: str = "Options"
) -> List[str]:
    """Display filtered options with numbering."""
    filtered = _filter_options(options, query)

    print(f"\n{title}:")
    if query:
        print(f"  Search: '{query}' -> {len(filtered)}/{len(options)} matches")
    else:
        print(f"  Showing all {len(filtered)} options")
    print()

    if not filtered:
        print("  No matches found. Try a different search term.")
        return filtered

    # Show up to 25 options
    display_count = min(len(filtered), 25)
    for i, option in enumerate(filtered[:display_count], 1):
        print(f"  {i:2}. {option}")

    if len(filtered) > display_count:
        print(f"  ... and {len(filtered) - display_count} more (type to narrow down)")

    return filtered


def _parse_numeric_selection(user_input: str, filtered: List[str]) -> Optional[str]:
    """Parse numeric selection input."""
    try:
        num = int(user_input)
        max_valid = min(len(filtered), 25)
        if 1 <= num <= max_valid:
            return filtered[num - 1]
        else:
            print(f"  Invalid number. Please enter 1-{max_valid}")
            return None
    except ValueError:
        return None


def _select_from_options(
    options: List[str],
    prompt: str = "Select",
    allow_cancel: bool = True
) -> Optional[str]:
    """
    Interactive searchable selection from a list of options.

    Parameters
    ----------
    options : List[str]
        The options to select from.
    prompt : str
        The prompt title to display.
    allow_cancel : bool
        Whether to allow cancellation with empty input.

    Returns
    -------
    str | None
        The selected option, or None if cancelled.
    """
    query = ""
    filtered = options.copy()

    while True:
        filtered = _display_filtered_options(options, query, prompt)

        if not filtered:
            hint = "Type to search"
            if allow_cancel:
                hint += ", or Ctrl+C to cancel"
            print(f"\n  {hint}")
        else:
            max_num = min(len(filtered), 25)
            print(f"\n  Type to search, Enter for #1, or number (1-{max_num}) to select:")

        try:
            user_input = input("  > ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n  Cancelled.")
            return None

        if not user_input:
            # Enter pressed - select first match
            if filtered:
                selected = filtered[0]
                print(f"\n  Selected: {selected}\n")
                return selected
            else:
                print("  No selection available. Please type to filter.")
                continue

        # Try parsing as number
        choice = _parse_numeric_selection(user_input, filtered)
        if choice:
            print(f"\n  Selected: {choice}\n")
            return choice

        # Not a number, treat as search query
        query = user_input
        filtered = _filter_options(options, query)

        # Auto-select if exactly one match
        if len(filtered) == 1:
            selected = filtered[0]
            print(f"\n  Auto-selected: {selected}\n")
            return selected


def select_provider(prompt: str = "Select AI Provider") -> Optional[str]:
    """
    Interactively select an AI provider.

    Parameters
    ----------
    prompt : str
        The prompt to display.

    Returns
    -------
    str | None
        The selected provider key, or None if cancelled.
    """
    providers = get_available_providers()

    # Create display options: "key - Display Name"
    options = []
    key_map = {}

    for key in providers:
        display_name = get_provider_display_name(key)
        option = f"{key} - {display_name}"
        options.append(option)
        key_map[option] = key

    print("\n" + "=" * 60)
    selected = _select_from_options(options, prompt)

    if selected:
        return key_map[selected]
    return None


def select_model(
    provider: str,
    prompt: Optional[str] = None
) -> Optional[str]:
    """
    Interactively select a model for a provider.

    Parameters
    ----------
    provider : str
        The provider key.
    prompt : str | None
        Optional custom prompt. Defaults to "Select Model for {provider}".

    Returns
    -------
    str | None
        The selected model name, or None if cancelled.
    """
    if provider == "none":
        return "template"

    models = get_provider_models(provider)
    if not models:
        print(f"No models available for {provider}")
        return None

    if prompt is None:
        display_name = get_provider_display_name(provider)
        prompt = f"Select Model for {display_name}"

    print("\n" + "=" * 60)
    return _select_from_options(models, prompt)


def select_provider_and_model(
    provider_prompt: str = "Select AI Provider",
    model_prompt: Optional[str] = None
) -> Tuple[Optional[str], Optional[str]]:
    """
    Interactively select both provider and model.

    Parameters
    ----------
    provider_prompt : str
        The prompt for provider selection.
    model_prompt : str | None
        The prompt for model selection. Defaults to "Select Model for {provider}".

    Returns
    -------
    Tuple[str | None, str | None]
        Tuple of (provider, model), either may be None if cancelled.
    """
    provider = select_provider(provider_prompt)
    if not provider:
        return None, None

    model = select_model(provider, model_prompt)
    return provider, model
