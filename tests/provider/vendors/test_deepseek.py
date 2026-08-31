"""Tests for `physiclaw.provider.vendors.deepseek` — DeepSeek/V4
declaration.

Pure pins plus one knob: cache markers are OFF (automatic prefix
cache; string-only system content — see `vendors/deepseek.py`).
Usage parsing rides the base `_parse_usage` fallback, exercised in
`test_openai_compat.py`.
"""

from __future__ import annotations

import pytest

from physiclaw.provider.openai_compat import OpenAICompatibleProvider
from physiclaw.provider.provider_base import NO_CACHE_MARKERS
from physiclaw.provider.vendors.deepseek import DeepSeekProvider

# ---------- class metadata ----------


def test_provider_id_and_base_url_pinned() -> None:
    """The provider id and base URL are part of the registry key —
    rename them and routing breaks silently elsewhere."""
    assert DeepSeekProvider.PROVIDER_ID == "deepseek"
    assert DeepSeekProvider.BASE_URL == "https://api.deepseek.com/v1"


def test_inherits_openai_compat() -> None:
    assert issubclass(DeepSeekProvider, OpenAICompatibleProvider)


def test_constructs_with_key_set(monkeypatch: pytest.MonkeyPatch) -> None:
    """No longer a stub — with a credential present, construction
    succeeds. The model id passes through verbatim; a text-only pick
    fails on the first peek with the API's own error."""
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")

    p = DeepSeekProvider(model="deepseek-v4-flash-vision-exp")

    assert p.model == "deepseek-v4-flash-vision-exp"


# ---------- declared knobs ----------


def test_cache_markers_disabled() -> None:
    """DeepSeek's prefix cache is automatic and marker-free, and its
    docs specify string-only system content, which the `mark_system`
    rewrap would violate — see `vendors/deepseek.py`. (The null-object
    template path itself is pinned in `test_provider_base.py`.)"""
    assert DeepSeekProvider.CACHE_MARKERS is NO_CACHE_MARKERS


def test_no_parse_usage_override() -> None:
    """DeepSeek's `prompt_cache_hit_tokens` spelling is handled by the
    base `_parse_usage` fallback (exercised in `test_openai_compat.py`)
    — a vendor override reappearing here would mean the quirk handling
    is duplicated."""
    assert "_parse_usage" not in DeepSeekProvider.__dict__
