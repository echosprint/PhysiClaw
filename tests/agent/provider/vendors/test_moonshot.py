"""Tests for `physiclaw.agent.provider.vendors.moonshot` — Moonshot/K2
declaration.

The vendor is a pure declaration: K2's top-level `usage.cached_tokens`
quirk is handled by the base `_parse_usage` fallback (exercised in
`test_openai_compat.py`), and the load-bearing cache markers are the
inherited `OpenAICacheMarkers`.
"""

from __future__ import annotations

from physiclaw.agent.provider.openai_compat import OpenAICompatibleProvider
from physiclaw.agent.provider.provider_base import NO_CACHE_MARKERS
from physiclaw.agent.provider.vendors.moonshot import MoonshotProvider

# ---------- class metadata ----------


def test_provider_id_and_base_url_pinned() -> None:
    """Routing key + default region. Override BASE_URL via config for
    the .ai endpoint."""
    assert MoonshotProvider.PROVIDER_ID == "moonshot"
    assert MoonshotProvider.BASE_URL == "https://api.moonshot.cn/v1"


def test_inherits_openai_compat() -> None:
    assert issubclass(MoonshotProvider, OpenAICompatibleProvider)


# ---------- inherited cache markers ----------


def test_inherits_cache_markers() -> None:
    """The inherited `OpenAICacheMarkers` are load-bearing for K2
    cross-wake cache hits (see the module docstring's A/B results) —
    a well-meaning `CACHE_MARKERS = NO_CACHE_MARKERS` here would
    silently cold-start every wake."""
    assert MoonshotProvider.CACHE_MARKERS is OpenAICompatibleProvider.CACHE_MARKERS
    assert MoonshotProvider.CACHE_MARKERS is not NO_CACHE_MARKERS


def test_no_parse_usage_override() -> None:
    """K2's top-level `cached_tokens` quirk is handled by the base
    `_parse_usage` fallback — a vendor override reappearing here would
    mean the quirk handling is duplicated."""
    assert "_parse_usage" not in MoonshotProvider.__dict__
