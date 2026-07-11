"""Tests for `physiclaw.agent.provider.vendors.anthropic` — adapter class.

Distinct from the `claude-code` engine. This vendor calls Anthropic's
`/v1/messages` endpoint via the `AsyncAnthropic` SDK; wire-format flow
lives in `provider/anthropic_compat.py` and is exercised separately.
"""

from __future__ import annotations

from physiclaw.agent.provider.vendors.anthropic import AnthropicProvider


def test_provider_id_pinned() -> None:
    """Routing key for the registry. Renaming breaks `physiclaw models
    use anthropic/<id>` and any config references."""
    assert AnthropicProvider.PROVIDER_ID == "anthropic"


def test_base_url_pinned_for_documentation() -> None:
    """The SDK manages the URL itself — this constant is documentation
    so operators can spot which endpoint family the provider hits."""
    assert AnthropicProvider.BASE_URL == "https://api.anthropic.com/v1"


def test_inherits_anthropic_compat() -> None:
    """Wire format / response parsing comes from the parent."""
    from physiclaw.agent.provider.anthropic_compat import (
        AnthropicCompatibleProvider,
    )

    assert issubclass(AnthropicProvider, AnthropicCompatibleProvider)


def test_distinct_from_claude_code_engine() -> None:
    """The `claude-code` provider id is the subprocess engine; this
    one (`anthropic`) is the in-process SDK adapter. They share an
    API key but are different code paths."""
    from physiclaw.agent.provider import CLAUDE_CODE_ID

    assert AnthropicProvider.PROVIDER_ID != CLAUDE_CODE_ID
