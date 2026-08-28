"""Tests for `physiclaw.provider.vendors.deepseek` — stub provider.

DeepSeek's chat-completions API is text-only (no `image_url` parts),
so PhysiClaw can't drive its observation→action loop with it.
`__init__` raises `NotImplementedError` immediately so misconfiguration
fails loud.
"""

from __future__ import annotations

import pytest

from physiclaw.provider.vendors.deepseek import DeepSeekProvider


def test_init_always_raises_notimplemented() -> None:
    with pytest.raises(NotImplementedError, match="not implemented"):
        DeepSeekProvider()


def test_init_message_explains_vision_requirement() -> None:
    with pytest.raises(NotImplementedError) as exc:
        DeepSeekProvider()

    assert "text-only" in str(exc.value)
    assert "vision" in str(exc.value)
    assert "peek" in str(exc.value)


def test_init_raises_with_args_passed() -> None:
    """The override accepts *args, **kwargs but raises before using them
    — caller's misconfigured construction still fails fast."""
    with pytest.raises(NotImplementedError):
        DeepSeekProvider(model="deepseek-chat", api_key="dummy")


def test_class_metadata_pinned() -> None:
    """The provider id and base URL are part of the registry key —
    rename them and routing breaks silently elsewhere."""
    assert DeepSeekProvider.PROVIDER_ID == "deepseek"
    assert DeepSeekProvider.BASE_URL == "https://api.deepseek.com/v1"


def test_inherits_from_openai_compat() -> None:
    """Confirms registry / discovery sees DeepSeekProvider as part of
    the OpenAI-compatible family even though instantiation is blocked."""
    from physiclaw.provider.openai_compat import OpenAICompatibleProvider

    assert issubclass(DeepSeekProvider, OpenAICompatibleProvider)
