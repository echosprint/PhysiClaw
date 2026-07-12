"""Tests for `physiclaw.agent.provider.registry` — id → class lookups.

The registry is a tiny dispatch table; tests pin the membership and
exercise each lookup branch including the `claude-code` subprocess
engine which is intentionally NOT in the in-process table.
"""

from __future__ import annotations

import pytest

from physiclaw.agent.provider import registry
from physiclaw.agent.provider.registry import (
    CLAUDE_CODE_ID,
    in_process_provider_ids,
    is_known,
    make_provider,
    provider_class,
    provider_key_status,
)
from physiclaw.agent.provider.vendors.anthropic import AnthropicProvider
from physiclaw.agent.provider.vendors.deepseek import DeepSeekProvider
from physiclaw.agent.provider.vendors.google import GoogleProvider
from physiclaw.agent.provider.vendors.moonshot import MoonshotProvider
from physiclaw.agent.provider.vendors.openai import OpenAIProvider
from physiclaw.agent.provider.vendors.qwen import QwenProvider


# ---------- constants ----------


def test_claude_code_id_constant_pinned() -> None:
    assert CLAUDE_CODE_ID == "claude-code"


def test_provider_classes_dict_has_exactly_six_known_ids() -> None:
    assert set(registry._PROVIDER_CLASSES) == {
        "qwen",
        "moonshot",
        "openai",
        "anthropic",
        "google",
        "deepseek",
    }


@pytest.mark.parametrize(
    "provider_id, cls",
    [
        ("qwen", QwenProvider),
        ("moonshot", MoonshotProvider),
        ("openai", OpenAIProvider),
        ("anthropic", AnthropicProvider),
        ("google", GoogleProvider),
        ("deepseek", DeepSeekProvider),
    ],
)
def test_provider_classes_each_id_maps_to_its_vendor_class(
    provider_id: str, cls: type
) -> None:
    assert registry._PROVIDER_CLASSES[provider_id] is cls


def test_claude_code_is_NOT_in_provider_classes_dict() -> None:
    # claude-code is the subprocess engine; routed by launcher, not
    # via _PROVIDER_CLASSES. Putting it in the dict would let
    # make_provider try to instantiate the wrong path.
    assert CLAUDE_CODE_ID not in registry._PROVIDER_CLASSES


# ---------- in_process_provider_ids ----------


def test_in_process_provider_ids_returns_tuple() -> None:
    out = in_process_provider_ids()

    assert isinstance(out, tuple)
    assert set(out) == set(registry._PROVIDER_CLASSES)


# ---------- is_known ----------


@pytest.mark.parametrize(
    "provider_id", ["qwen", "moonshot", "openai", "anthropic", "google", "deepseek"]
)
def test_is_known_true_for_each_in_process_provider(provider_id: str) -> None:
    assert is_known(provider_id) is True


@pytest.mark.parametrize("unknown", ["claude-code", "mystery", "", "Qwen"])
def test_is_known_false_for_other_ids(unknown: str) -> None:
    assert is_known(unknown) is False


# ---------- provider_class ----------


def test_provider_class_returns_class_for_known_id() -> None:
    assert provider_class("openai") is OpenAIProvider


def test_provider_class_returns_none_for_unknown_id() -> None:
    assert provider_class("mystery") is None


def test_provider_class_returns_none_for_claude_code() -> None:
    # claude-code routes via launcher, not the in-process table.
    assert provider_class(CLAUDE_CODE_ID) is None


# ---------- provider_key_status ----------


def test_provider_key_status_returns_none_for_unknown_provider_id() -> None:
    assert provider_key_status("mystery") == (None, None)


def test_provider_key_status_returns_none_for_claude_code() -> None:
    # Even though claude-code is a known concept, it isn't an
    # in-process provider — key_status falls through to (None, None).
    assert provider_key_status(CLAUDE_CODE_ID) == (None, None)


def test_provider_key_status_returns_masked_value_when_key_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-secret-1234")

    masked, source = provider_key_status("openai")

    # The actual secret is never returned — always shown as 8 stars.
    assert masked == "********"
    assert source is not None
    assert "OPENAI_API_KEY" in source


def test_provider_key_status_returns_none_source_when_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Strip every env var the resolver looks at + clear config.
    for var in ["OPENAI_API_KEY", "OPENAI_KEY"]:
        monkeypatch.delenv(var, raising=False)

    from physiclaw.common import config

    monkeypatch.setattr(config, "CONFIG", config.Config())

    masked, source = provider_key_status("openai")

    assert masked is None
    assert source is None


# ---------- make_provider ----------


def test_make_provider_instantiates_known_provider_class(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    p = make_provider("openai", "gpt-5")

    assert isinstance(p, OpenAIProvider)


def test_make_provider_passes_model_id_through(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")

    p = make_provider("anthropic", "claude-opus-4-7")

    assert p.model == "claude-opus-4-7"


def test_make_provider_raises_value_error_for_unknown_id() -> None:
    with pytest.raises(
        ValueError,
        match=(
            r"^unknown provider 'mystery' \(known in-process: .+; "
            r"or use 'claude-code' for the subprocess engine\)$"
        ),
    ):
        make_provider("mystery", "any-model")


def test_make_provider_error_message_lists_each_in_process_id() -> None:
    with pytest.raises(ValueError) as exc_info:
        make_provider("nope", "model")

    msg = str(exc_info.value)
    for known in ("qwen", "moonshot", "openai", "anthropic", "google", "deepseek"):
        assert known in msg
    # Comma-space join — kills the `', '` ↔ `'XX, XX'` separator mutation.
    assert "qwen, moonshot" in msg or ", " in msg
    assert "XX" not in msg
