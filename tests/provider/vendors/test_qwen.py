"""Tests for `physiclaw.provider.vendors.qwen` — DashScope
declaration.

Pure pins: the id and base URL are registry routing keys, the env vars
are auth precedence, and the declared `SYSTEM_PROMPT_FRAGMENT` is
returned by the base `system_prompt_fragment()` hook (mechanics
exercised in `test_provider_base.py`).
"""

from __future__ import annotations

from physiclaw.provider.vendors.qwen import QwenProvider

# ---------- class metadata ----------


def test_provider_id_pinned() -> None:
    assert QwenProvider.PROVIDER_ID == "qwen"


def test_base_url_pinned() -> None:
    """DashScope's OpenAI-compatible endpoint."""
    assert QwenProvider.BASE_URL == (
        "https://dashscope.aliyuncs.com/compatible-mode/v1"
    )


def test_inherits_openai_compat() -> None:
    from physiclaw.provider.openai_compat import OpenAICompatibleProvider

    assert issubclass(QwenProvider, OpenAICompatibleProvider)


def test_api_key_env_vars_supports_both_qwen_and_dashscope() -> None:
    """Auth precedence — QWEN_API_KEY first, DASHSCOPE_API_KEY as alias."""
    assert QwenProvider.API_KEY_ENV_VARS == ("QWEN_API_KEY", "DASHSCOPE_API_KEY")


# ---------- SYSTEM_PROMPT_FRAGMENT declaration ----------


def test_system_prompt_fragment_mentions_think_wrapper() -> None:
    """Qwen models emit reasoning in `<think>...</think>` blocks; the
    fragment is appended to the system prompt to teach this convention."""
    out = QwenProvider.system_prompt_fragment()

    assert "<think>" in out
    assert "</think>" in out


def test_system_prompt_fragment_warns_against_reasoning_in_args() -> None:
    """Without this warning, Qwen sometimes embeds chain-of-thought into
    tool_call arguments; handlers receive args raw."""
    out = QwenProvider.system_prompt_fragment()

    assert "tool" in out.lower()
    assert "scratchpad" in out.lower() or "raw" in out.lower()


def test_system_prompt_fragment_is_classmethod() -> None:
    """Must be callable on the class, not requiring an instance —
    avoids needing an API key just to render the system prompt."""
    out = QwenProvider.system_prompt_fragment()

    assert isinstance(out, str)
    assert len(out) > 0


def test_system_prompt_fragment_matches_declared_attr() -> None:
    """The base hook returns the declaration verbatim — any reword
    surfaces here as a failed test."""
    assert QwenProvider.system_prompt_fragment() == QwenProvider.SYSTEM_PROMPT_FRAGMENT
    assert QwenProvider.SYSTEM_PROMPT_FRAGMENT != ""
