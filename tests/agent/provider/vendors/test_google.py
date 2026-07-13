"""Tests for `physiclaw.agent.provider.vendors.google`.

GoogleProvider extends OpenAICompatibleProvider with shim-specific
quirks: image_url isn't allowed in role:tool, and tool_calls require
a thought_signature. Tests exercise both wire-level transformations
without going through the network.
"""

from __future__ import annotations

import pytest
import respx

from physiclaw.agent.engine.dto import (
    AssistantMessage,
    FinishReason,
    ImageBlock,
    SystemMessage,
    TextBlock,
    ToolCall,
    ToolResultMessage,
    UserMessage,
)
from physiclaw.agent.provider.provider_base import NO_CACHE_MARKERS
from physiclaw.agent.provider.vendors.google import (
    _SIG_BYPASS,
    GoogleProvider,
    _encode_tool_result,
    _extract_thought_signature,
)


@pytest.fixture(autouse=True)
def _stub_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")


@pytest.fixture
def provider() -> GoogleProvider:
    return GoogleProvider(model="gemini-3-flash-preview")


# ---------- constants ----------


def test_provider_id_is_google() -> None:
    assert GoogleProvider.PROVIDER_ID == "google"


def test_base_url_uses_google_openai_shim() -> None:
    assert GoogleProvider.BASE_URL == (
        "https://generativelanguage.googleapis.com/v1beta/openai"
    )


def test_sig_bypass_constant_pinned() -> None:
    assert _SIG_BYPASS == "skip_thought_signature_validator"


# ---------- list_models ----------


@pytest.mark.asyncio
async def test_list_models_strips_models_slash_prefix(
    provider: GoogleProvider, respx_mock: respx.MockRouter
) -> None:
    respx_mock.get(
        "https://generativelanguage.googleapis.com/v1beta/openai/models"
    ).respond(
        json={
            "data": [
                {"id": "models/gemini-3-flash-preview", "object": "model"},
                {"id": "models/gemini-2.5-pro", "owned_by": "google"},
            ],
        },
    )

    out = await provider.list_models()

    assert out == [
        {"id": "gemini-3-flash-preview", "object": "model"},
        {"id": "gemini-2.5-pro", "owned_by": "google"},
    ]


@pytest.mark.asyncio
async def test_list_models_passes_through_id_without_models_prefix(
    provider: GoogleProvider, respx_mock: respx.MockRouter
) -> None:
    # Defensive — if Google ever drops the prefix, the helper must not
    # double-mangle.
    respx_mock.get(
        "https://generativelanguage.googleapis.com/v1beta/openai/models"
    ).respond(json={"data": [{"id": "gemini-3"}]})

    assert await provider.list_models() == [{"id": "gemini-3"}]


# ---------- cache markers: disabled for Google ----------


def test_cache_markers_disabled() -> None:
    """The shim ignores `cache_control`, and the shifting stub marker
    would perturb Gemini's implicit prefix cache — markers off. (The
    null-object template path itself is pinned in
    `test_provider_base.py`.)"""
    assert GoogleProvider.CACHE_MARKERS is NO_CACHE_MARKERS


# ---------- _encode_message dispatch ----------


def test_encode_system_message_delegates_to_super(
    provider: GoogleProvider,
) -> None:
    out = provider._encode_message(SystemMessage(content="hi"))

    assert out == {"role": "system", "content": "hi"}


def test_encode_user_message_delegates_to_super(
    provider: GoogleProvider,
) -> None:
    out = provider._encode_message(UserMessage(content="ping"))

    assert out == {"role": "user", "content": "ping"}


def test_encode_assistant_with_no_tool_calls_delegates_to_super(
    provider: GoogleProvider,
) -> None:
    out = provider._encode_message(
        AssistantMessage(
            content="ack",
            tool_calls=[],
            finish_reason=FinishReason.STOP,
        )
    )

    assert out == {"role": "assistant", "content": "ack"}


def test_encode_assistant_with_tool_calls_uses_captured_thought_signature(
    provider: GoogleProvider,
) -> None:
    msg = AssistantMessage(
        content="reasoning",
        tool_calls=[ToolCall(id="t1", name="tap", arguments={"x": 1})],
        finish_reason=FinishReason.TOOL_CALLS,
        vendor_extra={"google": {"thought_signature": "sig-from-prev-turn"}},
    )

    out = provider._encode_message(msg)

    assert out["tool_calls"][0]["extra_content"] == {
        "google": {"thought_signature": "sig-from-prev-turn"},
    }


def test_encode_assistant_with_tool_calls_falls_back_to_bypass_when_sig_missing(
    provider: GoogleProvider,
) -> None:
    msg = AssistantMessage(
        content="reasoning",
        tool_calls=[ToolCall(id="t1", name="tap", arguments={})],
        finish_reason=FinishReason.TOOL_CALLS,
        # no vendor_extra set
    )

    out = provider._encode_message(msg)

    assert out["tool_calls"][0]["extra_content"] == {
        "google": {"thought_signature": _SIG_BYPASS},
    }


def test_encode_assistant_falls_back_to_bypass_when_sig_value_empty(
    provider: GoogleProvider,
) -> None:
    msg = AssistantMessage(
        content="",
        tool_calls=[ToolCall(id="t1", name="x", arguments={})],
        finish_reason=FinishReason.TOOL_CALLS,
        vendor_extra={"google": {"thought_signature": ""}},
    )

    out = provider._encode_message(msg)

    assert (
        out["tool_calls"][0]["extra_content"]["google"]["thought_signature"]
        == _SIG_BYPASS
    )


def test_encode_tool_result_message_routes_to_helper(
    provider: GoogleProvider,
) -> None:
    out = provider._encode_message(ToolResultMessage(tool_call_id="t1", content="ok"))

    # Helper returns a list (single entry for text-only).
    assert out == [{"role": "tool", "tool_call_id": "t1", "content": "ok"}]


# ---------- _parse_response ----------


def test_parse_response_captures_signature_from_tool_call_extra_content(
    provider: GoogleProvider,
) -> None:
    raw = {
        "choices": [
            {
                "message": {
                    "tool_calls": [
                        {
                            "id": "c1",
                            "function": {"name": "tap", "arguments": "{}"},
                            "extra_content": {"google": {"thought_signature": "sig-A"}},
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
    }

    out = provider._parse_response(raw)

    assert out.vendor_extra["google"]["thought_signature"] == "sig-A"


def test_parse_response_captures_signature_from_message_extra_content(
    provider: GoogleProvider,
) -> None:
    # Text-only response — signature on the message rather than tool_call.
    raw = {
        "choices": [
            {
                "message": {
                    "content": "thinking out loud",
                    "extra_content": {"google": {"thought_signature": "sig-B"}},
                },
                "finish_reason": "stop",
            }
        ],
    }

    out = provider._parse_response(raw)

    assert out.vendor_extra["google"]["thought_signature"] == "sig-B"


def test_parse_response_no_signature_leaves_vendor_extra_empty(
    provider: GoogleProvider,
) -> None:
    raw = {
        "choices": [
            {
                "message": {"content": "x"},
                "finish_reason": "stop",
            }
        ],
    }

    out = provider._parse_response(raw)

    assert "google" not in out.vendor_extra


def test_parse_response_preserves_existing_google_vendor_extra(
    provider: GoogleProvider, mocker
) -> None:
    # super()._parse_response is what builds the AssistantMessage; we
    # verify our override only overlays "thought_signature" without
    # clobbering other vendor_extra keys.
    raw = {
        "choices": [
            {
                "message": {
                    "content": "x",
                    "extra_content": {"google": {"thought_signature": "sig-C"}},
                },
                "finish_reason": "stop",
            }
        ],
    }
    # Patch base parse to inject pre-existing vendor_extra so we can
    # check that setdefault preserves it.
    base_msg = AssistantMessage(
        content="x",
        tool_calls=[],
        finish_reason=FinishReason.STOP,
        vendor_extra={"google": {"other_key": "kept"}},
    )
    mocker.patch(
        "physiclaw.agent.provider.openai_compat.OpenAICompatibleProvider._parse_response",
        return_value=base_msg,
    )

    out = provider._parse_response(raw)

    assert out.vendor_extra["google"]["other_key"] == "kept"
    assert out.vendor_extra["google"]["thought_signature"] == "sig-C"


# ---------- _encode_tool_result ----------


def test_encode_tool_result_text_only_returns_single_role_tool() -> None:
    result = ToolResultMessage(tool_call_id="t1", content="ok")

    out = _encode_tool_result(result)

    assert out == [{"role": "tool", "tool_call_id": "t1", "content": "ok"}]


def test_encode_tool_result_text_block_only_returns_single_entry() -> None:
    result = ToolResultMessage(
        tool_call_id="t1",
        content=[TextBlock(text="caption")],
    )

    out = _encode_tool_result(result)

    assert len(out) == 1
    assert out[0]["role"] == "tool"


def test_encode_tool_result_with_image_splits_into_two_entries() -> None:
    result = ToolResultMessage(
        tool_call_id="t1",
        content=[
            TextBlock(text="caption"),
            ImageBlock(media_type="image/png", data_b64="aGk="),
        ],
    )

    out = _encode_tool_result(result)

    assert len(out) == 2
    # First: role:tool with text content (preserving tool_call_id pairing)
    assert out[0]["role"] == "tool"
    assert out[0]["tool_call_id"] == "t1"
    # Second: synthetic role:user carrying the image parts
    assert out[1]["role"] == "user"
    assert out[1]["content"][0]["type"] == "image_url"


def test_encode_tool_result_image_only_uses_placeholder_text() -> None:
    # When there are images but NO text blocks, the role:tool entry
    # gets a placeholder text instead of an empty content list.
    result = ToolResultMessage(
        tool_call_id="t1",
        content=[ImageBlock(media_type="image/png", data_b64="aGk=")],
    )

    out = _encode_tool_result(result)

    assert len(out) == 2
    assert out[0]["content"] == "(image attached in next message)"
    assert out[1]["role"] == "user"


# ---------- _extract_thought_signature ----------


def test_extract_signature_from_tool_call_extra_content() -> None:
    raw = {
        "choices": [
            {
                "message": {
                    "tool_calls": [
                        {
                            "extra_content": {
                                "google": {"thought_signature": "sig-extra"}
                            },
                        }
                    ],
                },
            }
        ],
    }

    assert _extract_thought_signature(raw) == "sig-extra"


def test_extract_signature_from_tool_call_function_fallback() -> None:
    # Defensive — some shim variants nest under function.
    raw = {
        "choices": [
            {
                "message": {
                    "tool_calls": [
                        {
                            "function": {"thought_signature": "sig-from-fn"},
                        }
                    ],
                },
            }
        ],
    }

    assert _extract_thought_signature(raw) == "sig-from-fn"


def test_extract_signature_from_message_level_when_no_tool_calls() -> None:
    raw = {
        "choices": [
            {
                "message": {
                    "extra_content": {"google": {"thought_signature": "sig-msg"}},
                },
            }
        ],
    }

    assert _extract_thought_signature(raw) == "sig-msg"


def test_extract_signature_returns_none_when_absent() -> None:
    raw = {"choices": [{"message": {"content": "x"}}]}

    assert _extract_thought_signature(raw) is None


def test_extract_signature_returns_none_for_empty_choices() -> None:
    assert _extract_thought_signature({}) is None


def test_extract_signature_falls_through_to_message_level_when_tool_call_paths_empty() -> (
    None
):
    # tool_calls present but neither extra_content nor function carry
    # the signature — falls through to message.extra_content.
    raw = {
        "choices": [
            {
                "message": {
                    "tool_calls": [{"function": {"name": "tap"}}],
                    "extra_content": {"google": {"thought_signature": "msg-sig"}},
                },
            }
        ],
    }

    assert _extract_thought_signature(raw) == "msg-sig"


def test_extract_signature_prefers_extra_content_over_function_fallback() -> None:
    # Both paths populated — extra_content wins.
    raw = {
        "choices": [
            {
                "message": {
                    "tool_calls": [
                        {
                            "extra_content": {
                                "google": {"thought_signature": "primary"}
                            },
                            "function": {"thought_signature": "fallback"},
                        }
                    ],
                },
            }
        ],
    }

    assert _extract_thought_signature(raw) == "primary"
