"""Tests for `physiclaw.provider.wire` — DTO ↔ OpenAI wire format.

Five public functions, two directions:

  - request-side:  `tool_to_wire`
  - DTO → OpenAI:  `assistant_to_wire`, `tool_result_to_wire`,
                   `user_content_to_openai`
  - MCP → DTO:     `mcp_blocks_to_content_blocks` (image-scale step
                   patched to a stub — wire.py's contract is "scaled
                   bytes land in the ImageBlock", not "we know how to
                   scale", which lives in compact.py)

Anchored `match=` regexes used wherever a sibling raise/log message
exists, so a mutmut `XX…XX` wrap can't slip past. Log-message tests
assert via `record.getMessage()` with a known-prefix substring rather
than the raw format string.

Accepted equivalent mutants (cannot be killed by behavior tests):

  - The single-text fast-path predicate
    (`blocks[0].get("type") == "text"`) — mutating either string makes
    the predicate False, but the slow path produces the same joined
    string output for a single text block. Pure optimization, no
    observable difference.
"""

from __future__ import annotations

import json
import logging

import pytest
from hypothesis import given
from hypothesis import strategies as st

from physiclaw.contract.dto import (
    AssistantMessage,
    FinishReason,
    ImageBlock,
    TextBlock,
    ToolCall,
    ToolResultMessage,
)
from physiclaw.provider.wire import (
    assistant_to_wire,
    mcp_blocks_to_content_blocks,
    tool_result_to_wire,
    tool_to_wire,
    user_content_to_openai,
)

# ---------- tool_to_wire ----------


def test_tool_to_wire_full_payload() -> None:
    tool = {
        "name": "tap",
        "description": "Tap a coordinate",
        "input_schema": {"type": "object", "properties": {"bbox": {"type": "array"}}},
    }

    wire = tool_to_wire(tool)

    assert wire == {
        "type": "function",
        "function": {
            "name": "tap",
            "description": "Tap a coordinate",
            "parameters": {
                "type": "object",
                "properties": {"bbox": {"type": "array"}},
            },
        },
    }


def test_tool_to_wire_missing_description_defaults_to_empty_string() -> None:
    wire = tool_to_wire({"name": "x", "input_schema": {"type": "object"}})

    assert wire["function"]["description"] == ""


@pytest.mark.parametrize("missing", [None, {}])
def test_tool_to_wire_missing_or_empty_input_schema_defaults_to_empty_object(
    missing: object,
) -> None:
    # `tool.get("input_schema") or default` — both None and {} are falsy
    # and trigger the default.
    tool: dict = {"name": "x", "description": ""}
    if missing is not None:
        tool["input_schema"] = missing

    wire = tool_to_wire(tool)

    assert wire["function"]["parameters"] == {"type": "object", "properties": {}}


# ---------- assistant_to_wire ----------


def test_assistant_to_wire_text_only_no_tool_calls_omits_tool_calls_key() -> None:
    asst = AssistantMessage(
        content="hello", tool_calls=[], finish_reason=FinishReason.STOP
    )

    wire = assistant_to_wire(asst)

    assert wire == {"role": "assistant", "content": "hello"}


def test_assistant_to_wire_empty_content_becomes_empty_string_not_none() -> None:
    asst = AssistantMessage(content="", tool_calls=[], finish_reason=FinishReason.STOP)

    wire = assistant_to_wire(asst)

    assert wire["content"] == ""


def test_assistant_to_wire_includes_tool_calls_array() -> None:
    asst = AssistantMessage(
        content="thinking…",
        tool_calls=[
            ToolCall(id="call_1", name="tap", arguments={"bbox": [0, 0, 1, 1]}),
            ToolCall(id="call_2", name="peek", arguments={}),
        ],
        finish_reason=FinishReason.TOOL_CALLS,
    )

    wire = assistant_to_wire(asst)

    assert wire["role"] == "assistant"
    assert wire["content"] == "thinking…"
    assert len(wire["tool_calls"]) == 2
    assert wire["tool_calls"][0] == {
        "id": "call_1",
        "type": "function",
        "function": {
            "name": "tap",
            "arguments": json.dumps({"bbox": [0, 0, 1, 1]}, ensure_ascii=False),
        },
    }


def test_assistant_to_wire_tool_arguments_round_trip_via_json() -> None:
    args = {"k": "v", "n": 42, "nested": {"a": [1, 2]}}
    asst = AssistantMessage(
        content="",
        tool_calls=[ToolCall(id="x", name="t", arguments=args)],
        finish_reason=FinishReason.TOOL_CALLS,
    )

    wire = assistant_to_wire(asst)

    assert json.loads(wire["tool_calls"][0]["function"]["arguments"]) == args


def test_assistant_to_wire_tool_arguments_use_ensure_ascii_false() -> None:
    # ensure_ascii=False keeps non-ASCII as-is rather than \uXXXX-escaping;
    # critical for token-cache stability on Chinese/CJK content.
    args = {"text": "你好"}
    asst = AssistantMessage(
        content="",
        tool_calls=[ToolCall(id="x", name="t", arguments=args)],
        finish_reason=FinishReason.TOOL_CALLS,
    )

    wire = assistant_to_wire(asst)

    assert "你好" in wire["tool_calls"][0]["function"]["arguments"]


# ---------- tool_result_to_wire ----------


def test_tool_result_to_wire_string_content_passes_through() -> None:
    result = ToolResultMessage(tool_call_id="call_1", content="ok")

    wire = tool_result_to_wire(result)

    assert wire == {"role": "tool", "tool_call_id": "call_1", "content": "ok"}


def test_tool_result_to_wire_list_content_becomes_multipart_array() -> None:
    result = ToolResultMessage(
        tool_call_id="call_1",
        content=[
            TextBlock(text="caption"),
            ImageBlock(media_type="image/png", data_b64="aGk="),
        ],
    )

    wire = tool_result_to_wire(result)

    assert wire["role"] == "tool"
    assert wire["tool_call_id"] == "call_1"
    assert isinstance(wire["content"], list)
    assert wire["content"][0] == {"type": "text", "text": "caption"}
    assert wire["content"][1]["type"] == "image_url"


# ---------- user_content_to_openai ----------


def test_user_content_to_openai_string_passes_through_unchanged() -> None:
    assert user_content_to_openai("plain text") == "plain text"


def test_user_content_to_openai_textblock_emits_typed_text_part() -> None:
    parts = user_content_to_openai([TextBlock(text="hi")])

    assert parts == [{"type": "text", "text": "hi"}]


def test_user_content_to_openai_imageblock_emits_data_url_with_mime_and_b64() -> None:
    parts = user_content_to_openai(
        [ImageBlock(media_type="image/jpeg", data_b64="aGVsbG8=")]
    )

    assert parts == [
        {
            "type": "image_url",
            "image_url": {"url": "data:image/jpeg;base64,aGVsbG8="},
        }
    ]


def test_user_content_to_openai_mixed_text_and_image_preserves_order() -> None:
    parts = user_content_to_openai(
        [
            TextBlock(text="first"),
            ImageBlock(media_type="image/png", data_b64="aGk="),
            TextBlock(text="last"),
        ]
    )

    assert [p["type"] for p in parts] == ["text", "image_url", "text"]


def test_user_content_to_openai_unknown_block_is_dropped_with_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    class FakeBlock:
        pass

    with caplog.at_level(logging.WARNING, logger="physiclaw.provider.wire"):
        parts = user_content_to_openai([TextBlock(text="ok"), FakeBlock()])  # type: ignore[list-item]

    assert parts == [{"type": "text", "text": "ok"}]
    assert any(
        r.getMessage().startswith("user_content_to_openai: dropping unknown block")
        and "FakeBlock" in r.getMessage()
        for r in caplog.records
    )


def test_user_content_to_openai_all_unknown_blocks_returns_empty_string() -> None:
    class FakeBlock:
        pass

    # `return parts or ""` — empty parts list is falsy, falls back to "".
    result = user_content_to_openai([FakeBlock()])  # type: ignore[list-item]

    assert result == ""


def test_user_content_to_openai_empty_list_returns_empty_string() -> None:
    assert user_content_to_openai([]) == ""


# ---------- mcp_blocks_to_content_blocks ----------


def test_mcp_blocks_single_text_block_returns_bare_string() -> None:
    # Fast path: avoids list-of-blocks overhead for plain-text results.
    out = mcp_blocks_to_content_blocks([{"type": "text", "text": "hello"}])

    assert out == "hello"


def test_mcp_blocks_single_text_block_with_no_text_field_returns_empty_string() -> None:
    out = mcp_blocks_to_content_blocks([{"type": "text"}])

    assert out == ""


def test_mcp_blocks_multiple_text_blocks_joined_by_newline() -> None:
    out = mcp_blocks_to_content_blocks(
        [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]
    )

    assert out == "a\nb"


def test_mcp_blocks_empty_list_returns_empty_marker() -> None:
    out = mcp_blocks_to_content_blocks([])

    assert out == "(empty tool result)"


def test_mcp_blocks_single_image_block_returns_list_with_one_image(
    mocker,
) -> None:
    # Patch scale_image_bytes — the contract under test is "scaled bytes
    # land in the ImageBlock", not the scaler's behavior.
    import base64 as _b64

    scaler = mocker.patch(
        "physiclaw.common.image.scale_image_bytes",
        return_value=(b"scaled", "image/png"),
    )

    out = mcp_blocks_to_content_blocks(
        [{"type": "image", "data": "aGk=", "mime_type": "image/png"}]
    )

    # Scaler must have been called with the actual decoded bytes — kills
    # mutants that pass `None` or skip the decode.
    scaler.assert_called_once_with(b"hi")

    assert isinstance(out, list)
    assert len(out) == 1
    assert isinstance(out[0], ImageBlock)
    assert out[0].media_type == "image/png"
    # Scaled bytes go through base64.b64encode — kills mutants that drop
    # the encode step or null out new_b64.
    assert out[0].data_b64 == _b64.b64encode(b"scaled").decode()


def test_mcp_blocks_text_then_image_returns_list_with_textblock_first(
    mocker,
) -> None:
    mocker.patch(
        "physiclaw.common.image.scale_image_bytes",
        return_value=(b"scaled", "image/jpeg"),
    )

    out = mcp_blocks_to_content_blocks(
        [
            {"type": "text", "text": "caption-1"},
            {"type": "text", "text": "caption-2"},
            {"type": "image", "data": "aGk=", "mime_type": "image/jpeg"},
        ]
    )

    assert isinstance(out, list)
    assert isinstance(out[0], TextBlock)
    assert out[0].text == "caption-1\ncaption-2"
    assert isinstance(out[1], ImageBlock)


def test_mcp_blocks_image_with_no_mime_type_defaults_to_jpeg_on_scale_failure(
    mocker,
) -> None:
    # scaler raises → fallback path uses `b.get("mime_type") or "image/jpeg"`.
    mocker.patch(
        "physiclaw.common.image.scale_image_bytes",
        side_effect=RuntimeError("scaler down"),
    )

    out = mcp_blocks_to_content_blocks([{"type": "image", "data": "aGk="}])

    assert isinstance(out, list)
    assert isinstance(out[0], ImageBlock)
    assert out[0].media_type == "image/jpeg"
    # Original (unscaled) base64 forwarded.
    assert out[0].data_b64 == "aGk="


def test_mcp_blocks_image_scale_failure_logs_exception(
    mocker, caplog: pytest.LogCaptureFixture
) -> None:
    mocker.patch(
        "physiclaw.common.image.scale_image_bytes",
        side_effect=RuntimeError("scaler down"),
    )

    with caplog.at_level(logging.ERROR, logger="physiclaw.provider.wire"):
        mcp_blocks_to_content_blocks([{"type": "image", "data": "aGk="}])

    assert any(
        r.getMessage().startswith("scale failed; forwarding original image")
        for r in caplog.records
    )


def test_mcp_blocks_image_scale_failure_preserves_explicit_mime_type(
    mocker,
) -> None:
    # In the fallback path, mime falls back to b.get("mime_type") — when
    # the block declares a mime, that wins over the "image/jpeg" default.
    mocker.patch(
        "physiclaw.common.image.scale_image_bytes",
        side_effect=RuntimeError("scaler down"),
    )

    out = mcp_blocks_to_content_blocks(
        [{"type": "image", "data": "aGk=", "mime_type": "image/png"}]
    )

    assert isinstance(out, list)
    assert isinstance(out[0], ImageBlock)
    assert out[0].media_type == "image/png"


def test_mcp_blocks_unknown_block_type_silently_dropped(mocker) -> None:
    mocker.patch(
        "physiclaw.common.image.scale_image_bytes",
        return_value=(b"scaled", "image/png"),
    )

    # Unknown kind contributes nothing — outcome is determined by the
    # rest of the blocks.
    out = mcp_blocks_to_content_blocks(
        [{"type": "weird"}, {"type": "text", "text": "kept"}]
    )

    assert out == "kept"


def test_mcp_blocks_multi_block_text_and_no_image_returns_joined_string() -> None:
    # Skips the fast path (len != 1) but no images, so falls through to
    # the joined-string return.
    out = mcp_blocks_to_content_blocks(
        [{"type": "text", "text": "a"}, {"type": "weird"}]
    )

    assert out == "a"


def test_mcp_blocks_text_with_missing_text_field_appends_empty_string() -> None:
    # The slow path's `b.get("text") or ""` — when the field is missing,
    # the result must be empty string concatenation, not a sentinel.
    out = mcp_blocks_to_content_blocks(
        [{"type": "text"}, {"type": "text", "text": "second"}]
    )

    assert out == "\nsecond"


# ---------- Hypothesis ----------


@given(
    args=st.dictionaries(
        st.text(min_size=1, max_size=10),
        st.one_of(
            st.text(max_size=20),
            st.integers(),
            st.booleans(),
            st.none(),
        ),
        max_size=5,
    )
)
def test_assistant_to_wire_tool_arguments_json_round_trip_property(
    args: dict,
) -> None:
    asst = AssistantMessage(
        content="",
        tool_calls=[ToolCall(id="x", name="t", arguments=args)],
        finish_reason=FinishReason.TOOL_CALLS,
    )

    wire = assistant_to_wire(asst)
    parsed = json.loads(wire["tool_calls"][0]["function"]["arguments"])

    assert parsed == args
