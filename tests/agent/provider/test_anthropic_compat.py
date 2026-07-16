"""Tests for `physiclaw.agent.provider.anthropic_compat`.

The provider uses Anthropic's `AsyncAnthropic` SDK directly, so tests
mock the SDK methods rather than HTTP. A small `_TestAnthropic`
subclass declares PROVIDER_ID/BASE_URL to unblock construction; an
autouse fixture injects ANTHROPIC_API_KEY.

Anthropic SDK exception classes (`APIConnectionError`, `APIStatusError`,
etc.) are real classes and constructed with synthetic
`httpx.Request`/`Response` objects so error-mapping branches work.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from anthropic import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    RateLimitError,
)

from physiclaw.agent.engine.dto import (
    AssistantMessage,
    FinishReason,
    ImageBlock,
    SystemMessage,
    TextBlock,
    ToolCall,
    ToolResultMessage,
    Usage,
    UserMessage,
)
from physiclaw.agent.provider.anthropic_compat import (
    _DEFAULT_MAX_TOKENS,
    _STOP_REASON_MAP,
    AnthropicCacheMarkers,
    AnthropicCompatibleProvider,
    _assistant_blocks,
    _content_to_anthropic,
    _extract_system_text,
    _from_anthropic_response,
    _parse_anthropic_usage,
    _tool_to_anthropic,
)
from physiclaw.agent.provider.provider_base import (
    EPHEMERAL_CACHE_CONTROL,
    ProviderPermanentError,
    ProviderTransientError,
)

# ---------- a concrete subclass for testing ----------


class _TestAnthropic(AnthropicCompatibleProvider):
    PROVIDER_ID = "anthropic_test"
    BASE_URL = "https://api.anthropic.test/v1"


@pytest.fixture(autouse=True)
def _stub_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_TEST_API_KEY", "sk-test")


@pytest.fixture
def provider(mocker) -> _TestAnthropic:
    """Provider with a mocked AsyncAnthropic client (no real network)."""
    fake_client = mocker.MagicMock()
    fake_client.close = mocker.AsyncMock()
    fake_client.messages.create = mocker.AsyncMock()
    fake_client.models.list = mocker.AsyncMock()
    mocker.patch("anthropic.AsyncAnthropic", return_value=fake_client)
    return _TestAnthropic(model="claude-test")


# ---------- module constants ----------


def test_default_max_tokens_pinned() -> None:
    assert _DEFAULT_MAX_TOKENS == 8192


@pytest.mark.parametrize(
    "stop_raw, expected",
    [
        ("end_turn", FinishReason.STOP),
        ("stop_sequence", FinishReason.STOP),
        ("max_tokens", FinishReason.LENGTH),
        ("tool_use", FinishReason.TOOL_CALLS),
    ],
)
def test_stop_reason_map_pinned(stop_raw: str, expected: FinishReason) -> None:
    assert _STOP_REASON_MAP[stop_raw] == expected


# ---------- _build_client / aclose / list_models ----------


def test_build_client_uses_async_anthropic_sdk(mocker) -> None:
    fake = mocker.patch("anthropic.AsyncAnthropic")

    _TestAnthropic(model="m", base_url="https://override")

    fake.assert_called_once()
    kwargs = fake.call_args.kwargs
    assert kwargs["api_key"] == "sk-test"
    assert kwargs["base_url"] == "https://override"


def test_build_client_applies_config_base_url_override(mocker) -> None:
    fake = mocker.patch("anthropic.AsyncAnthropic")
    mocker.patch(
        "physiclaw.common.config.provider_base_url_override",
        return_value="https://corp-proxy/anthropic",
    )

    _TestAnthropic(model="m")

    assert fake.call_args.kwargs["base_url"] == "https://corp-proxy/anthropic"


def test_build_client_default_base_url_is_sdk_default_not_class_attr(mocker) -> None:
    """The class BASE_URL carries a `/v1` suffix (base-class non-empty
    check only) and the SDK appends `/v1/...` itself — the default must
    stay None so the SDK uses its own endpoint, never the class attr."""
    fake = mocker.patch("anthropic.AsyncAnthropic")
    mocker.patch(
        "physiclaw.common.config.provider_base_url_override", return_value=None
    )

    _TestAnthropic(model="m")

    assert fake.call_args.kwargs["base_url"] is None


@pytest.mark.asyncio
async def test_aclose_calls_sdk_close_not_aclose(provider: _TestAnthropic) -> None:
    await provider.aclose()

    provider._client.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_list_models_normalizes_each_entry(
    provider: _TestAnthropic, mocker
) -> None:
    provider._client.models.list.return_value = mocker.MagicMock(
        data=[
            mocker.MagicMock(
                id="claude-opus-4-7",
                display_name="Claude Opus 4.7",
                created_at="2026-01-01",
            ),
            mocker.MagicMock(
                id="claude-sonnet-4-6",
                display_name="Claude Sonnet 4.6",
                created_at="2025-09-01",
            ),
        ]
    )

    out = await provider.list_models()

    assert out == [
        {
            "id": "claude-opus-4-7",
            "display_name": "Claude Opus 4.7",
            "created_at": "2026-01-01",
        },
        {
            "id": "claude-sonnet-4-6",
            "display_name": "Claude Sonnet 4.6",
            "created_at": "2025-09-01",
        },
    ]


# ---------- _encode_message ----------


def test_encode_system_message_returns_none(provider: _TestAnthropic) -> None:
    # System rides outside the messages array — `_encode_message` skips it.
    assert provider._encode_message(SystemMessage(content="sys")) is None


def test_encode_user_message_with_string_content(provider: _TestAnthropic) -> None:
    out = provider._encode_message(UserMessage(content="hi"))

    assert out == {"role": "user", "content": "hi"}


def test_encode_user_message_with_block_list(provider: _TestAnthropic) -> None:
    out = provider._encode_message(
        UserMessage(
            content=[
                TextBlock(text="caption"),
                ImageBlock(media_type="image/png", data_b64="aGk="),
            ]
        )
    )

    assert out["role"] == "user"
    assert out["content"][0] == {"type": "text", "text": "caption"}
    assert out["content"][1] == {
        "type": "image",
        "source": {"type": "base64", "media_type": "image/png", "data": "aGk="},
    }


def test_encode_assistant_message_text_only(provider: _TestAnthropic) -> None:
    out = provider._encode_message(
        AssistantMessage(
            content="hello", tool_calls=[], finish_reason=FinishReason.STOP
        )
    )

    assert out == {"role": "assistant", "content": [{"type": "text", "text": "hello"}]}


def test_encode_assistant_message_with_tool_calls(provider: _TestAnthropic) -> None:
    out = provider._encode_message(
        AssistantMessage(
            content="thinking",
            tool_calls=[ToolCall(id="t1", name="tap", arguments={"x": 0.5})],
            finish_reason=FinishReason.TOOL_CALLS,
        )
    )

    blocks = out["content"]
    assert blocks[0] == {"type": "text", "text": "thinking"}
    assert blocks[1] == {
        "type": "tool_use",
        "id": "t1",
        "name": "tap",
        "input": {"x": 0.5},
    }


def test_encode_tool_result_message_wraps_in_tool_result_block(
    provider: _TestAnthropic,
) -> None:
    out = provider._encode_message(ToolResultMessage(tool_call_id="t1", content="ok"))

    assert out == {
        "role": "user",
        "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "ok"}],
    }


def test_encode_unknown_message_type_returns_none_with_warning(
    provider: _TestAnthropic, caplog: pytest.LogCaptureFixture
) -> None:
    import logging

    class _Weird:
        pass

    with caplog.at_level(
        logging.WARNING, logger="physiclaw.agent.provider.anthropic_compat"
    ):
        out = provider._encode_message(_Weird())  # type: ignore[arg-type]

    assert out is None
    assert any(
        r.getMessage().startswith("anthropic: dropping unknown message type")
        for r in caplog.records
    )


# ---------- CACHE_MARKERS factory ----------


def test_cache_markers_factory_is_anthropic_shape() -> None:
    assert isinstance(AnthropicCompatibleProvider.CACHE_MARKERS, AnthropicCacheMarkers)


def test_cache_markers_mark_stub_attaches_ephemeral_cache_control_to_inner_block(
    provider: _TestAnthropic,
) -> None:
    entry = {
        "role": "user",
        "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "stale"}],
    }

    out = provider.CACHE_MARKERS.mark_stub(entry)

    assert out["content"][0]["cache_control"] == EPHEMERAL_CACHE_CONTROL
    # Original entry untouched (shallow copy).
    assert "cache_control" not in entry["content"][0]


def test_cache_markers_mark_system_is_identity() -> None:
    # System rides outside the messages array — marked in chat(), not here.
    entry = {"role": "system", "content": "sys"}

    assert AnthropicCacheMarkers().mark_system(entry) is entry


# ---------- chat: payload construction ----------


@pytest.mark.asyncio
async def test_chat_constructs_payload_with_model_max_tokens_messages(
    provider: _TestAnthropic, mocker
) -> None:
    fake_resp = SimpleNamespace(
        content=[SimpleNamespace(type="text", text="hi", id=None)],
        stop_reason="end_turn",
        usage=None,
        model_dump=lambda: {"id": "m1"},
    )
    provider._client.messages.create.return_value = fake_resp

    await provider.chat([UserMessage(content="ping")], tools=[])

    payload = provider._client.messages.create.call_args.kwargs
    assert payload["model"] == "claude-test"
    assert payload["max_tokens"] == _DEFAULT_MAX_TOKENS
    assert payload["messages"] == [{"role": "user", "content": "ping"}]


@pytest.mark.asyncio
async def test_chat_payload_includes_system_with_cache_control_when_present(
    provider: _TestAnthropic,
) -> None:
    fake_resp = SimpleNamespace(
        content=[],
        stop_reason="end_turn",
        usage=None,
        model_dump=lambda: {},
    )
    provider._client.messages.create.return_value = fake_resp

    await provider.chat(
        [SystemMessage(content="be helpful"), UserMessage(content="hi")],
        tools=[],
    )

    payload = provider._client.messages.create.call_args.kwargs
    assert payload["system"] == [
        {
            "type": "text",
            "text": "be helpful",
            "cache_control": EPHEMERAL_CACHE_CONTROL,
        }
    ]


@pytest.mark.asyncio
async def test_chat_payload_omits_system_when_no_system_messages(
    provider: _TestAnthropic,
) -> None:
    fake_resp = SimpleNamespace(
        content=[],
        stop_reason="end_turn",
        usage=None,
        model_dump=lambda: {},
    )
    provider._client.messages.create.return_value = fake_resp

    await provider.chat([UserMessage(content="hi")], tools=[])

    payload = provider._client.messages.create.call_args.kwargs
    assert "system" not in payload


@pytest.mark.asyncio
async def test_chat_payload_includes_tools_when_present(
    provider: _TestAnthropic,
) -> None:
    fake_resp = SimpleNamespace(
        content=[],
        stop_reason="end_turn",
        usage=None,
        model_dump=lambda: {},
    )
    provider._client.messages.create.return_value = fake_resp

    await provider.chat(
        [UserMessage(content="hi")],
        tools=[
            {"name": "tap", "description": "Tap", "input_schema": {"type": "object"}}
        ],
    )

    payload = provider._client.messages.create.call_args.kwargs
    assert payload["tools"] == [
        {
            "name": "tap",
            "description": "Tap",
            "input_schema": {"type": "object"},
        }
    ]


@pytest.mark.asyncio
async def test_chat_payload_omits_tools_when_empty(
    provider: _TestAnthropic,
) -> None:
    fake_resp = SimpleNamespace(
        content=[],
        stop_reason="end_turn",
        usage=None,
        model_dump=lambda: {},
    )
    provider._client.messages.create.return_value = fake_resp

    await provider.chat([UserMessage(content="hi")], tools=[])

    payload = provider._client.messages.create.call_args.kwargs
    assert "tools" not in payload


# ---------- chat: error mapping ----------


def _conn_err() -> APIConnectionError:
    req = httpx.Request("POST", "https://x")
    return APIConnectionError(message="boom", request=req)


def _timeout_err() -> APITimeoutError:
    return APITimeoutError(httpx.Request("POST", "https://x"))


def _rate_limit_err() -> RateLimitError:
    req = httpx.Request("POST", "https://x")
    resp = httpx.Response(429, request=req, content=b'{"error":"limited"}')
    return RateLimitError(message="rate limited", response=resp, body={})


def _status_err(status: int, message: str = "boom") -> APIStatusError:
    req = httpx.Request("POST", "https://x")
    resp = httpx.Response(status, request=req, content=b"{}")
    return APIStatusError(message=message, response=resp, body=None)


@pytest.mark.asyncio
async def test_chat_maps_api_connection_error_to_transient(
    provider: _TestAnthropic,
) -> None:
    provider._client.messages.create.side_effect = _conn_err()

    with pytest.raises(ProviderTransientError, match=r"^transport: "):
        await provider.chat([UserMessage(content="x")], [])


@pytest.mark.asyncio
async def test_chat_maps_api_timeout_to_transient(provider: _TestAnthropic) -> None:
    provider._client.messages.create.side_effect = _timeout_err()

    with pytest.raises(ProviderTransientError, match=r"^transport: "):
        await provider.chat([UserMessage(content="x")], [])


@pytest.mark.asyncio
async def test_chat_maps_rate_limit_to_transient(provider: _TestAnthropic) -> None:
    provider._client.messages.create.side_effect = _rate_limit_err()

    with pytest.raises(ProviderTransientError, match=r"^rate limited: "):
        await provider.chat([UserMessage(content="x")], [])


@pytest.mark.asyncio
async def test_chat_maps_5xx_status_to_transient(
    provider: _TestAnthropic, caplog: pytest.LogCaptureFixture
) -> None:
    import logging

    provider._client.messages.create.side_effect = _status_err(
        503, "service unavailable"
    )

    with caplog.at_level(
        logging.WARNING, logger="physiclaw.agent.provider.anthropic_compat"
    ):
        with pytest.raises(ProviderTransientError, match=r"^HTTP 503: "):
            await provider.chat([UserMessage(content="x")], [])

    assert any(
        "anthropic HTTP 503 (transient)" in r.getMessage() for r in caplog.records
    )


@pytest.mark.asyncio
async def test_chat_maps_4xx_status_to_permanent(
    provider: _TestAnthropic, caplog: pytest.LogCaptureFixture
) -> None:
    import logging

    provider._client.messages.create.side_effect = _status_err(400, "bad model")

    with caplog.at_level(
        logging.ERROR, logger="physiclaw.agent.provider.anthropic_compat"
    ):
        with pytest.raises(ProviderPermanentError, match=r"^HTTP 400: "):
            await provider.chat([UserMessage(content="x")], [])

    assert any(
        "anthropic HTTP 400 (permanent)" in r.getMessage() for r in caplog.records
    )


@pytest.mark.asyncio
async def test_chat_returns_assistant_message_on_success(
    provider: _TestAnthropic,
) -> None:
    fake_resp = SimpleNamespace(
        content=[SimpleNamespace(type="text", text="hello", id=None)],
        stop_reason="end_turn",
        usage=SimpleNamespace(
            input_tokens=10,
            output_tokens=20,
            cache_read_input_tokens=0,
            cache_creation_input_tokens=0,
        ),
        model_dump=lambda: {"raw": True},
    )
    provider._client.messages.create.return_value = fake_resp

    result = await provider.chat([UserMessage(content="hi")], tools=[])

    assert isinstance(result, AssistantMessage)
    assert result.content == "hello"
    assert result.finish_reason == FinishReason.STOP
    assert result.usage.completion_tokens == 20


# ---------- _extract_system_text ----------


def test_extract_system_text_concatenates_with_double_newline() -> None:
    out = _extract_system_text(
        [
            SystemMessage(content="a"),
            UserMessage(content="x"),
            SystemMessage(content="b"),
        ]
    )

    assert out == "a\n\nb"


def test_extract_system_text_returns_empty_when_no_systems() -> None:
    assert _extract_system_text([UserMessage(content="x")]) == ""


def test_extract_system_text_skips_empty_system_content() -> None:
    out = _extract_system_text(
        [SystemMessage(content=""), SystemMessage(content="real")]
    )

    assert out == "real"


# ---------- _content_to_anthropic ----------


def test_content_to_anthropic_string_passes_through() -> None:
    assert _content_to_anthropic("hello") == "hello"


def test_content_to_anthropic_text_block_emits_text_dict() -> None:
    assert _content_to_anthropic([TextBlock(text="x")]) == [
        {"type": "text", "text": "x"}
    ]


def test_content_to_anthropic_image_block_emits_base64_source() -> None:
    out = _content_to_anthropic([ImageBlock(media_type="image/jpeg", data_b64="aGk=")])

    assert out == [
        {
            "type": "image",
            "source": {"type": "base64", "media_type": "image/jpeg", "data": "aGk="},
        }
    ]


def test_content_to_anthropic_drops_unknown_block_with_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    import logging

    class _Weird:
        pass

    # Block dispatch is shared — the warning is emitted by
    # `wire.encode_content`, tagged with the anthropic label.
    with caplog.at_level(logging.WARNING, logger="physiclaw.agent.provider.wire"):
        out = _content_to_anthropic([_Weird()])  # type: ignore[list-item]

    # Falls back to single empty-text block since list became empty.
    assert out == [{"type": "text", "text": ""}]
    assert any(
        r.getMessage().startswith("anthropic: dropping unknown block")
        and "_Weird" in r.getMessage()
        for r in caplog.records
    )


def test_content_to_anthropic_non_str_non_list_falls_back_to_str() -> None:
    assert _content_to_anthropic(42) == "42"


def test_content_to_anthropic_none_falls_back_to_empty_string() -> None:
    assert _content_to_anthropic(None) == ""


# ---------- _assistant_blocks ----------


def test_assistant_blocks_text_only() -> None:
    out = _assistant_blocks(
        AssistantMessage(content="hi", tool_calls=[], finish_reason=FinishReason.STOP)
    )

    assert out == [{"type": "text", "text": "hi"}]


def test_assistant_blocks_with_tool_calls() -> None:
    out = _assistant_blocks(
        AssistantMessage(
            content="reasoning",
            tool_calls=[ToolCall(id="t1", name="tap", arguments={"x": 1})],
            finish_reason=FinishReason.TOOL_CALLS,
        )
    )

    assert out == [
        {"type": "text", "text": "reasoning"},
        {"type": "tool_use", "id": "t1", "name": "tap", "input": {"x": 1}},
    ]


def test_assistant_blocks_empty_falls_back_to_empty_text(mocker) -> None:
    # Anthropic rejects empty assistant content arrays.
    out = _assistant_blocks(
        AssistantMessage(
            content="",
            tool_calls=[],
            finish_reason=FinishReason.STOP,
        )
    )

    assert out == [{"type": "text", "text": ""}]


def test_assistant_blocks_generates_id_when_tool_call_id_empty(mocker) -> None:
    mocker.patch(
        "physiclaw.agent.provider.anthropic_compat.uuid.uuid4",
        return_value=mocker.MagicMock(hex="abcdef1234567890"),
    )

    out = _assistant_blocks(
        AssistantMessage(
            content="",
            tool_calls=[ToolCall(id="", name="t", arguments={})],
            finish_reason=FinishReason.TOOL_CALLS,
        )
    )

    assert out[0]["id"] == "auto_abcdef12"


# ---------- _tool_to_anthropic ----------


def test_tool_to_anthropic_full_payload() -> None:
    out = _tool_to_anthropic(
        {
            "name": "tap",
            "description": "Tap a coord",
            "input_schema": {"type": "object", "properties": {}},
        }
    )

    assert out == {
        "name": "tap",
        "description": "Tap a coord",
        "input_schema": {"type": "object", "properties": {}},
    }


def test_tool_to_anthropic_missing_description_defaults_empty() -> None:
    out = _tool_to_anthropic({"name": "tap"})

    assert out["description"] == ""


@pytest.mark.parametrize("missing", [None, {}])
def test_tool_to_anthropic_missing_input_schema_defaults_to_empty_object(
    missing,
) -> None:
    tool = {"name": "x"}
    if missing is not None:
        tool["input_schema"] = missing

    out = _tool_to_anthropic(tool)

    assert out["input_schema"] == {"type": "object", "properties": {}}


# ---------- _from_anthropic_response ----------


def test_from_response_extracts_text_blocks_joined_by_newline() -> None:
    resp = SimpleNamespace(
        content=[
            SimpleNamespace(type="text", text="hello"),
            SimpleNamespace(type="text", text="world"),
        ],
        stop_reason="end_turn",
        usage=None,
        model_dump=lambda: {},
    )

    out = _from_anthropic_response(resp)

    assert out.content == "hello\nworld"
    assert out.tool_calls == []


def test_from_response_drops_thinking_blocks_from_content() -> None:
    # `thinking` blocks are stripped — only text + tool_use survive.
    resp = SimpleNamespace(
        content=[
            SimpleNamespace(type="thinking", thinking="internal"),
            SimpleNamespace(type="text", text="public"),
        ],
        stop_reason="end_turn",
        usage=None,
        model_dump=lambda: {},
    )

    out = _from_anthropic_response(resp)

    assert out.content == "public"


def test_from_response_extracts_tool_use_blocks() -> None:
    resp = SimpleNamespace(
        content=[
            SimpleNamespace(type="tool_use", id="t1", name="tap", input={"x": 1}),
        ],
        stop_reason="tool_use",
        usage=None,
        model_dump=lambda: {},
    )

    out = _from_anthropic_response(resp)

    assert out.tool_calls == [ToolCall(id="t1", name="tap", arguments={"x": 1})]
    assert out.finish_reason == FinishReason.TOOL_CALLS


def test_from_response_uses_default_finish_reason_when_stop_unknown() -> None:
    resp = SimpleNamespace(
        content=[],
        stop_reason="future_unknown_reason",
        usage=None,
        model_dump=lambda: {},
    )

    out = _from_anthropic_response(resp)

    assert out.finish_reason == FinishReason.STOP


def test_from_response_defaults_stop_reason_to_end_turn_when_none() -> None:
    resp = SimpleNamespace(
        content=[],
        stop_reason=None,
        usage=None,
        model_dump=lambda: {},
    )

    out = _from_anthropic_response(resp)

    assert out.finish_reason == FinishReason.STOP


def test_from_response_falls_back_to_dict_when_no_model_dump() -> None:
    # `dict(resp)` path — used when the SDK returns something that
    # isn't a Pydantic model (e.g. a stub in tests).
    class _DictableResp:
        def __init__(self) -> None:
            self.content = []
            self.stop_reason = "end_turn"
            self.usage = None

        def keys(self) -> list[str]:
            return ["content", "stop_reason", "usage"]

        def __getitem__(self, key: str) -> Any:
            return getattr(self, key)

    out = _from_anthropic_response(_DictableResp())

    assert isinstance(out.raw, dict)


def test_from_response_generates_id_for_tool_use_block_with_no_id(mocker) -> None:
    mocker.patch(
        "physiclaw.agent.provider.anthropic_compat.uuid.uuid4",
        return_value=mocker.MagicMock(hex="abcdef1234567890"),
    )
    resp = SimpleNamespace(
        content=[SimpleNamespace(type="tool_use", id=None, name="tap", input={})],
        stop_reason="tool_use",
        usage=None,
        model_dump=lambda: {},
    )

    out = _from_anthropic_response(resp)

    assert out.tool_calls[0].id == "auto_abcdef12"


# ---------- _parse_anthropic_usage ----------


def test_parse_usage_returns_zero_filled_when_resp_has_no_usage() -> None:
    resp = SimpleNamespace(usage=None)

    assert _parse_anthropic_usage(resp) == Usage()


def test_parse_usage_sums_fresh_cached_created_into_prompt_tokens() -> None:
    resp = SimpleNamespace(
        usage=SimpleNamespace(
            input_tokens=10,
            output_tokens=20,
            cache_read_input_tokens=100,
            cache_creation_input_tokens=5,
        )
    )

    out = _parse_anthropic_usage(resp)

    # prompt_tokens = fresh + cached + created = 10 + 100 + 5
    assert out.prompt_tokens == 115
    assert out.completion_tokens == 20
    assert out.cached_tokens == 100
    assert out.cache_creation_tokens == 5


def test_parse_usage_handles_missing_attrs_as_zero() -> None:
    out = _parse_anthropic_usage(SimpleNamespace(usage=SimpleNamespace()))

    assert out == Usage()


def test_parse_usage_handles_none_attribute_values_as_zero() -> None:
    out = _parse_anthropic_usage(
        SimpleNamespace(
            usage=SimpleNamespace(
                input_tokens=None,
                output_tokens=None,
                cache_read_input_tokens=None,
                cache_creation_input_tokens=None,
            )
        )
    )

    assert out == Usage()
