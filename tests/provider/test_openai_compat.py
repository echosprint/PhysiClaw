"""Tests for `physiclaw.provider.openai_compat`.

The provider uses `httpx.AsyncClient` (inherited from BaseProvider)
to POST to `/chat/completions`. Tests use `respx` to fake the HTTP
layer, which is HTTPX-native and avoids monkeypatching internals.

A small `_TestOpenAI(OpenAICompatibleProvider)` subclass is used
throughout; an autouse fixture supplies the API key.
"""

from __future__ import annotations

import httpx
import pytest
import respx

from physiclaw.contract.dto import (
    AssistantMessage,
    FinishReason,
    SystemMessage,
    TextBlock,
    ToolCall,
    ToolResultMessage,
    Usage,
    UserMessage,
)
from physiclaw.provider.openai_compat import (
    OpenAICacheMarkers,
    OpenAICompatibleProvider,
    _normalize_finish,
    _with_cache_marker,
)
from physiclaw.provider.provider_base import (
    EPHEMERAL_CACHE_CONTROL,
    ProviderPermanentError,
    ProviderTransientError,
)

# ---------- a concrete subclass for testing ----------


class _TestOpenAI(OpenAICompatibleProvider):
    PROVIDER_ID = "openai_test"
    BASE_URL = "https://api.openai.test/v1"


@pytest.fixture(autouse=True)
def _stub_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_TEST_API_KEY", "sk-test")


@pytest.fixture
def provider() -> _TestOpenAI:
    return _TestOpenAI(model="gpt-test")


# ---------- _with_cache_marker ----------


def test_with_cache_marker_wraps_content_in_text_block_with_ephemeral_marker() -> None:
    entry = {"role": "system", "content": "you are helpful"}

    out = _with_cache_marker(entry)

    assert out == {
        "role": "system",
        "content": [
            {
                "type": "text",
                "text": "you are helpful",
                "cache_control": EPHEMERAL_CACHE_CONTROL,
            }
        ],
    }


def test_with_cache_marker_does_not_mutate_original_entry() -> None:
    entry = {"role": "system", "content": "x"}

    _with_cache_marker(entry)

    assert entry["content"] == "x"


# ---------- _normalize_finish ----------


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("stop", FinishReason.STOP),
        ("length", FinishReason.LENGTH),
        ("tool_calls", FinishReason.TOOL_CALLS),
        ("content_filter", FinishReason.CONTENT_FILTER),
        # function_call (legacy OpenAI) maps to TOOL_CALLS.
        ("function_call", FinishReason.TOOL_CALLS),
    ],
)
def test_normalize_finish_known_values(raw: str, expected: FinishReason) -> None:
    assert _normalize_finish(raw) == expected


def test_normalize_finish_unknown_value_falls_back_to_stop_with_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    import logging

    with caplog.at_level(logging.WARNING, logger="physiclaw.provider.openai_compat"):
        out = _normalize_finish("future_unknown_reason")

    assert out == FinishReason.STOP
    assert any("unknown finish_reason" in r.getMessage() for r in caplog.records)


# ---------- _encode_message ----------


def test_encode_system_message(provider: _TestOpenAI) -> None:
    out = provider._encode_message(SystemMessage(content="hi"))

    assert out == {"role": "system", "content": "hi"}


def test_encode_user_message_string_content(provider: _TestOpenAI) -> None:
    out = provider._encode_message(UserMessage(content="ping"))

    assert out == {"role": "user", "content": "ping"}


def test_encode_user_message_with_block_list_routed_through_user_content(
    provider: _TestOpenAI,
) -> None:
    out = provider._encode_message(UserMessage(content=[TextBlock(text="hi")]))

    assert out["role"] == "user"
    assert out["content"] == [{"type": "text", "text": "hi"}]


def test_encode_assistant_message_routes_through_assistant_to_wire(
    provider: _TestOpenAI,
) -> None:
    out = provider._encode_message(
        AssistantMessage(
            content="ack",
            tool_calls=[],
            finish_reason=FinishReason.STOP,
        )
    )

    assert out == {"role": "assistant", "content": "ack"}


def test_encode_tool_result_message_routes_through_tool_result_to_wire(
    provider: _TestOpenAI,
) -> None:
    out = provider._encode_message(ToolResultMessage(tool_call_id="t1", content="ok"))

    assert out == {"role": "tool", "tool_call_id": "t1", "content": "ok"}


def test_encode_unknown_message_subtype_raises_assert_never(
    provider: _TestOpenAI,
) -> None:
    class _Weird:  # not in the Message Union
        pass

    with pytest.raises(AssertionError):
        provider._encode_message(_Weird())  # type: ignore[arg-type]


# ---------- CACHE_MARKERS factory ----------


def test_cache_markers_factory_is_openai_shape() -> None:
    assert isinstance(OpenAICompatibleProvider.CACHE_MARKERS, OpenAICacheMarkers)


def test_cache_markers_mark_system_delegates_to_with_cache_marker(
    provider: _TestOpenAI,
) -> None:
    entry = {"role": "system", "content": "sys"}

    out = provider.CACHE_MARKERS.mark_system(entry)

    assert out["content"][0]["cache_control"] == EPHEMERAL_CACHE_CONTROL


def test_cache_markers_mark_stub_adds_only_cache_control_to_block(
    provider: _TestOpenAI,
) -> None:
    # Stubs arrive as a one-element text-block list (guaranteed by
    # `_encode_message`); the marker must be additive so marked and
    # unmarked stubs serialize identically apart from the key.
    entry = {
        "role": "tool",
        "tool_call_id": "t1",
        "content": [{"type": "text", "text": "(superseded peek) stale"}],
    }

    out = provider.CACHE_MARKERS.mark_stub(entry)

    assert out == {
        "role": "tool",
        "tool_call_id": "t1",
        "content": [
            {
                "type": "text",
                "text": "(superseded peek) stale",
                "cache_control": EPHEMERAL_CACHE_CONTROL,
            }
        ],
    }


def test_cache_markers_mark_stub_does_not_mutate_original_entry(
    provider: _TestOpenAI,
) -> None:
    entry = {
        "role": "tool",
        "tool_call_id": "t1",
        "content": [{"type": "text", "text": "stale"}],
    }

    provider.CACHE_MARKERS.mark_stub(entry)

    assert entry["content"] == [{"type": "text", "text": "stale"}]


def test_encode_superseded_tool_result_wraps_string_in_text_block(
    provider: _TestOpenAI,
) -> None:
    out = provider._encode_message(
        ToolResultMessage(
            tool_call_id="t1", content="(superseded tap) gone", is_superseded=True
        )
    )

    assert out == {
        "role": "tool",
        "tool_call_id": "t1",
        "content": [{"type": "text", "text": "(superseded tap) gone"}],
    }


def test_superseded_stub_serializes_byte_identical_marked_vs_unmarked(
    provider: _TestOpenAI,
) -> None:
    """The invariant this shape exists for: when a newer stub takes the
    moving anchor, the previous stub's wire bytes must be unchanged
    except for losing the `cache_control` key — the contract vendors
    with strict anchor semantics depend on, and the same one the
    Anthropic shape keeps by construction."""
    sys_ = SystemMessage(content="sys")
    s1 = ToolResultMessage(
        tool_call_id="t1", content="(superseded peek) a", is_superseded=True
    )
    s2 = ToolResultMessage(
        tool_call_id="t2", content="(superseded tap) b", is_superseded=True
    )

    as_anchor = provider.serialize_history([sys_, s1])[1]  # s1 marked
    after_move = provider.serialize_history([sys_, s1, s2])[1]  # s1 unmarked

    stripped_blocks = []
    for block in as_anchor["content"]:
        block = dict(block)
        block.pop("cache_control", None)
        stripped_blocks.append(block)

    assert {**as_anchor, "content": stripped_blocks} == after_move


# ---------- chat: success path ----------


@pytest.mark.asyncio
async def test_chat_posts_to_chat_completions_with_payload(
    provider: _TestOpenAI, respx_mock: respx.MockRouter
) -> None:
    route = respx_mock.post("https://api.openai.test/v1/chat/completions").respond(
        json={
            "choices": [{"message": {"content": "hi"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        },
    )

    await provider.chat([UserMessage(content="ping")], tools=[])

    assert route.call_count == 1
    payload = route.calls.last.request.read().decode()
    import json

    body = json.loads(payload)
    assert body["model"] == "gpt-test"
    assert body["messages"] == [{"role": "user", "content": "ping"}]
    assert "tools" not in body


@pytest.mark.asyncio
async def test_chat_payload_includes_tools_and_tool_choice_when_present(
    provider: _TestOpenAI, respx_mock: respx.MockRouter
) -> None:
    route = respx_mock.post("https://api.openai.test/v1/chat/completions").respond(
        json={
            "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
            "usage": {},
        },
    )

    await provider.chat(
        [UserMessage(content="x")],
        tools=[
            {"name": "tap", "description": "Tap", "input_schema": {"type": "object"}}
        ],
    )

    import json

    body = json.loads(route.calls.last.request.read())
    assert body["tools"] == [
        {
            "type": "function",
            "function": {
                "name": "tap",
                "description": "Tap",
                "parameters": {"type": "object"},
            },
        }
    ]
    assert body["tool_choice"] == "auto"


@pytest.mark.asyncio
async def test_chat_returns_assistant_message_on_success(
    provider: _TestOpenAI, respx_mock: respx.MockRouter
) -> None:
    respx_mock.post("https://api.openai.test/v1/chat/completions").respond(
        json={
            "choices": [
                {
                    "message": {"content": "hello"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 7, "completion_tokens": 3},
        },
    )

    out = await provider.chat([UserMessage(content="hi")], tools=[])

    assert isinstance(out, AssistantMessage)
    assert out.content == "hello"
    assert out.finish_reason == FinishReason.STOP
    assert out.usage.prompt_tokens == 7


# ---------- chat: error mapping ----------


@pytest.mark.asyncio
async def test_chat_maps_transport_error_to_transient(
    provider: _TestOpenAI, respx_mock: respx.MockRouter
) -> None:
    respx_mock.post("https://api.openai.test/v1/chat/completions").mock(
        side_effect=httpx.ConnectError("conn refused")
    )

    with pytest.raises(ProviderTransientError, match=r"^transport: "):
        await provider.chat([UserMessage(content="x")], [])


@pytest.mark.asyncio
async def test_chat_maps_timeout_to_transient(
    provider: _TestOpenAI, respx_mock: respx.MockRouter
) -> None:
    respx_mock.post("https://api.openai.test/v1/chat/completions").mock(
        side_effect=httpx.ReadTimeout("timeout")
    )

    with pytest.raises(ProviderTransientError, match=r"^transport: "):
        await provider.chat([UserMessage(content="x")], [])


@pytest.mark.asyncio
async def test_chat_maps_429_to_transient(
    provider: _TestOpenAI,
    respx_mock: respx.MockRouter,
    caplog: pytest.LogCaptureFixture,
) -> None:
    import logging

    respx_mock.post("https://api.openai.test/v1/chat/completions").respond(
        429,
        json={"error": "rate limited"},
    )

    with caplog.at_level(logging.WARNING, logger="physiclaw.provider.openai_compat"):
        with pytest.raises(ProviderTransientError, match=r"^HTTP 429: "):
            await provider.chat([UserMessage(content="x")], [])

    assert any(
        "provider HTTP 429 (transient)" in r.getMessage() for r in caplog.records
    )


@pytest.mark.asyncio
async def test_chat_maps_500_to_transient(
    provider: _TestOpenAI, respx_mock: respx.MockRouter
) -> None:
    respx_mock.post("https://api.openai.test/v1/chat/completions").respond(
        500,
        json={"error": "boom"},
    )

    with pytest.raises(ProviderTransientError, match=r"^HTTP 500: "):
        await provider.chat([UserMessage(content="x")], [])


@pytest.mark.asyncio
async def test_chat_maps_400_to_permanent(
    provider: _TestOpenAI,
    respx_mock: respx.MockRouter,
    caplog: pytest.LogCaptureFixture,
) -> None:
    import logging

    respx_mock.post("https://api.openai.test/v1/chat/completions").respond(
        400,
        json={"error": "bad model"},
    )

    with caplog.at_level(logging.ERROR, logger="physiclaw.provider.openai_compat"):
        with pytest.raises(ProviderPermanentError, match=r"^HTTP 400: "):
            await provider.chat([UserMessage(content="x")], [])

    assert any(
        "provider HTTP 400 (permanent)" in r.getMessage() for r in caplog.records
    )


# ---------- list_models ----------


@pytest.mark.asyncio
async def test_list_models_returns_data_array(
    provider: _TestOpenAI, respx_mock: respx.MockRouter
) -> None:
    respx_mock.get("https://api.openai.test/v1/models").respond(
        json={
            "object": "list",
            "data": [
                {"id": "gpt-5", "object": "model"},
                {"id": "gpt-4", "object": "model"},
            ],
        },
    )

    out = await provider.list_models()

    assert out == [
        {"id": "gpt-5", "object": "model"},
        {"id": "gpt-4", "object": "model"},
    ]


@pytest.mark.asyncio
async def test_list_models_returns_empty_when_data_missing(
    provider: _TestOpenAI, respx_mock: respx.MockRouter
) -> None:
    respx_mock.get("https://api.openai.test/v1/models").respond(json={})

    assert await provider.list_models() == []


@pytest.mark.asyncio
async def test_list_models_maps_transport_to_transient(
    provider: _TestOpenAI, respx_mock: respx.MockRouter
) -> None:
    respx_mock.get("https://api.openai.test/v1/models").mock(
        side_effect=httpx.ConnectError("nope")
    )

    with pytest.raises(ProviderTransientError, match=r"^transport: "):
        await provider.list_models()


@pytest.mark.asyncio
async def test_list_models_maps_4xx_to_permanent(
    provider: _TestOpenAI, respx_mock: respx.MockRouter
) -> None:
    respx_mock.get("https://api.openai.test/v1/models").respond(
        401,
        json={"error": "unauthorized"},
    )

    with pytest.raises(ProviderPermanentError, match=r"^HTTP 401: "):
        await provider.list_models()


# ---------- _parse_response ----------


def test_parse_response_text_only(provider: _TestOpenAI) -> None:
    raw = {
        "choices": [{"message": {"content": "hi"}, "finish_reason": "stop"}],
    }

    out = provider._parse_response(raw)

    assert out.content == "hi"
    assert out.tool_calls == []
    assert out.finish_reason == FinishReason.STOP


def test_parse_response_non_string_content_serialized_to_json(
    provider: _TestOpenAI,
) -> None:
    raw = {
        "choices": [
            {
                "message": {"content": [{"type": "text", "text": "x"}]},
                "finish_reason": "stop",
            }
        ],
    }

    out = provider._parse_response(raw)

    # Serialized to compact JSON; bytes preserved.
    assert "x" in out.content
    assert out.content.startswith("[")


def test_parse_response_finish_reason_default_stop_when_missing(
    provider: _TestOpenAI,
) -> None:
    raw = {"choices": [{"message": {"content": "x"}}]}

    out = provider._parse_response(raw)

    assert out.finish_reason == FinishReason.STOP


def test_parse_response_extracts_tool_calls(provider: _TestOpenAI) -> None:
    raw = {
        "choices": [
            {
                "message": {
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "function": {"name": "tap", "arguments": '{"x": 0.5}'},
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
    }

    out = provider._parse_response(raw)

    assert out.tool_calls == [ToolCall(id="call_1", name="tap", arguments={"x": 0.5})]
    assert out.finish_reason == FinishReason.TOOL_CALLS


def test_parse_response_tool_call_arguments_dict_passthrough(
    provider: _TestOpenAI,
) -> None:
    # When `arguments` is already a dict (some non-strict providers),
    # it's passed through without JSON parsing.
    raw = {
        "choices": [
            {
                "message": {
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "c1",
                            "function": {"name": "tap", "arguments": {"x": 1}},
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
    }

    out = provider._parse_response(raw)

    assert out.tool_calls[0].arguments == {"x": 1}


def test_parse_response_tool_call_with_non_dict_args_wrapped_in_raw(
    provider: _TestOpenAI,
) -> None:
    # If arguments JSON parses to a list (or other non-dict), wrap so
    # the validator still sees a dict shape.
    raw = {
        "choices": [
            {
                "message": {
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "c1",
                            "function": {"name": "x", "arguments": "[1, 2, 3]"},
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
    }

    out = provider._parse_response(raw)

    assert out.tool_calls[0].arguments == {"_raw": [1, 2, 3]}


def test_parse_response_tool_call_with_malformed_json_passes_through(
    provider: _TestOpenAI,
) -> None:
    # Per principle 4/5: don't drop — surface the malformed string for
    # the validator to flag.
    raw = {
        "choices": [
            {
                "message": {
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "c1",
                            "function": {"name": "x", "arguments": "{invalid"},
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
    }

    out = provider._parse_response(raw)

    assert out.tool_calls[0].arguments == {"_malformed_json": "{invalid"}


def test_parse_response_generates_id_when_tool_call_id_missing(
    provider: _TestOpenAI, mocker
) -> None:
    mocker.patch(
        "physiclaw.provider.openai_compat.uuid.uuid4",
        return_value=mocker.MagicMock(hex="abcdef1234567890"),
    )
    raw = {
        "choices": [
            {
                "message": {
                    "tool_calls": [
                        {
                            "function": {"name": "tap", "arguments": "{}"},
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
    }

    out = provider._parse_response(raw)

    assert out.tool_calls[0].id == "auto_abcdef12"


def test_parse_response_logs_exception_when_tool_call_parse_fails(
    provider: _TestOpenAI, caplog: pytest.LogCaptureFixture, mocker
) -> None:
    import logging

    # Make ToolCall constructor raise to hit the broad-except branch.
    mocker.patch(
        "physiclaw.provider.openai_compat.ToolCall",
        side_effect=RuntimeError("boom"),
    )
    raw = {
        "choices": [
            {
                "message": {
                    "tool_calls": [
                        {
                            "id": "c1",
                            "function": {"name": "x", "arguments": "{}"},
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
    }

    with caplog.at_level(logging.ERROR, logger="physiclaw.provider.openai_compat"):
        out = provider._parse_response(raw)

    # The bad tool call gets dropped; the log captures the exception.
    assert out.tool_calls == []
    assert any("failed to parse tool_call" in r.getMessage() for r in caplog.records)


def test_parse_response_empty_choices_falls_back_to_defaults(
    provider: _TestOpenAI,
) -> None:
    out = provider._parse_response({})

    assert out.content == ""
    assert out.tool_calls == []
    assert out.finish_reason == FinishReason.STOP


# ---------- _parse_usage ----------


def test_parse_usage_with_full_openai_shape(provider: _TestOpenAI) -> None:
    raw = {
        "usage": {
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "prompt_tokens_details": {
                "cached_tokens": 80,
                "cache_creation_input_tokens": 10,
            },
        },
    }

    out = provider._parse_usage(raw)

    assert out == Usage(
        prompt_tokens=100,
        completion_tokens=50,
        cached_tokens=80,
        cache_creation_tokens=10,
    )


def test_parse_usage_returns_zero_filled_when_usage_missing(
    provider: _TestOpenAI,
) -> None:
    assert provider._parse_usage({}) == Usage()


def test_parse_usage_handles_none_usage_block(provider: _TestOpenAI) -> None:
    assert provider._parse_usage({"usage": None}) == Usage()


def test_parse_usage_handles_none_token_counts(provider: _TestOpenAI) -> None:
    raw = {"usage": {"prompt_tokens": None, "completion_tokens": None}}

    assert provider._parse_usage(raw) == Usage()


def test_parse_usage_handles_missing_prompt_tokens_details(
    provider: _TestOpenAI,
) -> None:
    raw = {"usage": {"prompt_tokens": 50, "completion_tokens": 25}}

    out = provider._parse_usage(raw)

    assert out.cached_tokens == 0
    assert out.cache_creation_tokens == 0


def test_parse_usage_handles_none_when_passed_a_none_raw(
    provider: _TestOpenAI,
) -> None:
    # Defensive: `(raw or {}).get("usage")` — None raw is permitted.
    assert provider._parse_usage(None) == Usage()


# ---------- _parse_usage: top-level cached_tokens fallback ----------
#
# Moonshot K2 sometimes surfaces `cached_tokens` at the top of the
# `usage` block instead of nested under `prompt_tokens_details`; the
# base parser probes there only when the nested read came up empty, so
# the fallback is a no-op for vendors that never do this.


def test_parse_usage_falls_through_to_top_level_cached_tokens_when_nested_empty(
    provider: _TestOpenAI,
) -> None:
    raw = {
        "usage": {
            "prompt_tokens": 200,
            "completion_tokens": 30,
            "cached_tokens": 150,  # top-level, no nested details
        },
    }

    out = provider._parse_usage(raw)

    assert out.cached_tokens == 150
    assert out.prompt_tokens == 200


def test_parse_usage_prefers_nested_cached_tokens_over_top_level_when_both_set(
    provider: _TestOpenAI,
) -> None:
    """Defensive: if a response carries both, nested wins (it's the
    standard location). The fallback only fires on empty nested."""
    raw = {
        "usage": {
            "prompt_tokens": 100,
            "completion_tokens": 10,
            "cached_tokens": 999,  # would-be top-level
            "prompt_tokens_details": {"cached_tokens": 80},
        },
    }

    out = provider._parse_usage(raw)

    assert out.cached_tokens == 80


def test_parse_usage_handles_string_top_level_cached_tokens(
    provider: _TestOpenAI,
) -> None:
    """Some upstream proxies stringify integers; the int() cast in the
    fallback must accept that."""
    raw = {
        "usage": {
            "prompt_tokens": 50,
            "completion_tokens": 5,
            "cached_tokens": "42",  # stringified
        },
    }

    out = provider._parse_usage(raw)

    assert out.cached_tokens == 42


@pytest.mark.asyncio
async def test_chat_non_json_200_maps_to_transient(
    provider: _TestOpenAI, respx_mock: respx.MockRouter
) -> None:
    """A gateway can 200 with an HTML body mid-restart — retryable."""
    respx_mock.post("https://api.openai.test/v1/chat/completions").respond(
        content=b"<html>Bad gateway page</html>", content_type="text/html"
    )

    with pytest.raises(ProviderTransientError, match=r"^non-JSON response: "):
        await provider.chat([UserMessage(content="x")], [])


@pytest.mark.asyncio
async def test_chat_empty_choices_returns_empty_assistant_message(
    provider: _TestOpenAI, respx_mock: respx.MockRouter
) -> None:
    respx_mock.post("https://api.openai.test/v1/chat/completions").respond(
        json={"choices": [], "usage": {}}
    )

    out = await provider.chat([UserMessage(content="x")], [])

    assert out.content == ""
    assert out.tool_calls == []


@pytest.mark.asyncio
async def test_list_models_non_json_200_maps_to_transient(
    provider: _TestOpenAI, respx_mock: respx.MockRouter
) -> None:
    respx_mock.get("https://api.openai.test/v1/models").respond(
        content=b"<html>proxy</html>", content_type="text/html"
    )

    with pytest.raises(ProviderTransientError, match=r"^non-JSON response: "):
        await provider.list_models()
