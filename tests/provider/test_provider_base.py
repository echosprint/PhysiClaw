"""Tests for `physiclaw.provider.provider_base` — abstract base.

The module is mostly contracts (the concrete request/response flow
lives in `OpenAICompatibleProvider` / `AnthropicCompatibleProvider`).
Tests use a small `_TestProvider(BaseProvider)` stub that satisfies
PROVIDER_ID / BASE_URL and a minimal `_encode_message` so we can
exercise the inherited template methods.

Accepted equivalent mutmut survivors:

  - `log = logging.getLogger(__name__)` ↔ `log = None` — the module
    body never calls log.* itself; the import is reserved for
    subclass use.
  - Local-variable type annotations `Type | None` ↔ `Type & None` —
    not evaluated at runtime.
"""

from __future__ import annotations

import httpx
import pytest

from physiclaw.contract.dto import (
    AssistantMessage,
    CollapsePolicy,
    FinishReason,
    ImageBlock,
    Message,
    SystemMessage,
    TextBlock,
    ToolResultMessage,
    UserMessage,
)
from physiclaw.provider.provider_base import (
    EPHEMERAL_CACHE_CONTROL,
    NO_CACHE_MARKERS,
    BaseProvider,
    CacheMarkers,
    Provider,
    ProviderError,
    ProviderPermanentError,
    ProviderTransientError,
)

# ---------- a minimal concrete stub for inherited-behavior tests ----------


class _TestProvider(BaseProvider):
    PROVIDER_ID = "stub"
    BASE_URL = "https://stub.example/v1"

    def _encode_message(self, msg: Message) -> dict | list[dict] | None:
        if isinstance(msg, SystemMessage):
            return {"role": "system", "content": msg.content}
        if isinstance(msg, UserMessage):
            return {"role": "user", "content": msg.content}
        if isinstance(msg, ToolResultMessage):
            return {"role": "tool", "id": msg.tool_call_id, "content": msg.content}
        if isinstance(msg, AssistantMessage):
            return {"role": "assistant", "content": msg.content}
        return None


@pytest.fixture(autouse=True)
def _stub_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("STUB_API_KEY", "test-key")


@pytest.fixture
def stub() -> _TestProvider:
    return _TestProvider()


class _RecordingMarkers(CacheMarkers):
    """Marker factory that records what it was asked to mark and tags
    the returned copies, so tests can pin *where* the template applies
    markers without a concrete wire shape."""

    def __init__(self) -> None:
        self.system_entries: list[dict] = []
        self.stub_entries: list[dict] = []
        self.tail_entries: list[dict] = []

    def mark_system(self, entry: dict) -> dict:
        self.system_entries.append(entry)
        return {**entry, "marked": "system"}

    def mark_stub(self, entry: dict) -> dict:
        self.stub_entries.append(entry)
        return {**entry, "marked": "stub"}

    def mark_tail(self, entry: dict) -> dict:
        self.tail_entries.append(entry)
        return {**entry, "marked": "tail"}


# ---------- error hierarchy ----------


def test_provider_error_is_exception_subclass() -> None:
    assert issubclass(ProviderError, Exception)


def test_provider_transient_error_is_provider_error_subclass() -> None:
    assert issubclass(ProviderTransientError, ProviderError)


def test_provider_permanent_error_is_provider_error_subclass() -> None:
    assert issubclass(ProviderPermanentError, ProviderError)


def test_transient_and_permanent_are_distinct() -> None:
    assert not issubclass(ProviderTransientError, ProviderPermanentError)
    assert not issubclass(ProviderPermanentError, ProviderTransientError)


# ---------- Protocol & constants ----------


def test_provider_protocol_is_importable() -> None:
    # Protocol class exists with the expected method shape; we can't
    # easily isinstance-check Protocols without runtime_checkable, but
    # we can assert it's a typing.Protocol subclass conceptually.
    assert hasattr(Provider, "chat")
    assert hasattr(Provider, "serialize_history")
    assert hasattr(Provider, "aclose")


def test_ephemeral_cache_control_constant_pinned() -> None:
    assert EPHEMERAL_CACHE_CONTROL == {"type": "ephemeral"}


# ---------- BaseProvider class attributes ----------


def test_baseprovider_default_class_attributes_pinned() -> None:
    assert BaseProvider.PROVIDER_ID == ""
    assert BaseProvider.BASE_URL == ""
    assert BaseProvider.API_KEY_ENV_VARS == ()


def test_baseprovider_collapse_policy_default_pinned() -> None:
    assert BaseProvider.COLLAPSE == CollapsePolicy(first_at=30, keep=10, interval=20)


# ---------- __init__ ----------


def test_init_raises_when_provider_id_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _NoId(BaseProvider):
        PROVIDER_ID = ""
        BASE_URL = "https://x"

    with pytest.raises(
        RuntimeError,
        match=r"^_NoId: PROVIDER_ID and BASE_URL must be set on the subclass$",
    ):
        _NoId()


def test_init_raises_when_base_url_missing() -> None:
    class _NoUrl(BaseProvider):
        PROVIDER_ID = "noid"
        BASE_URL = ""

    with pytest.raises(
        RuntimeError,
        match=r"^_NoUrl: PROVIDER_ID and BASE_URL must be set on the subclass$",
    ):
        _NoUrl()


def test_init_raises_with_missing_key_message_when_no_credential(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("STUB_API_KEY", raising=False)
    from physiclaw.common import config

    monkeypatch.setattr(config, "CONFIG", config.Config())

    with pytest.raises(
        RuntimeError,
        match=(
            r"^stub credential not found\. Set STUB_API_KEY env var "
            r"or \[provider\] stub_api_key in ~/\.physiclaw/config\.toml\.$"
        ),
    ):
        _TestProvider()


def test_init_default_timeout_is_120_seconds() -> None:
    import inspect

    sig = inspect.signature(BaseProvider.__init__)
    assert sig.parameters["timeout"].default == 120.0


def test_init_uses_explicit_model_arg(stub: _TestProvider) -> None:
    p = _TestProvider(model="my-model")

    assert p.model == "my-model"


def test_init_falls_back_to_model_env_var_when_arg_omitted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("STUB_MODEL", "from-env")

    p = _TestProvider()

    assert p.model == "from-env"


def test_init_model_falls_back_to_empty_string_when_arg_and_env_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("STUB_MODEL", raising=False)

    p = _TestProvider()

    assert p.model == ""


# ---------- _build_client / _auth_headers ----------


def test_build_client_uses_resolved_base_url_when_arg_omitted() -> None:
    p = _TestProvider()

    assert str(p._client.base_url).rstrip("/") == "https://stub.example/v1"


def test_build_client_uses_explicit_base_url_arg() -> None:
    p = _TestProvider(base_url="https://override.example/v2")

    assert str(p._client.base_url).rstrip("/") == "https://override.example/v2"


def test_build_client_includes_authorization_header() -> None:
    p = _TestProvider()

    assert p._client.headers["Authorization"] == "Bearer test-key"
    assert p._client.headers["Content-Type"] == "application/json"


def test_build_client_returns_httpx_async_client() -> None:
    p = _TestProvider()

    assert isinstance(p._client, httpx.AsyncClient)


def test_build_client_honours_system_proxy() -> None:
    # The provider is an external endpoint, so its client must trust the proxy
    # env (HTTP(S)_PROXY) — the inverse of the localhost clients that bypass it.
    # Guards against a future "unify proxy handling" change breaking LLM access.
    p = _TestProvider()

    assert p._client.trust_env is True


# ---------- _resolved_base_url ----------


def test_resolved_base_url_returns_class_default_when_no_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from physiclaw.common import config

    monkeypatch.setattr(config, "provider_base_url_override", lambda pid: None)

    assert _TestProvider._resolved_base_url() == "https://stub.example/v1"


def test_resolved_base_url_uses_config_override_when_present(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from physiclaw.common import config

    monkeypatch.setattr(
        config, "provider_base_url_override", lambda pid: "https://proxy/v3"
    )

    assert _TestProvider._resolved_base_url() == "https://proxy/v3"


# ---------- _auth_headers ----------


def test_auth_headers_default_uses_bearer_scheme() -> None:
    p = _TestProvider()

    assert p._auth_headers("xyz") == {"Authorization": "Bearer xyz"}


# ---------- _env_vars / _config_key / _model_env_var ----------


def test_env_vars_default_is_uppercase_id_underscore_API_KEY() -> None:
    assert _TestProvider._env_vars() == ("STUB_API_KEY",)


def test_env_vars_uses_explicit_API_KEY_ENV_VARS_when_set() -> None:
    class _Multi(BaseProvider):
        PROVIDER_ID = "multi"
        BASE_URL = "https://x"
        API_KEY_ENV_VARS = ("MULTI_KEY", "ALT_MULTI_KEY")

    assert _Multi._env_vars() == ("MULTI_KEY", "ALT_MULTI_KEY")


def test_config_key_is_lowercase_id_underscore_api_key() -> None:
    assert _TestProvider._config_key() == "stub_api_key"


def test_model_env_var_is_uppercase_id_underscore_MODEL() -> None:
    assert _TestProvider._model_env_var() == "STUB_MODEL"


# ---------- _missing_key_message ----------


def test_missing_key_message_lists_each_env_var_with_slashes() -> None:
    class _Multi(BaseProvider):
        PROVIDER_ID = "multi"
        BASE_URL = "https://x"
        API_KEY_ENV_VARS = ("MULTI_KEY", "ALT_MULTI_KEY")

    assert _Multi.__bases__[0]._missing_key_message(_Multi.__new__(_Multi)) == (
        "multi credential not found. Set MULTI_KEY / ALT_MULTI_KEY env var "
        "or [provider] multi_api_key in ~/.physiclaw/config.toml."
    )


# ---------- system_prompt_fragment ----------


def test_system_prompt_fragment_default_is_empty_string() -> None:
    assert BaseProvider.system_prompt_fragment() == ""


def test_system_prompt_fragment_returns_declared_class_attr() -> None:
    class _Frag(BaseProvider):
        PROVIDER_ID = "frag"
        BASE_URL = "https://x"
        SYSTEM_PROMPT_FRAGMENT = "wrap reasoning in <think>"

    assert _Frag.system_prompt_fragment() == "wrap reasoning in <think>"


# ---------- serialize_history ----------


def test_serialize_history_dispatches_each_message_through_encode(
    stub: _TestProvider,
) -> None:
    history: list[Message] = [
        SystemMessage(content="sys"),
        UserMessage(content="hi"),
        AssistantMessage(content="ack", tool_calls=[], finish_reason=FinishReason.STOP),
    ]

    out = stub.serialize_history(history)

    assert [e["role"] for e in out] == ["system", "user", "assistant"]


def test_serialize_history_marks_system_at_index_zero(
    stub: _TestProvider,
) -> None:
    markers = _RecordingMarkers()
    stub.CACHE_MARKERS = markers

    out = stub.serialize_history([SystemMessage(content="sys")])

    assert markers.system_entries == [{"role": "system", "content": "sys"}]
    assert out[0]["marked"] == "system"


def test_serialize_history_does_not_mark_system_at_non_zero_index(
    stub: _TestProvider,
) -> None:
    markers = _RecordingMarkers()
    stub.CACHE_MARKERS = markers

    stub.serialize_history([UserMessage(content="hi"), SystemMessage(content="late")])

    assert markers.system_entries == []


def test_serialize_history_marks_last_superseded_tool_result_stub(
    stub: _TestProvider,
) -> None:
    markers = _RecordingMarkers()
    stub.CACHE_MARKERS = markers
    earlier_stub = ToolResultMessage(
        tool_call_id="t1", content="old", is_superseded=True
    )
    later_stub = ToolResultMessage(
        tool_call_id="t2", content="newer", is_superseded=True
    )

    out = stub.serialize_history([earlier_stub, UserMessage(content="x"), later_stub])

    assert [e["id"] for e in markers.stub_entries] == ["t2"]
    assert out[2]["marked"] == "stub"


def test_serialize_history_marks_last_tool_result_as_tail(
    stub: _TestProvider,
) -> None:
    markers = _RecordingMarkers()
    stub.CACHE_MARKERS = markers
    history: list[Message] = [
        ToolResultMessage(tool_call_id="t1", content="a"),
        ToolResultMessage(tool_call_id="t2", content="b"),
        UserMessage(content="volatile tail"),  # rides after; never marked
    ]

    out = stub.serialize_history(history)

    assert [e["id"] for e in markers.tail_entries] == ["t2"]
    assert out[1]["marked"] == "tail"
    assert "marked" not in out[2]


def test_serialize_history_marks_stub_and_tail_on_distinct_results(
    stub: _TestProvider,
) -> None:
    markers = _RecordingMarkers()
    stub.CACHE_MARKERS = markers
    history: list[Message] = [
        ToolResultMessage(tool_call_id="t1", content="old", is_superseded=True),
        ToolResultMessage(tool_call_id="t2", content="live"),
    ]

    out = stub.serialize_history(history)

    assert out[0]["marked"] == "stub"
    assert out[1]["marked"] == "tail"


def test_serialize_history_dual_tail_anchors_straddle_the_live_screen(
    stub: _TestProvider,
) -> None:
    # Two moving anchors: the result BEFORE the live screen (survives
    # the next supersede rewrite) and the last result overall (advances
    # every turn) — here that last result IS the live screen ([note,
    # view] dispatch order).
    markers = _RecordingMarkers()
    stub.CACHE_MARKERS = markers
    history: list[Message] = [
        ToolResultMessage(tool_call_id="note_1", content="noted"),
        ToolResultMessage(
            tool_call_id="peek_1",
            content=[
                TextBlock(text="listing"),
                ImageBlock(media_type="image/jpeg", data_b64="QUJD"),
            ],
        ),
    ]

    out = stub.serialize_history(history)

    assert [e["id"] for e in markers.tail_entries] == ["note_1", "peek_1"]
    assert out[0]["marked"] == "tail"
    assert out[1]["marked"] == "tail"


def test_serialize_history_single_tail_when_nothing_precedes_the_live_screen(
    stub: _TestProvider,
) -> None:
    # [view, note] dispatch order with no earlier results: no pre-view
    # anchor exists; only the advancing last-result anchor is marked.
    markers = _RecordingMarkers()
    stub.CACHE_MARKERS = markers
    history: list[Message] = [
        ToolResultMessage(
            tool_call_id="peek_1",
            content=[
                TextBlock(text="listing"),
                ImageBlock(media_type="image/jpeg", data_b64="QUJD"),
            ],
        ),
        ToolResultMessage(tool_call_id="note_1", content="noted"),
    ]

    out = stub.serialize_history(history)

    assert [e["id"] for e in markers.tail_entries] == ["note_1"]
    assert "marked" not in out[0]
    assert out[1]["marked"] == "tail"


def test_serialize_history_skips_tail_when_it_is_the_marked_stub(
    stub: _TestProvider,
) -> None:
    # The newest tool result IS the latest superseded stub — one anchor,
    # not two stacked marks on the same entry.
    markers = _RecordingMarkers()
    stub.CACHE_MARKERS = markers
    history: list[Message] = [
        ToolResultMessage(tool_call_id="t1", content="stale", is_superseded=True),
    ]

    out = stub.serialize_history(history)

    assert out[0]["marked"] == "stub"
    assert markers.tail_entries == []


def test_serialize_history_skips_messages_when_encode_returns_none() -> None:
    class _NoneEncoder(_TestProvider):
        def _encode_message(self, msg: Message) -> dict | list[dict] | None:
            return None

    out = _NoneEncoder().serialize_history(
        [SystemMessage(content="x"), UserMessage(content="y")]
    )

    assert out == []


def test_serialize_history_continues_past_none_encoded_message() -> None:
    # `continue` on `entries is None` must NOT be `break` — a mid-list
    # None must skip just that message, not abort the loop.
    class _MixedEncoder(_TestProvider):
        def _encode_message(self, msg: Message) -> dict | list[dict] | None:
            if isinstance(msg, SystemMessage):
                return None  # skip systems
            return {"role": "user", "content": getattr(msg, "content", "")}

    out = _MixedEncoder().serialize_history(
        [SystemMessage(content="skip-me"), UserMessage(content="kept")]
    )

    assert out == [{"role": "user", "content": "kept"}]


def test_serialize_history_continues_past_empty_list_encoded_message() -> None:
    # `continue` on the post-list-coerce empty check must NOT be `break`.
    class _EmptyMidEncoder(_TestProvider):
        def _encode_message(self, msg: Message) -> dict | list[dict] | None:
            if isinstance(msg, SystemMessage):
                return []  # empty list → also skip
            return {"role": "user", "content": getattr(msg, "content", "")}

    out = _EmptyMidEncoder().serialize_history(
        [SystemMessage(content="skip-me"), UserMessage(content="kept")]
    )

    assert out == [{"role": "user", "content": "kept"}]


def test_serialize_history_handles_list_form_from_encode() -> None:
    class _ListEncoder(_TestProvider):
        def _encode_message(self, msg: Message) -> dict | list[dict] | None:
            return [{"role": "tool", "part": 1}, {"role": "user", "part": 2}]

    out = _ListEncoder().serialize_history([UserMessage(content="x")])

    assert out == [{"role": "tool", "part": 1}, {"role": "user", "part": 2}]


def test_serialize_history_skips_empty_list_from_encode() -> None:
    class _EmptyEncoder(_TestProvider):
        def _encode_message(self, msg: Message) -> dict | list[dict] | None:
            return []

    out = _EmptyEncoder().serialize_history([UserMessage(content="x")])

    assert out == []


# ---------- raises on the not-yet-overridden hooks ----------


def test_default_encode_message_raises_not_implemented_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Construct a NEW BaseProvider subclass that doesn't override encode.
    monkeypatch.setenv("BARE_API_KEY", "k")

    class _Bare(BaseProvider):
        PROVIDER_ID = "bare"
        BASE_URL = "https://x"

    p = _Bare()

    with pytest.raises(
        NotImplementedError,
        match=r"^_Bare must implement _encode_message$",
    ):
        p._encode_message(SystemMessage(content="x"))


@pytest.mark.asyncio
async def test_default_chat_raises_not_implemented_error(
    stub: _TestProvider,
) -> None:
    with pytest.raises(
        NotImplementedError,
        match=(
            r"^_TestProvider must inherit from a wire-shape base "
            r"\(OpenAICompatibleProvider or AnthropicCompatibleProvider\)$"
        ),
    ):
        await stub.chat([], [])


@pytest.mark.asyncio
async def test_default_list_models_raises_not_implemented_error(
    stub: _TestProvider,
) -> None:
    with pytest.raises(
        NotImplementedError,
        match=r"^_TestProvider must inherit from a wire-shape base$",
    ):
        await stub.list_models()


# ---------- CacheMarkers null object ----------


def test_base_provider_cache_markers_defaults_to_null_object() -> None:
    assert BaseProvider.CACHE_MARKERS is NO_CACHE_MARKERS


def test_no_cache_markers_mark_system_is_identity() -> None:
    entry = {"role": "system", "content": "x"}

    assert NO_CACHE_MARKERS.mark_system(entry) is entry


def test_no_cache_markers_mark_tail_is_identity() -> None:
    entry = {"role": "tool", "content": "x"}

    assert NO_CACHE_MARKERS.mark_tail(entry) is entry


def test_no_cache_markers_mark_stub_is_identity() -> None:
    entry = {"role": "tool", "content": "x"}

    assert NO_CACHE_MARKERS.mark_stub(entry) is entry


def test_serialize_history_leaves_entries_unmarked_with_null_markers(
    stub: _TestProvider,
) -> None:
    # `_TestProvider` inherits NO_CACHE_MARKERS — both anchor
    # positions must come through untouched.
    out = stub.serialize_history(
        [
            SystemMessage(content="sys"),
            ToolResultMessage(tool_call_id="t1", content="old", is_superseded=True),
        ]
    )

    assert out == [
        {"role": "system", "content": "sys"},
        {"role": "tool", "id": "t1", "content": "old"},
    ]


# ---------- aclose ----------


@pytest.mark.asyncio
async def test_aclose_closes_underlying_client(stub: _TestProvider, mocker) -> None:
    aclose_mock = mocker.patch.object(stub._client, "aclose")

    await stub.aclose()

    aclose_mock.assert_awaited_once()
