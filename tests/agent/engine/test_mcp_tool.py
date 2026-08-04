"""Tests for `physiclaw.agent.engine.mcp_tool` — MCP HTTP wrapper.

The MCP transport (`streamable_http_client`) and `ClientSession` are
both mocked with `AsyncMock`. Tests assert the public-surface
behavior: URL construction, session lifecycle, content normalization,
error mapping, and the process-level singleton + cache.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest
from mcp.types import ImageContent, TextContent

from physiclaw.agent.engine import mcp_tool


@pytest.fixture(autouse=True)
def _reset_singleton(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tests must start with the module-level singleton cleared."""
    monkeypatch.setattr(mcp_tool, "_stack", None)
    monkeypatch.setattr(mcp_tool, "_mcp", None)
    monkeypatch.setattr(mcp_tool, "_tools_cache", None)


@pytest.fixture(autouse=True)
def _no_real_server_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Strip PHYSICLAW_SERVER so the URL construction tests start clean."""
    monkeypatch.delenv("PHYSICLAW_SERVER", raising=False)


# ---------- URL construction ----------


def test_default_url_is_localhost_8048() -> None:
    c = mcp_tool.McpClient()

    assert c._url == "http://127.0.0.1:8048/mcp"


def test_url_uses_environment_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PHYSICLAW_SERVER", "http://other.host:9999")

    c = mcp_tool.McpClient()

    assert c._url == "http://other.host:9999/mcp"


def test_url_uses_constructor_base_url_over_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PHYSICLAW_SERVER", "http://env.host")

    c = mcp_tool.McpClient(base_url="http://explicit.host:1234")

    assert c._url == "http://explicit.host:1234/mcp"


def test_url_strips_trailing_slash_from_base() -> None:
    c = mcp_tool.McpClient(base_url="http://host:1234/")

    assert c._url == "http://host:1234/mcp"


# ---------- McpClient lifecycle ----------


@pytest.fixture
def fake_session(mocker):
    """A mocked ClientSession with controllable list_tools / call_tool."""
    sess = mocker.MagicMock()
    sess.initialize = mocker.AsyncMock(
        return_value=SimpleNamespace(instructions="server says hi")
    )
    sess.list_tools = mocker.AsyncMock()
    sess.call_tool = mocker.AsyncMock()
    return sess


async def _no_probe(self) -> None:
    """Stand-in for the socket pre-flight when the transport is faked."""


@pytest.fixture
def patched_transport(mocker, fake_session):
    """Patch streamable_http_client + ClientSession so __aenter__ wires
    a synthetic read/write triple and our fake session. Records the
    httpx client handed to the transport on ``fake_session.captured``."""
    captured: dict = {}

    @asynccontextmanager
    async def fake_http(url: str, *, http_client=None):
        captured["http_client"] = http_client
        yield ("read", "write", "extra")

    mocker.patch.object(mcp_tool, "streamable_http_client", fake_http)
    # The connect pre-flight opens a real socket; a faked transport means
    # there is nothing to reach, so stand it down with the rest.
    mocker.patch.object(mcp_tool.McpClient, "_probe", _no_probe)

    @asynccontextmanager
    async def fake_session_ctx(read, write):
        yield fake_session

    mocker.patch.object(mcp_tool, "ClientSession", side_effect=fake_session_ctx)
    fake_session.captured = captured
    return fake_session


@pytest.mark.asyncio
async def test_async_enter_hands_transport_a_trust_env_aware_client(
    patched_transport,
) -> None:
    # The transport must get our own httpx client whose trust_env follows the
    # per-platform proxy policy — otherwise httpx routes the localhost MCP
    # request through a configured system proxy (Windows) and initialize hangs.
    async with mcp_tool.McpClient():
        pass

    client = patched_transport.captured["http_client"]
    assert client is not None
    assert client.trust_env is mcp_tool.platform.TRUST_PROXY_ENV


@pytest.mark.asyncio
async def test_async_enter_calls_initialize_and_records_instructions(
    patched_transport,
) -> None:
    async with mcp_tool.McpClient() as c:
        assert c.server_instructions == "server says hi"

    patched_transport.initialize.assert_awaited_once()


@pytest.mark.asyncio
async def test_async_enter_strips_trailing_whitespace_from_instructions(
    patched_transport, mocker
) -> None:
    patched_transport.initialize.return_value = SimpleNamespace(
        instructions="hello   \n"
    )

    async with mcp_tool.McpClient() as c:
        assert c.server_instructions == "hello"


@pytest.mark.asyncio
async def test_async_enter_handles_none_instructions(
    patched_transport,
) -> None:
    patched_transport.initialize.return_value = SimpleNamespace(instructions=None)

    async with mcp_tool.McpClient() as c:
        assert c.server_instructions == ""


# ---------- list_tools ----------


@pytest.mark.asyncio
async def test_list_tools_returns_normalized_dicts(patched_transport) -> None:
    patched_transport.list_tools.return_value = SimpleNamespace(
        tools=[
            SimpleNamespace(
                name="tap", description="Tap", inputSchema={"type": "object"}
            ),
            SimpleNamespace(name="peek", description=None, inputSchema={}),
        ]
    )

    async with mcp_tool.McpClient() as c:
        out = await c.list_tools()

    assert out == [
        {"name": "tap", "description": "Tap", "input_schema": {"type": "object"}},
        # None description normalized to empty string.
        {"name": "peek", "description": "", "input_schema": {}},
    ]


@pytest.mark.asyncio
async def test_list_tools_asserts_session_entered() -> None:
    c = mcp_tool.McpClient()

    with pytest.raises(AssertionError, match=r"^McpClient not entered$"):
        await c.list_tools()


# ---------- call_tool ----------


@pytest.mark.asyncio
async def test_call_tool_normalizes_text_blocks(patched_transport) -> None:
    patched_transport.call_tool.return_value = SimpleNamespace(
        content=[
            TextContent(type="text", text="hello"),
        ],
        isError=False,
    )

    async with mcp_tool.McpClient() as c:
        out = await c.call_tool("tap")

    assert out == [{"type": "text", "text": "hello"}]


@pytest.mark.asyncio
async def test_call_tool_normalizes_image_blocks(patched_transport) -> None:
    patched_transport.call_tool.return_value = SimpleNamespace(
        content=[
            ImageContent(type="image", mimeType="image/png", data="aGk="),
        ],
        isError=False,
    )

    async with mcp_tool.McpClient() as c:
        out = await c.call_tool("peek")

    assert out == [{"type": "image", "mime_type": "image/png", "data": "aGk="}]


@pytest.mark.asyncio
async def test_call_tool_image_shaped_duck_falls_to_repr(
    patched_transport,
) -> None:
    # Normalization is type-driven (isinstance over mcp.types), not
    # duck-typed — an image-shaped impostor takes the stringify branch.
    class _ImgBlob:
        type = "image"
        data = "aGk="

        def __repr__(self) -> str:
            return "<img-blob>"

    patched_transport.call_tool.return_value = SimpleNamespace(
        content=[_ImgBlob()],
        isError=False,
    )

    async with mcp_tool.McpClient() as c:
        out = await c.call_tool("p")

    assert out == [{"type": "text", "text": "<img-blob>"}]


@pytest.mark.asyncio
async def test_call_tool_unknown_block_type_stringified_as_text(
    patched_transport,
) -> None:
    class _Mystery:
        type = "future_resource"

        def __repr__(self) -> str:
            return "<future-thing>"

    patched_transport.call_tool.return_value = SimpleNamespace(
        content=[_Mystery()],
        isError=False,
    )

    async with mcp_tool.McpClient() as c:
        out = await c.call_tool("p")

    assert out == [{"type": "text", "text": "<future-thing>"}]


@pytest.mark.asyncio
async def test_call_tool_passes_args_to_session(patched_transport) -> None:
    patched_transport.call_tool.return_value = SimpleNamespace(
        content=[],
        isError=False,
    )

    async with mcp_tool.McpClient() as c:
        await c.call_tool("tap", {"bbox": [0.0, 0.0, 1.0, 1.0]})

    patched_transport.call_tool.assert_awaited_once_with(
        "tap", {"bbox": [0.0, 0.0, 1.0, 1.0]}
    )


@pytest.mark.asyncio
async def test_call_tool_uses_empty_dict_when_args_none(
    patched_transport,
) -> None:
    patched_transport.call_tool.return_value = SimpleNamespace(
        content=[],
        isError=False,
    )

    async with mcp_tool.McpClient() as c:
        await c.call_tool("peek")

    args = patched_transport.call_tool.await_args.args
    assert args == ("peek", {})


@pytest.mark.asyncio
async def test_call_tool_raises_runtime_error_when_is_error_true(
    patched_transport,
) -> None:
    patched_transport.call_tool.return_value = SimpleNamespace(
        content=[
            TextContent(type="text", text="bad arg"),
            TextContent(type="text", text="bbox missing"),
        ],
        isError=True,
    )

    async with mcp_tool.McpClient() as c:
        with pytest.raises(
            RuntimeError, match=r"^tool 'tap' failed: bad arg \| bbox missing$"
        ):
            await c.call_tool("tap")


@pytest.mark.asyncio
async def test_call_tool_asserts_session_entered() -> None:
    c = mcp_tool.McpClient()

    with pytest.raises(AssertionError, match=r"^McpClient not entered$"):
        await c.call_tool("any")


# ---------- get_mcp / list_tools_cached / close_mcp ----------


@pytest.mark.asyncio
async def test_get_mcp_creates_singleton_on_first_call(
    patched_transport,
) -> None:
    c1 = await mcp_tool.get_mcp()
    c2 = await mcp_tool.get_mcp()

    assert c1 is c2
    assert mcp_tool._mcp is c1
    assert mcp_tool._stack is not None


@pytest.mark.asyncio
async def test_get_mcp_cleans_up_stack_on_aenter_failure(mocker) -> None:
    @asynccontextmanager
    async def boom_http(url: str, *, http_client=None):
        raise RuntimeError("transport failed")
        yield None  # pragma: no cover

    mocker.patch.object(mcp_tool, "streamable_http_client", boom_http)
    mocker.patch.object(mcp_tool.McpClient, "_probe", _no_probe)

    with pytest.raises(ConnectionError, match="transport failed"):
        await mcp_tool.get_mcp()

    # Stack must NOT have been retained.
    assert mcp_tool._stack is None
    assert mcp_tool._mcp is None


@pytest.mark.asyncio
async def test_list_tools_cached_caches_first_result(
    patched_transport,
) -> None:
    patched_transport.list_tools.return_value = SimpleNamespace(
        tools=[
            SimpleNamespace(name="tap", description="Tap", inputSchema={}),
        ]
    )

    a = await mcp_tool.list_tools_cached()
    b = await mcp_tool.list_tools_cached()

    assert a is b
    # Server only contacted once for the list.
    patched_transport.list_tools.assert_awaited_once()


@pytest.mark.asyncio
async def test_close_mcp_no_op_when_singleton_unset() -> None:
    # No prior get_mcp call — close_mcp should be safe.
    await mcp_tool.close_mcp()


@pytest.mark.asyncio
async def test_close_mcp_clears_singleton_state(patched_transport) -> None:
    await mcp_tool.get_mcp()
    assert mcp_tool._stack is not None

    await mcp_tool.close_mcp()

    assert mcp_tool._stack is None
    assert mcp_tool._mcp is None
    assert mcp_tool._tools_cache is None


@pytest.mark.asyncio
async def test_close_mcp_logs_warning_on_aclose_exception(
    patched_transport, mocker, caplog: pytest.LogCaptureFixture
) -> None:
    import logging

    await mcp_tool.get_mcp()
    # Replace the stack with one that raises on aclose.
    bad_stack = mocker.MagicMock()
    bad_stack.aclose = mocker.AsyncMock(side_effect=RuntimeError("nope"))
    mcp_tool._stack = bad_stack

    with caplog.at_level(logging.WARNING, logger="physiclaw.agent.engine.mcp_tool"):
        await mcp_tool.close_mcp()

    assert any(
        r.getMessage().startswith("MCP client close failed") for r in caplog.records
    )


# ---------- connect failures are legible ----------


@pytest.mark.asyncio
async def test_probe_raises_connection_error_when_nothing_is_listening() -> None:
    # A refused connection inside `streamable_http_client` collapses its
    # anyio scope into a bare CancelledError — a BaseException no
    # `except Exception` can catch, which is why the CLI's "cannot reach the
    # server" message never fired and `dispatch` could break its "never
    # raises" contract. Answer the question before the transport does.
    client = mcp_tool.McpClient("http://127.0.0.1:1")

    with pytest.raises(ConnectionError, match="cannot reach the MCP server"):
        await client._probe()


@pytest.mark.asyncio
async def test_probe_passes_when_a_socket_is_listening() -> None:
    server = await asyncio.start_server(lambda r, w: w.close(), "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    try:
        await mcp_tool.McpClient(f"http://127.0.0.1:{port}")._probe()
    finally:
        server.close()
        await server.wait_closed()


def test_probe_targets_the_url_host_and_port() -> None:
    # A raw socket, not an HTTP request: `trust_env` is on, so a configured
    # system proxy answers an HTTP pre-flight itself and returns 502 for a
    # dead upstream — which reads as "reachable".
    assert mcp_tool.McpClient("http://127.0.0.1:8899")._host_port() == (
        "127.0.0.1",
        8899,
    )
    assert mcp_tool.McpClient("https://rig.local")._host_port() == ("rig.local", 443)
    assert mcp_tool.McpClient("http://rig.local")._host_port() == ("rig.local", 80)
