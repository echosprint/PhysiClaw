"""MCP HTTP client wrapper for the engine.

Holds one MCP session open for the runtime process's lifetime — opening
a new session per agent wake would add ~100ms of handshake per cycle
and spam the log with "GET stream disconnected" on every close.
Module-level `get_mcp()` returns the singleton; `close_mcp()` tears it
down at process exit.
"""

import logging
from contextlib import AsyncExitStack
from typing import Any

import httpx
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from mcp.types import ImageContent, TextContent

from physiclaw.common import platform
from physiclaw.common.config import server_url

log = logging.getLogger(__name__)


class McpClient:
    """Async-context wrapper around an MCP streamable-HTTP session.

        async with McpClient() as mcp:
            tools = await mcp.list_tools()
            blocks = await mcp.call_tool("tap", {"bbox": [0.4, 0.5, 0.5, 0.6]})

    `server_instructions` is the `instructions` field the server sent back
    in its initialize response — empty string if the server set nothing.
    Caller injects it into the system prompt.
    """

    def __init__(self, base_url: str | None = None):
        base = base_url or server_url()
        self._url = base.rstrip("/") + "/mcp"
        self._stack = AsyncExitStack()
        self._session: ClientSession | None = None
        self.server_instructions: str = ""

    async def __aenter__(self) -> "McpClient":
        # Hand the transport our own httpx client so `trust_env` follows the
        # same per-platform proxy policy as every other localhost client (the
        # status poll, phone-watch hook, doctor). The MCP server is this
        # runtime's parent on 127.0.0.1; with httpx's default trust_env=True a
        # configured system proxy (common on Windows) hijacks that localhost
        # request and the initialize() handshake hangs forever. We own the
        # client, so our stack closes it — entered first so it outlives the
        # transport's terminate-on-close DELETE. Other kwargs match mcp's
        # create_mcp_http_client defaults.
        http_client = await self._stack.enter_async_context(
            httpx.AsyncClient(
                follow_redirects=True,
                timeout=httpx.Timeout(30.0, read=300.0),
                trust_env=platform.TRUST_PROXY_ENV,
            )
        )
        read, write, _ = await self._stack.enter_async_context(
            streamable_http_client(self._url, http_client=http_client)
        )
        self._session = await self._stack.enter_async_context(
            ClientSession(read, write)
        )
        init = await self._session.initialize()
        self.server_instructions = (init.instructions or "").rstrip()
        log.info("MCP client connected (%s)", self._url)
        return self

    async def __aexit__(self, *exc) -> None:
        await self._stack.aclose()
        self._session = None

    async def list_tools(self) -> list[dict]:
        """Return tool schemas as plain dicts: {name, description, input_schema}."""
        assert self._session is not None, "McpClient not entered"
        result = await self._session.list_tools()
        return [
            {
                "name": t.name,
                "description": t.description or "",
                "input_schema": t.inputSchema,
            }
            for t in result.tools
        ]

    async def call_tool(
        self, name: str, args: dict[str, Any] | None = None
    ) -> list[dict]:
        """Call an MCP tool and return its content as a list of normalized blocks.

        Each block is either:
          {"type": "text", "text": str}
          {"type": "image", "mime_type": str, "data": <base64 str>}
        """
        assert self._session is not None, "McpClient not entered"
        result = await self._session.call_tool(name, args or {})
        blocks: list[dict] = []
        for c in result.content:
            if isinstance(c, TextContent):
                blocks.append({"type": "text", "text": c.text})
            elif isinstance(c, ImageContent):
                blocks.append(
                    {
                        "type": "image",
                        "mime_type": c.mimeType,
                        "data": c.data,
                    }
                )
            else:
                # Unknown block type — stringify for debugging. MCP may grow
                # resource/embedded types later; we don't fail on those.
                blocks.append({"type": "text", "text": repr(c)})
        if getattr(result, "isError", False):
            joined = " | ".join(
                b.get("text", "") for b in blocks if b["type"] == "text"
            )
            raise RuntimeError(f"tool {name!r} failed: {joined}")
        return blocks


# ---------- process-level singleton ----------

_stack: AsyncExitStack | None = None
_mcp: "McpClient | None" = None
_tools_cache: list[dict] | None = None


async def get_mcp() -> McpClient:
    """Return the process-level McpClient, opening it on first call.

    Persists across agent wakes so the SSE channel and initialize
    handshake are paid once per runtime process, not once per session.
    """
    global _stack, _mcp
    if _mcp is None:
        stack = AsyncExitStack()
        await stack.__aenter__()
        try:
            _mcp = await stack.enter_async_context(McpClient())
        except BaseException:
            # Unwind anything entered before the failure so the stack
            # doesn't leak transports / sessions on a half-open client.
            await stack.aclose()
            raise
        _stack = stack
    return _mcp


async def list_tools_cached() -> list[dict]:
    """Cached MCP tool schemas. Stable for the runtime's lifetime — the
    MCP server is the runtime's parent process, so the tool surface can't
    change without restarting both."""
    global _tools_cache
    if _tools_cache is None:
        mcp = await get_mcp()
        _tools_cache = await mcp.list_tools()
    return _tools_cache


async def close_mcp() -> None:
    """Close the singleton if it's open. Safe to call when it isn't."""
    global _stack, _mcp, _tools_cache
    if _stack is None:
        return
    stack, _stack, _mcp, _tools_cache = _stack, None, None, None
    try:
        await stack.aclose()
    except Exception:
        log.warning("MCP client close failed", exc_info=True)
