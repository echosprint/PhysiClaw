"""The studio's one MCP session — the hardware side of the boundary.

Owns the persistent `McpClient`, the one-rig serialization lock, and
the published-tool allowlist. Every phone interaction goes through
`act`, which returns the normalized view the two browser panes render:
action text, the annotated JPEG, and the listing parsed into element
rows. HTTP marshalling lives in `server.py`; this module never sees a
Request.
"""

import asyncio
import logging
from contextlib import asynccontextmanager

from physiclaw.agent.engine.mcp_tool import McpClient
from physiclaw.common import verdict
from physiclaw.common.config import server_url
from physiclaw.common.listing import Screen
from physiclaw.macros.model import MacroError

log = logging.getLogger(__name__)


def view_reply(blocks: list[dict]) -> dict:
    """Tool-reply blocks → the studio wire shape {text, image, rows,
    listing}.

    The action/listing split is `common.verdict`'s block-shape rule
    (block 0 is action text iff it is a text block) — not re-derived
    here. `listing` is the verbatim text — the shot store keeps it
    whole (Screen.content matching needs its non-row lines too); the
    panes render from `rows`.
    """
    image = next((b for b in blocks if b["type"] == "image"), None)
    listing = verdict.screen_text(blocks)
    return {
        "text": verdict.action_text(blocks),
        "image": (
            {"mime_type": image["mime_type"], "data": image["data"]}
            if image is not None
            else None
        ),
        "rows": [el.to_dict() for el in Screen.read(listing).rows],
        "listing": listing,
    }


class StudioSession:
    """Lazy-connecting wrapper: the studio starts (and stays useful for
    offline curation later) with no server running; the first `act`
    dials out. A transport-level failure drops the client so the next
    call reconnects — the browser sees one failed action and a banner,
    not a dead process."""

    def __init__(self, mcp_url: str | None = None, app: str | None = None):
        # Accept the URL with or without the /mcp suffix — McpClient
        # appends it, and the plan's example spells it with.
        self._base = mcp_url.rstrip("/").removesuffix("/mcp") if mcp_url else None
        self.app = app
        self._client: McpClient | None = None
        self._tools: list[str] = []
        self._lock = asyncio.Lock()

    @property
    def mcp_url(self) -> str:
        return (self._base or server_url()).rstrip("/") + "/mcp"

    @property
    def busy(self) -> bool:
        return self._lock.locked()

    def state(self) -> dict:
        """For GET /api/state — cheap, never dials the server."""
        return {
            "connected": self._client is not None,
            "mcp_url": self.mcp_url,
            "app": self.app,
            "tools": list(self._tools),
        }

    @asynccontextmanager
    async def _rig(self, passthrough: "tuple[type[BaseException], ...]", doing: str):
        """The one transport frame: hold the one-rig lock, lazily
        connect, and on any failure that is not the caller's own
        `passthrough` drop the client (so the next call reconnects)
        and raise ConnectionError. The reconnect policy lives HERE
        alone — act/run_macro/run_walk must never disagree on it."""
        async with self._lock:
            client = await self._connected()
            try:
                yield client
            except passthrough:
                raise
            except Exception as e:
                await self.close()
                raise ConnectionError(f"{doing} failed: {e}") from e

    async def act(self, tool: str, args: dict) -> dict:
        """Call one published tool, serialized (one rig). Raises
        ValueError for a tool outside the published surface,
        ConnectionError when the server is unreachable (client dropped
        for reconnect), RuntimeError when the tool itself errored (it
        ran and said no — the session is fine, so it passes through)."""
        async with self._rig(
            (RuntimeError, ValueError), f"MCP call {tool!r}"
        ) as client:
            if tool not in self._tools:
                raise ValueError(f"tool {tool!r} is not on the published MCP surface")
            blocks = await client.call_tool(tool, args)
        return view_reply(blocks)

    async def run_macro(self, spec, values: dict, start_at: str = ""):
        """Rehearse one macro over THIS session — `run_and_record`, the
        identical code path `macros run` and the engine drive, holding
        the one-rig lock for the whole replay. MacroError (bad input /
        start_at — the draft's to fix) passes through; a mid-run
        gesture failure is a RESULT (the runner's contract)."""
        from physiclaw.macros import runner as macro_runner

        async with self._rig((MacroError,), "rehearsal") as client:
            return await macro_runner.run_and_record(
                spec, values, client, caller="studio", start_at=start_at
            )

    async def run_walk(self, program, registry: dict, emit):
        """Rehearse one armed playbook walk over THIS session —
        `rehearsal.walk`, the identical loop `playbooks run` drives,
        holding the one-rig lock for the whole walk. The caller arms
        first (`rehearsal.arm` — bad specs fail before we dial);
        RuntimeError (no model configured for a decide) passes through
        as config, not transport."""
        from physiclaw.conductor import rehearsal

        async with self._rig((RuntimeError,), "walk") as client:
            return await rehearsal.walk(
                program, registry, client, emit=emit, caller="studio"
            )

    async def _connected(self) -> McpClient:
        if self._client is None:
            client = McpClient(self._base)
            await client.__aenter__()
            self._client = client
            self._tools = [t["name"] for t in await client.list_tools()]
            log.info("studio connected (%s, %d tools)", self.mcp_url, len(self._tools))
        return self._client

    async def close(self) -> None:
        """Drop the client (idempotent). The allowlist stays — it names
        a frozen surface, and /api/state showing it is harmless."""
        client, self._client = self._client, None
        if client is not None:
            try:
                await client.__aexit__(None, None, None)
            except Exception:
                log.debug("studio MCP teardown", exc_info=True)
