"""Server-ready check — the one definition of "PhysiClaw is ready".

``/api/status`` reports ``ready: true`` once hardware bring-up completes
(hot-start resume, the setup wizard's final step). Every out-of-process
reader shares this module: the agent runtime's wake loop and the CLI's
mcp-mode announcement poll via the probes below; doctor and the setup
wizard fetch the richer status body through their fail-soft urllib
helper and share ``STATUS_PATH`` + ``ready_from_status`` — the path and
the flag reading have one home either way. Both probes raise on
connect/HTTP errors: blip policy (hold last-known, retry) is the
caller's, and the callers deliberately differ.

httpx loads inside the sync probe, not at module top, so the pure
contract stays free to import from CLI paths that never probe.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from physiclaw.common import platform

if TYPE_CHECKING:
    import httpx

STATUS_PATH = "/api/status"


def ready_from_status(payload: dict) -> bool:
    """Read the ``ready`` flag out of an ``/api/status`` JSON body."""
    return bool(payload.get("ready"))


async def check_ready(client: httpx.AsyncClient) -> bool:
    """One async probe. ``client.base_url`` must point at the server."""
    r = await client.get(STATUS_PATH)
    r.raise_for_status()
    return ready_from_status(r.json())


def check_ready_once(base_url: str, *, timeout: float = 2.0) -> bool:
    """One sync probe against ``base_url`` — the async twin for callers
    without an event loop (a CLI watcher thread)."""
    import httpx

    r = httpx.get(
        f"{base_url}{STATUS_PATH}",
        timeout=timeout,
        trust_env=platform.TRUST_PROXY_ENV,
    )
    r.raise_for_status()
    return ready_from_status(r.json())
