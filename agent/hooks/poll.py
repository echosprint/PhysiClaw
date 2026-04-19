"""Phone watchdog hook — fires when /api/phone/watch reports an event.

Auto-discovered by `agent.runtime.hook.load_hooks()`. Reads the MCP
server URL from the `PHYSICLAW_SERVER` env var, which `__main__` sets
from the `--server` flag before hooks are loaded.

Later siblings (e.g. a cron hook) live next to this one and return the
same `Trigger` shape, so the runtime loop treats all event sources
uniformly.
"""

import logging
import os

import httpx

from agent.runtime.hook import Trigger, register

log = logging.getLogger(__name__)

_client: httpx.AsyncClient | None = None
_in_blip = False


def _get_client() -> httpx.AsyncClient:
    global _client
    if _client is None:
        base_url = os.environ.get("PHYSICLAW_SERVER", "http://127.0.0.1:8048")
        _client = httpx.AsyncClient(base_url=base_url, timeout=10.0)
    return _client


@register
async def phone_watch() -> Trigger | None:
    global _in_blip
    try:
        r = await _get_client().get("/api/phone/watch")
        r.raise_for_status()
        _in_blip = False
    except Exception as e:
        if not _in_blip:
            log.warning("phone watch poll failed: %s", e)
        _in_blip = True
        return None
    data = r.json()
    if not data.get("wake"):
        return None
    return Trigger(description=data.get("reason", "phone screen changed"), source="phone")
