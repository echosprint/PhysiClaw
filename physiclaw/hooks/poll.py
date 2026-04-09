"""Phone watchdog hook — fires when /api/phone/watch reports an event.

Auto-discovered by `physiclaw.runtime.hook.load_hooks()`. Reads the MCP
server URL from the `PHYSICLAW_SERVER` env var, which `__main__` sets
from the `--server` flag before hooks are loaded.

Later siblings (e.g. a cron hook) live next to this one and return the
same `Trigger` shape, so the runtime loop treats all event sources
uniformly.
"""

import logging
import os

import httpx

from physiclaw.runtime.hook import Trigger, register

log = logging.getLogger(__name__)

_client: httpx.AsyncClient | None = None


def _get_client() -> httpx.AsyncClient:
    global _client
    if _client is None:
        base_url = os.environ.get("PHYSICLAW_SERVER", "http://127.0.0.1:8048")
        _client = httpx.AsyncClient(base_url=base_url, timeout=10.0)
    return _client


@register
async def phone_watch() -> Trigger | None:
    try:
        r = await _get_client().get("/api/phone/watch")
        if r.status_code == 200 and r.json().get("event"):
            return Trigger(
                description="phone screen changed since last check",
                source="phone",
            )
    except Exception:
        log.debug("phone watch poll failed", exc_info=True)
    return None
