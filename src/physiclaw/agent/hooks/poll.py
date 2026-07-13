"""Phone watchdog hook — fires when /api/phone/watch reports an event.

Auto-discovered by `physiclaw.agent.runtime.hook.load_hooks()`. Dials
`config.server_url()` — the launcher pins `PHYSICLAW_SERVER` from the
`--server` flag before hooks are loaded, and that env var wins there.

Later siblings (e.g. a cron hook) live next to this one and return the
same `Trigger` shape, so the runtime loop treats all event sources
uniformly.
"""

import logging

import httpx

from physiclaw.agent.runtime.hook import Trigger, register
from physiclaw.common import platform
from physiclaw.common.config import server_url

log = logging.getLogger(__name__)

_client: httpx.AsyncClient | None = None
_in_blip = False


def _get_client() -> httpx.AsyncClient:
    global _client
    if _client is None:
        _client = httpx.AsyncClient(
            base_url=server_url(), timeout=10.0, trust_env=platform.TRUST_PROXY_ENV
        )
    return _client


@register
async def phone_watch() -> Trigger | None:
    global _in_blip
    try:
        r = await _get_client().get("/api/phone/watch")
        r.raise_for_status()
        # Parse inside the try: a 200 with a non-JSON body (server
        # mid-restart, proxy page) is a blip, not a crash per tick.
        data = r.json()
        _in_blip = False
    except Exception as e:
        if not _in_blip:
            log.warning("phone watch poll failed: %s", e)
        _in_blip = True
        return None
    if not data.get("wake"):
        return None
    return Trigger(
        description=data.get("reason", "phone screen changed"), source="phone"
    )
