"""PhysiClaw Runtime — poll hooks on a timer, react on any trigger.

    while running:
        if not ready: sleep; continue
        triggers = await check_hooks()
        if triggers: await react(triggers)
        sleep(interval)

Hooks stay idle until `/api/status` reports `calibrated: true`. The
`react` callable is the only injection point — typically
`agent.runtime.claude.spawn_claude`. Because `check_hooks()` and
`react` are awaited in sequence, no new tick starts while a reaction
is in progress.
"""

import asyncio
import inspect
import logging
import os
from typing import Awaitable, Callable, Union

import httpx

from agent.runtime.hook import Trigger, check_hooks, load_hooks

log = logging.getLogger(__name__)

React = Callable[[list[Trigger]], Union[None, Awaitable[None]]]


async def _maybe_await(value):
    if inspect.isawaitable(value):
        return await value
    return value


_client: httpx.AsyncClient | None = None


def _get_client() -> httpx.AsyncClient:
    global _client
    if _client is None:
        base_url = os.environ.get("PHYSICLAW_SERVER", "http://127.0.0.1:8048")
        _client = httpx.AsyncClient(base_url=base_url, timeout=5.0)
    return _client


async def _check_ready() -> bool:
    """Query /api/status to see if PhysiClaw hardware is set up and calibrated."""
    try:
        r = await _get_client().get("/api/status")
        return r.status_code == 200 and bool(r.json().get("calibrated"))
    except Exception:
        return False


class Runtime:
    """Run every registered hook on a fixed interval; react on any trigger.

    Args:
        react: Called with the list of triggers whenever `check_hooks()`
            returns a non-empty list. Sync or async. Typical wiring is
            `agent.runtime.claude.spawn_claude`, but tests can pass
            any callable.
        interval: Seconds to sleep between hook checks. Not a rate limit
            while `react` is running — sleep only happens after it
            returns, so a slow reaction naturally throttles the loop.
    """

    def __init__(self, react: React, *, interval: float = 1.0):
        self.react = react
        self.interval = interval
        self._running = False

    async def start(self) -> None:
        """Run the loop until `stop()` is called or the task is cancelled."""
        load_hooks()
        self._running = True
        log.info("runtime started (interval=%.2fs)", self.interval)
        last_ready = None
        try:
            while self._running:
                try:
                    ready = await _check_ready()
                    if ready != last_ready:
                        log.info("physiclaw ready=%s", ready)
                        last_ready = ready
                    if not ready:
                        await asyncio.sleep(self.interval)
                        continue

                    triggers = await check_hooks()
                    if triggers:
                        sources = [t.source or "?" for t in triggers]
                        log.info("triggers fired: %s", sources)
                        await _maybe_await(self.react(triggers))
                except asyncio.CancelledError:
                    raise
                except Exception:
                    log.exception("runtime tick failed")
                await asyncio.sleep(self.interval)
        finally:
            self._running = False
            log.info("runtime stopped")

    def stop(self) -> None:
        """Signal the loop to exit after the current iteration finishes."""
        self._running = False
