"""PhysiClaw Runtime — check every hook on a timer, react on any trigger.

Runtime owns one job: every `interval` seconds, run every registered
hook via `check_hooks()` and hand the resulting list of `Trigger`s to
`react` if anything fired.

    while running:
        triggers = await check_hooks()
        if triggers:
            await react(triggers)
        sleep(interval)

`react` is the only injection point — typically
`physiclaw.runtime.claude.spawn_claude`, but any
`(list[Trigger]) -> None | Awaitable[None]` callable works, which keeps
the loop trivially testable. The hook registry itself is not injected:
Runtime calls `load_hooks()` and `check_hooks()` directly, because
that's the whole point.

Because the loop `await`s `check_hooks()` and `react` in sequence, no
new tick can start while a reaction is in progress — serialization is
structural, no busy flag needed.
"""

import asyncio
import inspect
import logging
from typing import Awaitable, Callable, Union

from physiclaw.runtime.hook import Trigger, check_hooks, load_hooks

log = logging.getLogger(__name__)

React = Callable[[list[Trigger]], Union[None, Awaitable[None]]]


async def _maybe_await(value):
    if inspect.isawaitable(value):
        return await value
    return value


class Runtime:
    """Run every registered hook on a fixed interval; react on any trigger.

    Args:
        react: Called with the list of triggers whenever `check_hooks()`
            returns a non-empty list. Sync or async. Typical wiring is
            `physiclaw.runtime.claude.spawn_claude`, but tests can pass
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
        try:
            while self._running:
                try:
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
