"""PhysiClaw Runtime — a thin poll/dispatch loop.

Runtime is stateless and transport-agnostic. It takes two callables:

    poll     : () -> bool | Awaitable[bool]
    dispatch : () -> None | Awaitable[None]

The loop is a single coroutine:

    while running:
        if await poll():
            await dispatch()
        sleep(interval)

Because the loop `await`s both the poll and the dispatch directly, no new
poll can start while a dispatch is running — no busy flag, no re-entrance
lock, serialization is structural.

Runtime imports nothing from PhysiClaw. The caller supplies the poll
function (typically an HTTP GET against /api/phone/watch) and the dispatch
function (typically `physiclaw.runtime.hook.dispatch`, but any callable
works for tests and custom wiring).
"""

import asyncio
import inspect
import logging
from typing import Awaitable, Callable, Union

log = logging.getLogger(__name__)

Poll = Callable[[], Union[bool, Awaitable[bool]]]
Dispatch = Callable[[], Union[None, Awaitable[None]]]


async def _maybe_await(value):
    if inspect.isawaitable(value):
        return await value
    return value


class Runtime:
    """Polls `poll` on a fixed interval; calls `dispatch` when it returns True.

    Args:
        poll: Returns True if an event occurred since the last poll. May be
            sync or async. Typically wraps an HTTP GET to the watchdog route.
        dispatch: Called (awaited if async) on every True poll. Typically
            `physiclaw.runtime.hook.dispatch`, but any no-arg callable works.
        interval: Seconds to sleep between polls. Not a rate limit while the
            dispatch is running — sleep only happens after dispatch returns.
    """

    def __init__(
        self,
        poll: Poll,
        dispatch: Dispatch,
        *,
        interval: float = 1.0,
    ):
        self.poll = poll
        self.dispatch = dispatch
        self.interval = interval
        self._running = False

    async def start(self) -> None:
        """Run the loop until `stop()` is called or the task is cancelled."""
        from physiclaw.runtime.hook import load_hooks

        load_hooks()
        self._running = True
        log.info("runtime started (interval=%.2fs)", self.interval)
        try:
            while self._running:
                try:
                    if await _maybe_await(self.poll()):
                        log.info("event detected → dispatch")
                        await _maybe_await(self.dispatch())
                except asyncio.CancelledError:
                    raise
                except Exception:
                    log.exception("poll/dispatch failed")
                await asyncio.sleep(self.interval)
        finally:
            self._running = False
            log.info("runtime stopped")

    def stop(self) -> None:
        """Signal the loop to exit after the current iteration finishes."""
        self._running = False
