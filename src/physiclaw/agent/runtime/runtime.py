"""PhysiClaw Runtime — poll hooks on a timer, react on any trigger.

    while running:
        if not ready: sleep; continue
        triggers = await check_hooks()
        if triggers: await react(triggers)
        sleep(interval)

Hooks stay idle until `/api/status` reports `ready: true` (flipped by
`/setup` on its final step). The `react` callable is the only injection
point — typically `physiclaw.agent.claude.spawn_claude`. Because
`check_hooks()` and `react` are awaited in sequence, no new tick starts
while a reaction is in progress.
"""

import asyncio
import inspect
import logging
from typing import Awaitable, Callable, Union

import httpx

from physiclaw.agent.runtime import contract
from physiclaw.agent.runtime.hook import Trigger, check_hooks, load_hooks
from physiclaw.common import platform
from physiclaw.common.config import CONFIG, server_url
from physiclaw.common.ready import check_ready

log = logging.getLogger(__name__)

# React may return the session's final outcome; None keeps the legacy
# fire-and-forget contract (no streak bookkeeping either way).
_ReactResult = Union["contract.SessionOutcome", None]
React = Callable[[list[Trigger]], Union[_ReactResult, Awaitable[_ReactResult]]]

# Unproductive-streak backoff: a session that closes DONE / WAIT (or asks
# for a restart) resets the streak; anything else — STUCK, FAIL, IDLE, no
# clean close — doubles the post-react cooldown, capped here. A dead
# phone must cost one cheap wake per backoff window, not a full session
# per watchdog blip (field-measured: 12 sessions / ~1.2M tokens in 25min
# against a dead bridge).
BACKOFF_CAP_SECONDS = 600.0


async def _maybe_await(value):
    if inspect.isawaitable(value):
        return await value
    return value


_client: httpx.AsyncClient | None = None


def _get_client() -> httpx.AsyncClient:
    global _client
    if _client is None:
        _client = httpx.AsyncClient(
            base_url=server_url(), timeout=5.0, trust_env=platform.TRUST_PROXY_ENV
        )
    return _client


async def _check_ready() -> bool:
    """Query /api/status (via `common.ready` — shared with the CLI's
    mcp-mode watcher) — True once /setup has finished. Raises on error;
    the server is this process's parent, so any failure is a client-side
    blip and callers should hold last-known."""
    return await check_ready(_get_client())


class Runtime:
    """Run every registered hook on a fixed interval; react on any trigger.

    Args:
        react: Called with the list of triggers whenever `check_hooks()`
            returns a non-empty list. Sync or async. Typical wiring is
            `physiclaw.agent.claude.spawn_claude`, but tests can pass
            any callable.
        interval: Seconds to sleep between hook checks. Not a rate limit
            while `react` is running — sleep only happens after it
            returns, so a slow reaction naturally throttles the loop.
    """

    def __init__(self, react: React, *, interval: float = 1.0, label: str = ""):
        self.react = react
        self.interval = interval
        # human-readable engine/provider tag, surfaced in ready logs so the
        # operator sees what's driving the loop without scrolling startup.
        self.label = label
        self._running = False
        self._streak = 0  # consecutive unproductive sessions — see BACKOFF_CAP

    async def start(self) -> None:
        """Run the loop until `stop()` is called or the task is cancelled."""
        load_hooks()
        self._running = True
        log.info("runtime started (interval=%.2fs)", self.interval)
        last_ready: bool | None = None
        in_blip = False
        suffix = f" [{self.label}]" if self.label else ""
        try:
            while self._running:
                try:
                    ready = last_ready
                    try:
                        ready = await _check_ready()
                        in_blip = False
                    except Exception as e:
                        if not in_blip:
                            log.warning("status poll failed: %s", e)
                        in_blip = True
                    if ready != last_ready:
                        log.info("physiclaw ready=%s%s", ready, suffix)
                        last_ready = ready
                    if not ready or in_blip:
                        await asyncio.sleep(self.interval)
                        continue

                    triggers = await check_hooks()
                    if triggers:
                        sources = [t.source or "?" for t in triggers]
                        log.info("triggers fired: %s", sources)
                        outcome = await _maybe_await(self.react(triggers))
                        # Lets screen animations settle + exceeds watchdog
                        # EMA_STALE so the next poll re-inits its baseline
                        # — stretched by the unproductive-streak backoff.
                        await asyncio.sleep(self._react_cooldown(outcome))
                except asyncio.CancelledError:
                    raise
                except Exception:
                    log.exception("runtime tick failed")
                await asyncio.sleep(self.interval)
        finally:
            self._running = False
            log.info("runtime stopped")

    def _react_cooldown(self, outcome) -> float:
        """Streak bookkeeping + the post-react sleep. `contract.productive`
        owns what counts as an earned close; a legacy react that returned
        None leaves the streak untouched."""
        if outcome is not None:
            if contract.productive(outcome):
                self._streak = 0
            else:
                self._streak += 1
        cooldown = CONFIG.engine.react_cooldown_seconds
        if self._streak:
            cooldown = min(cooldown * (2**self._streak), BACKOFF_CAP_SECONDS)
            log.warning(
                "%d unproductive session(s) in a row (last: %s) — backing off %.0fs",
                self._streak,
                (outcome.status if outcome else None) or "no close",
                cooldown,
            )
        return cooldown

    def stop(self) -> None:
        """Signal the loop to exit after the current iteration finishes."""
        self._running = False
