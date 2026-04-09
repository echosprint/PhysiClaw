"""Hook registry — subscribe event sources to the runtime loop.

A hook is a no-arg (sync or async) callable that checks some condition
and returns a `Trigger` if it fired, or `None` if it didn't:

    @register
    async def phone_watch() -> Trigger | None:
        if <something happened>:
            return Trigger(description="phone screen changed", source="phone")
        return None

The runtime loop calls `check_hooks()` every interval. It runs every
registered hook in order, collects the triggers that fired, and returns
them as a list. An empty list means nothing fired. Exceptions in a
single hook are logged and swallowed so one bad hook can't kill the
loop.

`Runtime` imports `check_hooks` and `load_hooks` from this module
directly — the hook registry is not an injection point. Tests exercise
the same path by `clear()`ing and re-`register()`ing hooks.
"""

import importlib
import inspect
import logging
import pkgutil
from dataclasses import dataclass
from typing import Awaitable, Callable, Optional, Union

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Trigger:
    """A hook firing — what happened, and who reported it.

    `description` is the human-readable reason the hook fired, used to
    build the prompt handed to Claude. `source` is a short tag
    identifying the hook (e.g. "phone", "cron:daily-cleanup") so the
    dispatch side can group, filter, or log by origin.
    """

    description: str
    source: str = ""


Hook = Callable[[], Union[Optional[Trigger], Awaitable[Optional[Trigger]]]]
HOOKS_PACKAGE = "physiclaw.hooks"

_hooks: list[Hook] = []
_hooks_loaded = False


def register(fn: Hook) -> Hook:
    """Register a hook. Usable as a decorator or a plain call.

    The function may be sync or async and must take no arguments. It
    must return a `Trigger` when it fires, or `None` when it doesn't.
    If it needs context (an http client, a schedule, bridge state), it
    should read that from module-level state or environment — the
    runtime does not pass any payload.
    """
    _hooks.append(fn)
    return fn


async def check_hooks() -> list[Trigger]:
    """Run every registered hook; return triggers from those that fired.

    Sequential, not concurrent: hooks are expected to be cheap checks,
    and sequential execution keeps logs readable and ordering stable.
    Exceptions are logged per hook and treated as no-trigger so one
    flaky hook can't starve the others.
    """
    triggers: list[Trigger] = []
    for fn in list(_hooks):
        try:
            result = fn()
            if inspect.isawaitable(result):
                result = await result
            if result is not None:
                triggers.append(result)
        except Exception:
            name = getattr(fn, "__name__", repr(fn))
            log.exception("hook failed: %s", name)
    return triggers


def clear() -> None:
    """Remove all registered hooks. Intended for tests."""
    global _hooks_loaded
    _hooks.clear()
    _hooks_loaded = False


def load_hooks() -> None:
    """Auto-import every module under the `physiclaw.hooks` package.

    Each module registers itself at import time via `@register`, so
    simply importing it is enough. Drop a new `.py` file into
    `physiclaw/hooks/` and it will be picked up on the next
    `Runtime.start()` — no metadata, no install step, no manual wiring.
    Idempotent: subsequent calls no-op.
    """
    global _hooks_loaded
    if _hooks_loaded:
        return
    _hooks_loaded = True
    pkg = importlib.import_module(HOOKS_PACKAGE)
    for info in pkgutil.iter_modules(pkg.__path__, prefix=f"{HOOKS_PACKAGE}."):
        if info.name.rsplit(".", 1)[-1].startswith("_"):
            continue
        try:
            importlib.import_module(info.name)
            log.info("loaded hook module: %s", info.name)
        except Exception:
            log.exception("failed to load hook module: %s", info.name)
