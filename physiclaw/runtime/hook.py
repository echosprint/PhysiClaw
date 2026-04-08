"""Hook registry — subscribe user code to runtime events.

Usage:
    from physiclaw.runtime import register

    @register
    async def handle_phone_event():
        print("phone woke up")

The `Runtime` loop calls `dispatch()` when its poll returns True. `dispatch`
iterates the registry and runs every registered hook in order, awaiting
async ones, calling sync ones directly. Exceptions in a hook are logged
and swallowed so one bad hook can't kill the loop.

Runtime does not import this module — it takes `dispatch` as a callable
argument. That keeps the loop testable with a plain lambda and makes the
registry an optional convenience, not a requirement.
"""

import importlib
import inspect
import logging
import pkgutil
from typing import Awaitable, Callable, Union

log = logging.getLogger(__name__)

Hook = Callable[[], Union[None, Awaitable[None]]]
HOOKS_PACKAGE = "physiclaw.hooks"

_hooks: list[Hook] = []
_hooks_loaded = False


def register(fn: Hook) -> Hook:
    """Register a hook. Usable as a decorator or a plain call.

    The function may be sync or async and must take no arguments. If it
    needs context (a screenshot, the camera, bridge state), it should pull
    that from the environment itself — Runtime does not pass any payload.
    """
    _hooks.append(fn)
    return fn


async def dispatch() -> None:
    """Call every registered hook sequentially.

    Exceptions are logged per hook and do not abort the rest of the chain,
    and do not propagate out of dispatch. Runtime's loop treats a clean
    return as "hooks ran"; if every hook errored, Runtime still sleeps
    normally and polls again.
    """
    for fn in list(_hooks):
        try:
            result = fn()
            if inspect.isawaitable(result):
                await result
        except Exception:
            name = getattr(fn, "__name__", repr(fn))
            log.exception("hook failed: %s", name)


def clear() -> None:
    """Remove all registered hooks. Intended for tests."""
    global _hooks_loaded
    _hooks.clear()
    _hooks_loaded = False


def load_hooks() -> None:
    """Auto-import every module under the `physiclaw.hooks` package.

    Each module registers itself at import time via `@register`, so simply
    importing it is enough. Drop a new `.py` file into `physiclaw/hooks/`
    and it will be picked up on the next `Runtime.start()` — no metadata,
    no install step, no manual wiring. Idempotent: subsequent calls no-op.
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
