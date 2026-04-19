"""Shared log formatting and helpers for PhysiClaw.

Exports:
    setup_logging(tag, level) — colored TTY-aware root-logger config.
    logged(fn) — decorator that logs an MCP tool call's completion
    with args and duration.
"""

import logging
import os
import sys
import time
from collections.abc import Awaitable, Callable
from functools import wraps
from typing import Any, TypeVar, cast

log = logging.getLogger("physiclaw.tools")

# Bound tight so @logged on a sync function is a type error, not a
# runtime `TypeError` from awaiting a non-coroutine.
AsyncFn = TypeVar("AsyncFn", bound=Callable[..., Awaitable[Any]])

# ANSI codes. 256-color greys so timestamp/message fade without going invisible
# on light-theme terminals; levels that should pop use standard 8-color.
_GREY_DARK = "38;5;244"
_GREY_LIGHT = "38;5;250"
_YELLOW = "33"
_RED = "31"

# Tag → accent color. Each entry-point picks its own so devs can skim
# interleaved output at a glance.
_TAG_COLORS = {
    "physiclaw": "36",  # cyan — hardware server
    "runtime": "35",    # magenta — agent loop
}


def _colorize() -> bool:
    return bool(sys.stderr.isatty() and not os.environ.get("NO_COLOR"))


class _TaggedFormatter(logging.Formatter):
    def __init__(self, tag: str, color: bool):
        super().__init__(datefmt="%H:%M")
        self.color = color
        if color:
            self._tag_segment = f"\033[{_TAG_COLORS[tag]}m[{tag}]\033[0m"
        else:
            self._tag_segment = f"[{tag}]"
        # Derive the continuation indent from the actual uncolored prefix
        # so tweaks to the datefmt or tag layout can't drift.
        self._cont_indent = "\n" + " " * len(f"00:00 [{tag}] ")

    def format(self, record: logging.LogRecord) -> str:
        ts = self.formatTime(record, self.datefmt)
        msg = record.getMessage()
        if "\n" in msg:
            msg = msg.replace("\n", self._cont_indent)
        if not self.color:
            return f"{ts} {self._tag_segment} {msg}"
        if record.levelno >= logging.ERROR:
            msg_color = _RED
        elif record.levelno >= logging.WARNING:
            msg_color = _YELLOW
        else:
            msg_color = _GREY_LIGHT
        return (
            f"\033[{_GREY_DARK}m{ts}\033[0m "
            f"{self._tag_segment} "
            f"\033[{msg_color}m{msg}\033[0m"
        )


def setup_logging(tag: str, level: int = logging.INFO) -> None:
    """Configure the root logger with the colored, tagged format."""
    handler = logging.StreamHandler()
    handler.setFormatter(_TaggedFormatter(tag, _colorize()))
    logging.basicConfig(level=level, handlers=[handler], force=True)


# Caps a tool-call log line so a 100KB clipboard body doesn't flood it.
_MAX_ARG_LOG_LEN = 80


def _format_args(fn_name: str, kwargs: dict) -> str:
    """Redact clipboard text (IM bodies, search queries, anything pasted)
    and summarize sequence steps to tool names only, since a step may
    itself be a send_to_clipboard."""
    if fn_name == "send_to_clipboard":
        return f"text=<{len(kwargs.get('text', ''))} chars>"
    if fn_name == "sequence":
        steps = [v for v in kwargs.values() if isinstance(v, dict)]
        names = [s.get("tool_name", "?") for s in steps]
        return f"{len(steps)} steps: {', '.join(names)}"
    arg_str = ", ".join(f"{k}={v!r}" for k, v in kwargs.items())
    if len(arg_str) > _MAX_ARG_LOG_LEN:
        arg_str = arg_str[: _MAX_ARG_LOG_LEN - 3] + "..."
    return arg_str


def logged(fn: AsyncFn) -> AsyncFn:
    """Log the wrapped MCP tool's completion with args and duration."""
    # FastMCP dispatches tool calls with keyword args only (positional
    # args land in `args` but never in practice); the log reads kwargs.
    @wraps(fn)
    async def wrapper(*args, **kwargs):
        if not log.isEnabledFor(logging.INFO):
            return await fn(*args, **kwargs)
        t0 = time.monotonic()
        try:
            return await fn(*args, **kwargs)
        finally:
            log.info(
                "tool %s(%s) — %.1fs",
                fn.__name__,
                _format_args(fn.__name__, kwargs),
                time.monotonic() - t0,
            )
    return cast(AsyncFn, wrapper)
