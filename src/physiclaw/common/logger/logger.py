"""Shared log formatting and helpers for PhysiClaw.

Exports:
    setup_logging(tag, level, file_dir=None) — colored TTY-aware
    root-logger config, optionally mirrored to a daily log file.
    logged(fn) — decorator that logs an MCP tool call's completion
    with args and duration.
"""

import datetime as dt
import logging
import os
import re
import sys
import time
from collections.abc import Awaitable, Callable
from functools import wraps
from pathlib import Path
from typing import Any, TypeVar, cast

_ANSI_RE = re.compile(r"\033\[[0-9;]*m")

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
    "runtime": "35",  # magenta — agent loop
    "setup wizard": "32",  # green — `physiclaw auto` calibration steps
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


class _DailyFileHandler(logging.Handler):
    """Mirror log records into `<dir>/<prefix>-YYYY-MM-DD.log`, uncolored.

    Re-opens the file when the date changes — the runtime is a long-lived
    process, so a single open would strand everything in the start day's
    file. Construction purges dailies older than
    CONFIG.retention.log_days."""

    def __init__(self, dir: Path, prefix: str, tag: str):
        super().__init__()
        from physiclaw.common.config import CONFIG
        from physiclaw.common.logger.retention import purge_daily_logs

        self._dir = dir
        self._prefix = prefix
        self._date = ""
        self._f = None
        self.setFormatter(_TaggedFormatter(tag, color=False))
        dir.mkdir(parents=True, exist_ok=True)
        purge_daily_logs(dir, prefix, CONFIG.retention.log_days)

    def emit(self, record: logging.LogRecord) -> None:
        try:
            today = dt.datetime.now().strftime("%Y-%m-%d")
            if today != self._date:
                if self._f is not None:
                    self._f.close()
                self._f = open(
                    self._dir / f"{self._prefix}-{today}.log",
                    "a",
                    encoding="utf-8",
                    newline="\n",
                )
                self._date = today
            self._f.write(self.format(record) + "\n")
            self._f.flush()
        except Exception:
            self.handleError(record)

    def close(self) -> None:
        try:
            if self._f is not None and not self._f.closed:
                self._f.close()
        finally:
            super().close()


def setup_logging(
    tag: str,
    level: int = logging.INFO,
    *,
    file_dir: Path | None = None,
) -> None:
    """Configure the root logger with the colored, tagged format.

    With `file_dir`, records are additionally mirrored (uncolored) to a
    daily `<tag>-YYYY-MM-DD.log` there — used by the runtime so wake
    decisions and poll errors survive for post-mortems. File-handler
    setup is fail-open: logging must never take the process down."""
    handlers: list[logging.Handler] = []
    stream = logging.StreamHandler()
    stream.setFormatter(_TaggedFormatter(tag, _colorize()))
    handlers.append(stream)
    if file_dir is not None:
        try:
            handlers.append(_DailyFileHandler(file_dir, tag, tag))
        except OSError:
            pass  # unwritable dir → stderr-only; better than not starting
    logging.basicConfig(level=level, handlers=handlers, force=True)


def make_tagged_logger(
    name: str, tag: str, level: int = logging.INFO
) -> logging.Logger:
    """A standalone logger that emits in the ``[tag]`` format, independent of
    the root handler (``propagate=False``) so its lines carry ``tag`` rather
    than the process-wide one. Lets a sub-stream — e.g. the setup wizard's
    output under ``physiclaw auto`` — be badged distinctly in interleaved
    logs. Idempotent: re-calling returns the same logger without stacking
    handlers."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(_TaggedFormatter(tag, _colorize()))
        logger.addHandler(handler)
    return logger


class LineLogStream:
    """A line-buffered file-like object that re-emits written text through a
    logger, one record per line. Pair with ``contextlib.redirect_stdout`` to
    fold a component's ``print()`` output into the tagged log stream — e.g.
    the setup wizard under ``physiclaw auto``. ANSI escapes are stripped (the
    formatter re-colours) and blank lines dropped (they'd render as an empty
    tagged prefix)."""

    def __init__(self, logger: logging.Logger):
        self._logger = logger
        self._buf = ""

    def write(self, s: str) -> int:
        self._buf += s
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            line = _ANSI_RE.sub("", line).rstrip()
            if line:
                self._logger.info(line)
        return len(s)

    def flush(self) -> None:
        pass


# Caps a tool-call log line so a 100KB clipboard body doesn't flood it.
_MAX_ARG_LOG_LEN = 80


def _format_args(fn_name: str, kwargs: dict) -> str:
    """Redact clipboard text (IM bodies, search queries, anything pasted)
    and summarize sequence steps to tool names only, since a step may
    itself be a send_to_clipboard."""
    if fn_name == "send_to_clipboard":
        return f"text=<{len(kwargs.get('text', ''))} chars>"
    if fn_name == "sequence":
        # The schema's single param is a LIST of step dicts:
        # actions=[{...}, ...] — scanning kwargs.values() for dicts finds
        # nothing and logged every batch as "0 steps".
        actions = kwargs.get("actions") or []
        names = [
            s.get("tool_name", "?") if isinstance(s, dict) else "?" for s in actions
        ]
        return f"{len(names)} steps: {', '.join(names)}"
    arg_str = ", ".join(f"{k}={v!r}" for k, v in kwargs.items())
    if len(arg_str) > _MAX_ARG_LOG_LEN:
        arg_str = arg_str[: _MAX_ARG_LOG_LEN - 3] + "..."
    return arg_str


def logged(fn: AsyncFn) -> AsyncFn:
    """Log the wrapped MCP tool's completion with args and duration.

    A failing tool logs `… — 0.0s FAILED: <err>` instead of a bare duration, so
    the server stream tells a fast failure apart from a fast success (both would
    otherwise read as an identical `tool X(…) — 0.0s`). Cancellation
    (BaseException) propagates unlogged — it isn't a tool completion."""

    # FastMCP dispatches tool calls with keyword args only (positional
    # args land in `args` but never in practice); the log reads kwargs.
    @wraps(fn)
    async def wrapper(*args, **kwargs):
        if not log.isEnabledFor(logging.INFO):
            return await fn(*args, **kwargs)
        t0 = time.monotonic()
        try:
            result = await fn(*args, **kwargs)
        except Exception as e:
            log.info(
                "tool %s(%s) — %.1fs FAILED: %s",
                fn.__name__,
                _format_args(fn.__name__, kwargs),
                time.monotonic() - t0,
                e,
            )
            raise
        log.info(
            "tool %s(%s) — %.1fs",
            fn.__name__,
            _format_args(fn.__name__, kwargs),
            time.monotonic() - t0,
        )
        return result

    return cast(AsyncFn, wrapper)
