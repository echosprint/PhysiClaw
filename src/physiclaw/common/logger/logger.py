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
from typing import Any, TextIO, TypeVar, cast

# common.text imports only pathlib — no cycle back to logger.
from physiclaw.common.text import read_text, write_text

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
        # Append exception / stack traceback the way logging.Formatter does —
        # otherwise log.exception() / exc_info=True tracebacks vanish from
        # every sink (console, daily files, per-session runtime.log). The
        # continuation indent below then aligns the traceback under the tag.
        if record.exc_info and not record.exc_text:
            record.exc_text = self.formatException(record.exc_info)
        if record.exc_text:
            msg = f"{msg}\n{record.exc_text}"
        if record.stack_info:
            msg = f"{msg}\n{self.formatStack(record.stack_info)}"
        if "\n" in msg:
            msg = msg.replace("\n", self._cont_indent)
        if not self.color:
            # File sinks (daily logs, runtime.log, mcp.log) and piped output
            # must be plain text: strip any ANSI the MESSAGE itself carried
            # (subprocess / claude stream output, etc.) — color=False means we
            # ADD no color, not that the content has none. The `\033` pre-check
            # skips the regex on the vast majority of records, which have none.
            if "\033" in msg:
                msg = _ANSI_RE.sub("", msg)
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


# The MCP-server process logs under this tag (both its console `[physiclaw]`
# prefix and its daily-log file name); the server's mcp tee reuses it too so
# the tag lives in one place.
SERVER_LOG_TAG = "physiclaw"


def daily_log_path(log_dir: Path, prefix: str, date: str) -> Path:
    """`<dir>/<prefix>-YYYY-MM-DD.log` — the one daily-log naming scheme,
    shared by the writer (`_DailyFileHandler`) and the reader (retention's
    purge), so the format lives in a single place."""
    return log_dir / f"{prefix}-{date}.log"


def _open_log_file(path: Path) -> TextIO:
    """Append-open a log file as UTF-8 with LF newlines — the one open every
    file-backed handler here shares."""
    return open(path, "a", encoding="utf-8", newline="\n")


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
        self._f: TextIO | None = None
        self.setFormatter(_TaggedFormatter(tag, color=False))
        dir.mkdir(parents=True, exist_ok=True)
        purge_daily_logs(dir, prefix, CONFIG.retention.log_days)

    def emit(self, record: logging.LogRecord) -> None:
        try:
            today = dt.datetime.now().strftime("%Y-%m-%d")
            f = self._f
            # Reopen on a new day OR a closed handle: at interpreter exit
            # logging.shutdown() closes every handler, but the app still logs
            # (physiclaw.shutdown drives the arm, emitting DEBUG G-code) — so
            # reopen and capture that tail instead of raising per record.
            if f is None or f.closed or today != self._date:
                if f is not None and not f.closed:
                    f.close()
                f = _open_log_file(daily_log_path(self._dir, self._prefix, today))
                self._f = f
                self._date = today
            f.write(self.format(record) + "\n")
            f.flush()
        except Exception:
            self.handleError(record)

    def close(self) -> None:
        try:
            if self._f is not None and not self._f.closed:
                self._f.close()
        finally:
            super().close()


class _OwnDebugFilter(logging.Filter):
    """Admit every ``physiclaw.*`` record (down to DEBUG) plus any INFO+
    record from other loggers. Keeps a DEBUG capture full-detail for our
    own code without drowning it in third-party DEBUG spew — or leaking
    SDK request bodies some libraries log at DEBUG. A no-op on an INFO
    handler (nothing below INFO reaches it), so it's safe to attach
    unconditionally."""

    def filter(self, record: logging.LogRecord) -> bool:
        # Test the level first: INFO+ is the common case and skips the
        # per-record startswith on the majority of records.
        return record.levelno >= logging.INFO or record.name.startswith("physiclaw")


def setup_logging(
    tag: str,
    level: int = logging.INFO,
    *,
    file_dir: Path | None = None,
    file_level: int | None = None,
) -> None:
    """Configure the root logger with the colored, tagged format.

    With `file_dir`, records are additionally mirrored (uncolored) to a
    daily `<tag>-YYYY-MM-DD.log` there — used by the runtime so wake
    decisions and poll errors survive for post-mortems. `file_level`
    overrides the daily file's threshold (defaults to `level`): the
    server passes DEBUG so its camera/exposure/tune detail is on disk
    regardless of console verbosity. File-handler setup is fail-open:
    logging must never take the process down.

    The root logger is kept permissive (DEBUG); each handler gates its
    own verbosity via its own level. This is what lets a per-session
    capture handler (`attach_session_log`) — or a DEBUG daily file —
    record full DEBUG while the console stays at `level`; with a NOTSET
    root the console would print DEBUG the moment any physiclaw logger
    emitted it."""
    handlers: list[logging.Handler] = []
    stream = logging.StreamHandler()
    stream.setFormatter(_TaggedFormatter(tag, _colorize()))
    stream.setLevel(level)
    handlers.append(stream)
    if file_dir is not None:
        try:
            fh = _DailyFileHandler(file_dir, tag, tag)
            fh.setLevel(level if file_level is None else file_level)
            fh.addFilter(_OwnDebugFilter())  # gates third-party DEBUG when file<INFO
            handlers.append(fh)
        except OSError:
            pass  # unwritable dir → stderr-only; better than not starting
    logging.basicConfig(level=logging.DEBUG, handlers=handlers, force=True)


class _SessionCaptureHandler(logging.StreamHandler):
    """DEBUG mirror of the process log into one flat per-session file
    (LF newlines, uncolored). Owns its file handle so it closes cleanly
    when detached (a bare StreamHandler would leave the file open). The
    runtime process is the only caller, so the tag is fixed."""

    def __init__(self, path: Path):
        self._file = _open_log_file(path)
        super().__init__(self._file)
        self.setFormatter(_TaggedFormatter("runtime", color=False))
        self.setLevel(logging.DEBUG)
        self.addFilter(_OwnDebugFilter())

    def close(self) -> None:
        try:
            super().close()
        finally:
            try:
                if not self._file.closed:
                    self._file.close()
            except OSError:
                pass


def attach_session_log(path: Path) -> logging.Handler | None:
    """Mirror the process log stream into `path` for one session's lifetime.

    Adds a DEBUG-level handler to the root logger that captures
    ``physiclaw.*`` records (down to DEBUG) plus any INFO+ record —
    tracebacks and warnings included — so a shared session dir is
    self-contained. Returns the handler to pass to `detach_session_log`,
    or None if the file couldn't be opened (fail-open: the capture is
    best-effort and never blocks the session). Requires the permissive
    (DEBUG) root that `setup_logging` installs."""
    try:
        handler: logging.Handler = _SessionCaptureHandler(path)
    except OSError:
        return None
    logging.getLogger().addHandler(handler)
    return handler


def detach_session_log(handler: logging.Handler | None) -> None:
    """Remove and close a handler returned by `attach_session_log`. No-op
    on None (the open failed) — so callers store-and-detach unconditionally."""
    if handler is None:
        return
    logging.getLogger().removeHandler(handler)
    try:
        handler.close()
    except OSError:
        pass


# ---------- per-session log sidecars (cross-process) ----------
#
# runtime.log mirrors the runtime process's own log stream (attach_session_log
# above). mcp.log carries the MCP-SERVER process's log for the session — but
# the server is a separate process, so the two can't share an in-memory object.
# Instead the agent publishes the active session ID to a marker file, and the
# server's tee (`attach_server_mcp_tee`) reads it, resolves the id to its
# session dir, and writes mcp.log there live.


def _publish_active_session(sid: str) -> None:
    """Point the cross-process marker at session `sid` so the server's mcp
    tee writes into that session's dir. Fail-open — best-effort."""
    from physiclaw.common import paths

    marker = paths.active_session_marker()
    try:
        marker.parent.mkdir(parents=True, exist_ok=True)
        write_text(marker, sid)
    except OSError:
        log.debug("active-session marker write failed", exc_info=True)


def _clear_active_session(sid: str) -> None:
    """Remove the marker if it still names this session — guards against
    clobbering a newer session's marker. Fail-open."""
    from physiclaw.common import paths

    marker = paths.active_session_marker()
    try:
        if read_text(marker).strip() == sid:
            marker.unlink()
    except OSError:
        pass


def _active_session_dir(sid: str) -> Path | None:
    """Resolve a session id to its on-disk dir. The engine and claude writers
    use different session roots, so probe both and take the one that exists —
    exactly one does (the writer created it before publishing the marker)."""
    from physiclaw.common import paths

    for base in (paths.engine_sessions_dir(), paths.claude_sessions_dir()):
        d = base / sid
        if d.is_dir():
            return d
    return None


class SessionLogSidecars:
    """The per-session log sidecars owned by the agent session writers:
    `runtime.log` (a mirror of the runtime process log stream) and the
    active-session marker (the session id) that steers the server's mcp.log
    tee into this session's dir. One home for both so they can't drift."""

    def __init__(self, session_dir: Path):
        self._sid = session_dir.name
        self._runtime = attach_session_log(session_dir / "runtime.log")
        _publish_active_session(self._sid)

    def close(self) -> None:
        # Stop steering server logs here first, then detach the runtime
        # mirror last so any close-time warning is still captured.
        _clear_active_session(self._sid)
        detach_session_log(self._runtime)


class _SessionMcpTee(logging.Handler):
    """Server-side: tee the server's own log into the ACTIVE session's
    mcp.log. The agent publishes the active session id to a marker file
    (`_publish_active_session`); this follows it — resolving the id to its
    dir and repointing whenever the id changes — so the server's camera/
    exposure/tune detail lands live in the running session's dir. Between
    sessions (no marker) records are dropped: no session. Fail-open."""

    def __init__(self) -> None:
        from physiclaw.common import paths

        super().__init__(level=logging.DEBUG)
        self.setFormatter(_TaggedFormatter(SERVER_LOG_TAG, color=False))
        self.addFilter(_OwnDebugFilter())
        self._marker = paths.active_session_marker()  # process-invariant
        self._sid = ""
        self._f: TextIO | None = None

    def _refresh(self) -> None:
        """Repoint the mcp.log handle when the marker names a different session.
        Keying on the session id (not the marker's mtime) so a new session is
        followed even when two markers land in the same mtime tick; the id is
        resolved to its dir only on a change."""
        try:
            sid = read_text(self._marker).strip()
        except OSError:
            sid = ""
        if sid == self._sid:
            return
        self._sid = sid
        self._close_file()
        target = _active_session_dir(sid) if sid else None
        if target is not None:
            try:
                self._f = _open_log_file(target / "mcp.log")
            except OSError:
                self._f = None

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self._refresh()
            if self._f is not None:
                self._f.write(self.format(record) + "\n")
                self._f.flush()
        except Exception:
            self.handleError(record)

    def _close_file(self) -> None:
        if self._f is not None and not self._f.closed:
            try:
                self._f.close()
            except OSError:
                pass
        self._f = None

    def close(self) -> None:
        self._close_file()
        super().close()


def attach_server_mcp_tee() -> logging.Handler:
    """Server-side: attach the handler that tees the server's own log into
    the agent's active session dir as `mcp.log`. Call once after
    `setup_logging` in the MCP-server process (the runtime process uses
    `SessionLogSidecars` to publish which session is active)."""
    handler = _SessionMcpTee()
    logging.getLogger().addHandler(handler)
    return handler


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
