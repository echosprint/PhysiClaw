"""Engine session logs — a daily human narrative + a per-session artifact dir.

1. `Trace` — per-day human-readable log
     log/engine/engine-YYYY-MM-DD.log
   Written through the shared `DailyLogWriter`, the same one
   `agent/claude/session_log.py`'s _SessionLog uses, so operators
   scan either runtime the same way. One-line summaries via
   `_summarize(event)`; internal bookkeeping events in
   `_SILENT_EVENTS` are skipped there. Each session ends with an
   `END session=… outcome=… turns=…` footer carrying the headline
   metrics.

2. Per-session artifacts under log/engine/sessions/<sid>/ — the dir is
   self-contained (image refs are relative), so "share the bad session"
   is one directory copy:

     events.jsonl   every engine event (incl. `_SILENT_EVENTS`) as
                    structured data: {"t": iso-ms, "event": ..., turn?,
                    ...fields}. `tool_result.blocks` is summarized
                    (`result_summary`) — the full payload already lives
                    in wire.jsonl. Written by `Trace`.
     wire.jsonl     the provider round-trips, written by `RawLog`:
                    {"t","kind":"session_start",provider,model,
                     prompt_hash,tools} once, then per turn
                    {"t","turn","kind":"request","messages":[...]} and
                    {"t","turn","kind":"response","elapsed_ms","raw"}.
                    Inline base64 images are extracted to
                    images/<HHMMSS>_<mmm>_t<turn>.<ext> and replaced by
                    that relative path.
     summary.json   session metrics derived from the event stream at
                    `Trace.close()` (schema v1): outcome, turns, token/
                    cache totals, tool-call counts, error counts. The
                    `physiclaw logs` CLI reads these.
     images/        post-view screenshots, turn-tagged.

   Retention: on each session bootstrap, session dirs (and legacy
   log/engine/raw/ files) older than CONFIG.retention.trace_days are
   purged; daily engine-*.log files older than
   CONFIG.retention.log_days.
"""

import datetime as dt
import json
import logging
import secrets
import time
from collections import Counter
from pathlib import Path
from typing import Any

from physiclaw.common import paths
from physiclaw.common.config import CONFIG
from physiclaw.common.logger import (
    DailyLogWriter,
    SessionLogSidecars,
    build_summary,
    ensure_readme,
    env_snapshot,
    iso_now,
    save_image,
    write_json_atomic,
)
from physiclaw.common.logger.retention import purge_daily_logs, purge_old_sessions

log = logging.getLogger(__name__)

_LOG_DIR = paths.engine_log_dir()
_RAW_DIR = _LOG_DIR / "raw"  # legacy layout — purge-only, no new writes
_SESSIONS_DIR = paths.engine_sessions_dir()

# Purge session dirs (and legacy raw files) older than this on session
# bootstrap. One week is generous for post-mortem debugging while
# keeping disk usage bounded for long-running operators.
_RETENTION_DAYS = CONFIG.retention.trace_days
_LOG_RETENTION_DAYS = CONFIG.retention.log_days


def _session_dir(sid: str) -> Path:
    return _SESSIONS_DIR / sid


def new_sid() -> str:
    """Mint a session id: `YYYYMMDD-HHMMSS-<6 hex digits>`.

    The timestamp prefix keeps ids human-readable and lexicographically
    chronological; the random suffix (16^6 ≈ 16.7M) makes them unique —
    the STUCK-retry loop can start a new session within the same
    wall-clock second as an instantly-crashed one, and a bare-timestamp
    id would silently merge the two sessions' artifacts. The suffix
    doubles as a short handle: `physiclaw logs <suffix>` resolves a
    session by it.
    """
    return f"{dt.datetime.now():%Y%m%d-%H%M%S}-{secrets.token_hex(3)}"


# Events that are internal bookkeeping — don't surface in the human log.
# Add here when silencing a new event is cheaper than adding a dedicated
# summary branch.
_SILENT_EVENTS = frozenset({"prefix_pinned", "finish_length_warning"})

# Format documentation shipped WITH the data: written once to
# sessions/README.md and embedded in every `physiclaw logs --save` zip,
# so an analyst — human or AI agent — can bootstrap from the artifacts
# alone, no source access needed. Update alongside any schema change.
SESSIONS_README = """\
# PhysiClaw agent session logs

One directory per agent session, named `YYYYMMDD-HHMMSS-<6 hex digits>`
(sorts chronologically; the id suffix is a unique short handle). All
files are UTF-8 with LF newlines on every platform. Timestamps (`t`)
are LOCAL time with millisecond precision — the session's UTC offset is
in the `env` event / `summary.json.env.utc_offset`.

## Files per session

- `summary.json` — start here. Session metrics derived from the event
  stream (schema v1): sid, started_at/ended_at (local ISO), duration_s,
  model_ref ("provider/model"), provider, prompt_hash, triggers,
  outcome {sentinel: DONE|WAIT|IDLE|STUCK|FAIL, recap, crashed},
  turns, provider_calls, provider_time_ms, usage {input_tokens,
  output_tokens, cache_read_tokens, cache_creation_tokens,
  cache_hit_pct}, tool_calls {name: count}, errors {blocked_plan,
  blocked_layout, blocked_stuck, invalid_args, unknown_tool,
  tool_errors, correctives, provider_failures}, stuck_events, images,
  env {physiclaw/python versions, os, platform, host, utc_offset,
  config.camera, config.compact}. Missing summary.json = the session
  was killed or is still running.

- `events.jsonl` — the engine's decision stream, one JSON object per
  line: `{"t": iso-ms, "event": <type>, "turn": <int, when turn-scoped>,
  ...event fields}`. First line is always `env`. Key event types:
  `wake` (triggers), `response` (finish_reason, tool_calls requested,
  elapsed_ms), `cache` (token usage: total/hit/create/out),
  `tool_result` (name, arguments, text or result_summary),
  `tool_blocked_no_plan|_layout|_stuck` (engine refused the call),
  `stuck_warning` (loop detector fired), `bad_turn_shape` /
  `*_checkpoint` (turn rejected, corrective sent), `done`
  (sentinel + recap), `crashed`.

- `wire.jsonl` — the provider round-trips: `session_start` (provider,
  model, prompt_hash, tool schemas) once, then per turn `request`
  (full message array) and `response` (raw provider reply,
  elapsed_ms). Inline base64 images are replaced by relative paths
  into `images/`.

- `images/<HHMMSS>_<mmm>_t<turn>.<ext>` — screenshots the model saw
  (typically .jpg). The name is the local capture time (hour-minute-
  second `_` milliseconds) plus `_t<turn>` = the turn whose request
  carried the frame, so `ls` sorts them in capture order and each links
  back to its turn in `events.jsonl` / `wire.jsonl`. Example:
  `104542_123_t20.jpg` = 10:45:42.123, turn 20.

- `notes.md` — the agent's own turn-by-turn narration: one line per
  `note(summary=...)`, `- turn N — <summary>`. The fastest human read
  of what the agent thought it was doing, in order.

- `runtime.log` — a mirror of the runtime process's log stream for
  this session (uncolored, full DEBUG for `physiclaw.*` plus INFO+
  from other loggers): the turn-by-turn engine log, provider timings,
  warnings, and full tracebacks that the structured events only record
  as an error string.

- `mcp.log` — the MCP-server process's log for this session (camera,
  exposure/tune, gesture execution), written live by the server: the
  runtime publishes the active session id to a marker and the server
  resolves it to this dir and tees its log there. This is where to look
  when a view came back wrong (blur, glare, bad exposure). Absent if the
  server logged nothing, was not running, or predates this feature.

## Analysis tips

- Underperforming session? Check `images/` first — bad runs usually
  trace to what the model actually saw, not how it reasoned. Look for
  blur, glare, or a cropped/off-frame screen before blaming the
  prompt or the model.
- Failure post-mortem: `summary.json` outcome + errors first, then grep
  `events.jsonl` for `tool_blocked`/`stuck_warning`/`bad_turn_shape`
  around the last turns, then view the matching `images/`.
- Cross-session stats: every summary is one JSON file —
  `jq .usage.cache_hit_pct sessions/*/summary.json`.
- Turn timeline: events with the same `turn` value tell one
  turn's story: response → cache → tool_result(s).

Privacy: wire.jsonl carries full prompts (including the user profile /
memory) and images/ are phone screenshots. Treat a session dir as
sensitive.
"""


# ---------- public formatting helpers (shared with dispatch.py) ----------


def brief(value: Any, limit: int = 80) -> str:
    """One-line truncated repr for log output."""
    s = value if isinstance(value, str) else repr(value)
    return s if len(s) <= limit else s[: limit - 1] + "…"


def brief_args(args: dict[str, Any]) -> str:
    return ", ".join(f"{k}={brief(v, 40)}" for k, v in args.items())


def _full_args(args: dict[str, Any]) -> str:
    """Like `brief_args` but no per-value truncation — for tools whose
    args carry irreplaceable planning/decision context (`update_progress`
    steps, etc.) that get hidden by 40-char truncation."""
    return ", ".join(f"{k}={v!r}" for k, v in args.items())


def format_call_args(tool_name: str, args: dict[str, Any]) -> str:
    """Render tool-call args for the human log. `update_progress`
    bypasses the default 40-char truncation — the plan content IS the
    point of the call, and it never appears in the result line (which
    is just "progress updated"). Other tools use the brief default."""
    if tool_name == "update_progress":
        return _full_args(args)
    return brief_args(args)


def format_call_result(tool_name: str, text: str) -> str:
    """Render a tool's result text for the human log. `note` bypasses
    the default 80-char truncation — its result is `noted: <summary>`,
    a literal echo of the summary that's the sole turn-survivor under
    compaction (CONVENTION § Compaction); truncating the result hides
    the canonical record of what the agent committed to."""
    if tool_name == "note":
        return text
    return brief(text, 80)


def brief_content(content: Any) -> str:
    """Compact summary of a `ToolResultMessage.content` (DTO) or an MCP
    blocks list (raw dicts). Handles both because `dispatch.dispatch`
    summarizes after MCP→DTO conversion, but tools that bypass that path
    still pass raw blocks through."""
    from physiclaw.agent.engine.dto import ImageBlock, TextBlock

    if isinstance(content, str):
        return brief(content, 80)
    if not isinstance(content, list):
        return brief(repr(content), 80)
    parts: list[str] = []
    for b in content:
        if isinstance(b, TextBlock):
            parts.append(brief(b.text, 80))
        elif isinstance(b, ImageBlock):
            parts.append(f"<image {len(b.data_b64)}b>")
        elif isinstance(b, dict):
            t = b.get("type")
            if t == "text":
                parts.append(brief(b.get("text", ""), 80))
            elif t == "image":
                parts.append(f"<image {len(b.get('data', ''))}b>")
            elif t == "image_url":
                url = (b.get("image_url") or {}).get("url", "")
                _, _, data = url.partition(",")
                parts.append(f"<image {len(data)}b>")
            else:
                parts.append(t or "?")
        else:
            parts.append("?")
    return " + ".join(parts) or "(empty)"


# ---------- Trace ----------


class Trace:
    """Daily human log + per-session events.jsonl + summary.json.

    One instance per session. `write(event)` fans out three ways: the
    structured event lands in events.jsonl verbatim (data for analysis),
    feeds the summary accumulator, and renders as a one-liner in the
    daily log (unless silent). `close()` finalizes summary.json and the
    daily-log END footer.
    """

    def __init__(self, session_id: str):
        self.session_id = session_id
        self._daily = DailyLogWriter(_LOG_DIR, "engine")
        d = _session_dir(session_id)
        d.mkdir(parents=True, exist_ok=True)
        ensure_readme(_SESSIONS_DIR, SESSIONS_README)
        self._events = open(d / "events.jsonl", "a", encoding="utf-8", newline="\n")
        # Self-containment: runtime.log (process log mirror) + mcp.log
        # (the server's log for this session's window) via the shared
        # sidecars, plus a human-readable per-turn note history — so a
        # shared session dir explains itself without the terminal or `jq`.
        self._sidecars = SessionLogSidecars(d)
        self._notes = self._open_notes(d / "notes.md")
        self._summary = _Summary(session_id)
        self._closed = False
        # First line of every session: the environment it ran in —
        # crash-safe (flushed now), so even a killed session is
        # self-describing.
        self.write({"event": "env", **env_snapshot()})

    def _open_notes(self, path: Path):
        """Open the per-turn note-history file and write its header.
        Fail-open (returns None) — a note-log failure must never sink a
        session."""
        try:
            f = open(path, "a", encoding="utf-8", newline="\n")
            f.write(f"# Session {self.session_id} — note history\n\n")
            f.flush()
            return f
        except OSError:
            log.debug("notes.md open failed", exc_info=True)
            return None

    def write(self, event: dict[str, Any]) -> None:
        self._write_event(event)
        self._summary.observe(event)
        self._append_note(event)
        msg = _summarize(event)
        if msg is None:
            return
        self._daily.line(msg)

    def _append_note(self, event: dict[str, Any]) -> None:
        """Append a `note` tool_result's summary to notes.md — the
        session's story in the agent's own words, one line per turn.
        Fail-open; newlines in a summary are flattened to keep one line
        per turn."""
        if (
            self._notes is None
            or event.get("event") != "tool_result"
            or event.get("name") != "note"
        ):
            return
        summary = (event.get("arguments") or {}).get("summary")
        if not summary:
            return
        turn = event.get("turn")
        pfx = f"turn {turn}" if turn is not None else "note"
        line = " ".join(str(summary).split())
        try:
            self._notes.write(f"- {pfx} — {line}\n")
            self._notes.flush()
        except OSError:
            log.debug("notes.md append failed", exc_info=True)

    def close(self) -> None:
        """Finalize the session: summary.json, END footer, close files.
        Idempotent, and OSError-safe — a full disk must not turn a DONE
        session into a crash."""
        if self._closed:
            return
        self._closed = True
        try:
            summary = self._summary.finalize(
                images=_count_images(self.session_id),
            )
            write_json_atomic(_session_dir(self.session_id) / "summary.json", summary)
            self._daily.line(_end_footer(summary))
        except OSError:
            log.warning("session summary write failed", exc_info=True)
        finally:
            # Consolidate the MCP-server log window + detach the runtime
            # mirror (ordering owned by SessionLogSidecars) before closing
            # the trace files below.
            self._sidecars.close()
            self._daily.close()
            for f in (self._events, self._notes):
                try:
                    if f is not None and not f.closed:
                        f.close()
                except OSError:
                    pass

    def _write_event(self, event: dict[str, Any]) -> None:
        """Append the structured event to events.jsonl. `tool_result`
        payloads are summarized — `blocks` may carry base64 screenshots
        whose full bytes already live in wire.jsonl/images; duplicating
        them here would double the session's disk cost for nothing."""
        obj: dict[str, Any] = {"t": iso_now(), **event}
        if event.get("event") == "tool_result" and "blocks" in obj:
            obj["result_summary"] = brief_content(obj.pop("blocks"))
        try:
            line = json.dumps(obj, ensure_ascii=False, default=repr)
            self._events.write(line + "\n")
            self._events.flush()
        except (OSError, TypeError, ValueError):
            log.warning("events.jsonl write failed", exc_info=True)


# ---------- session summary (derived from the event stream) ----------


# Corrective events: the engine rejected a turn and re-asked. Grouped so
# summary.errors.correctives counts every rejection kind uniformly.
_CORRECTIVE_EVENTS = frozenset(
    {
        "bad_turn_shape",
        "checkpoint_corrective",
        "stuck_reflection",
        "pitfall_checkpoint",
        "memory_cue_checkpoint",
    }
)
# Events that mirror a `session.stuck_events += 1` in policy.py — keep in
# sync with those increment sites.
_STUCK_EVENTS = frozenset({"stuck_warning", "tool_blocked_stuck"})
_BLOCKED_KEYS = {
    "tool_blocked_no_plan": "blocked_plan",
    "tool_blocked_layout": "blocked_layout",
    "tool_blocked_stuck": "blocked_stuck",
}


class _Summary:
    """Accumulates session metrics from the events flowing through
    `Trace.write` — zero extra plumbing in the engine; everything in
    summary.json is derivable from the stream (loop.py enriches two
    events for it: `response.elapsed_ms` and `cache.out`)."""

    def __init__(self, sid: str):
        self.sid = sid
        self.started_at = iso_now()
        self._start_mono = time.monotonic()
        self.model_ref = ""
        self.prompt_hash = ""
        self.triggers: list[dict] = []
        self.sentinel: str | None = None
        self.recap = ""
        self.crashed = False
        self.max_turn = -1
        self.provider_calls = 0
        self.provider_time_ms = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.cache_read = 0
        self.cache_creation = 0
        self.tool_calls: Counter[str] = Counter()
        self.errors: Counter[str] = Counter()
        self.stuck_events = 0
        self.env: dict[str, Any] = {}

    def observe(self, event: dict[str, Any]) -> None:
        name = event.get("event", "")
        turn = event.get("turn")
        if isinstance(turn, int) and turn > self.max_turn:
            self.max_turn = turn
        if name == "env":
            self.env = {k: v for k, v in event.items() if k != "event"}
        elif name == "wake":
            self.model_ref = event.get("model_ref") or ""
            self.triggers = event.get("triggers") or []
        elif name == "prefix_pinned":
            self.prompt_hash = event.get("hash") or ""
        elif name == "response":
            self.provider_calls += 1
            self.provider_time_ms += int(event.get("elapsed_ms") or 0)
        elif name == "cache":
            self.input_tokens += int(event.get("total") or 0)
            self.output_tokens += int(event.get("out") or 0)
            self.cache_read += int(event.get("hit") or 0)
            self.cache_creation += int(event.get("create") or 0)
        elif name == "tool_result":
            self.tool_calls[event.get("name") or "?"] += 1
        elif name in _BLOCKED_KEYS:
            self.errors[_BLOCKED_KEYS[name]] += 1
        elif name == "tool_invalid_args":
            self.errors["invalid_args"] += 1
        elif name == "tool_unknown":
            self.errors["unknown_tool"] += 1
        elif name == "tool_error":
            self.errors["tool_errors"] += 1
        elif name == "provider_failed":
            self.errors["provider_failures"] += 1
        elif name == "done":
            self.sentinel = event.get("sentinel")
            self.recap = event.get("recap") or ""
        elif name == "crashed":
            self.crashed = True
        if name in _CORRECTIVE_EVENTS:
            self.errors["correctives"] += 1
        if name in _STUCK_EVENTS:
            self.stuck_events += 1

    def finalize(self, *, images: int) -> dict[str, Any]:
        return build_summary(
            sid=self.sid,
            started_at=self.started_at,
            duration_s=time.monotonic() - self._start_mono,
            model_ref=self.model_ref,
            prompt_hash=self.prompt_hash,
            triggers=self.triggers,
            sentinel=self.sentinel,
            recap=self.recap,
            crashed=self.crashed,
            turns=self.max_turn + 1,
            provider_calls=self.provider_calls,
            provider_time_ms=self.provider_time_ms,
            input_tokens=self.input_tokens,
            output_tokens=self.output_tokens,
            cache_read_tokens=self.cache_read,
            cache_creation_tokens=self.cache_creation,
            tool_calls=self.tool_calls,
            errors=self.errors,
            stuck_events=self.stuck_events,
            images=images,
            env=self.env,
        )


def fmt_tokens(n: int) -> str:
    """Human token count: 980 → '980', 9_800 → '9.8k', 1_200_000 → '1.2M'."""
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}k"
    return str(n)


def _end_footer(summary: dict[str, Any]) -> str:
    """The daily-log session footer — headline metrics on one greppable line."""
    u = summary["usage"]
    return (
        f"END session={summary['sid']} "
        f"outcome={summary['outcome']['sentinel'] or '(none)'} "
        f"turns={summary['turns']} duration={summary['duration_s']:.0f}s "
        f"tokens={fmt_tokens(u['input_tokens'])}/{fmt_tokens(u['output_tokens'])} "
        f"cache={u['cache_hit_pct']:.0f}% "
        f"tools={sum(summary['tool_calls'].values())}"
    )


def _count_images(sid: str) -> int:
    try:
        return sum(1 for _ in (_session_dir(sid) / "images").glob("*"))
    except OSError:
        return 0


# ---------- event → one-line summary ----------


def _summarize(event: dict[str, Any]) -> str | None:  # noqa: C901 — flat dispatch
    name = event.get("event", "")
    t = event.get("turn")
    pfx = f"turn {t}: " if t is not None else ""

    if name == "wake":
        triggers = event.get("triggers") or []
        sources = [x.get("source") or "?" for x in triggers]
        return (
            f"WAKE session={event.get('session', '?')} "
            f"model={event.get('model_ref', '?')} triggers={sources}"
        )
    if name == "env":
        return (
            f"env physiclaw={event.get('physiclaw', '?')} "
            f"python={event.get('python', '?')} {event.get('platform', '?')} "
            f"host={event.get('host', '?')} utc{event.get('utc_offset', '?')}"
        )
    if name == "tools_loaded":
        return (
            f"tools: {len(event.get('mcp') or [])} MCP + "
            f"{len(event.get('local') or [])} local"
        )
    if name == "request":
        return f"{pfx}request ({event.get('message_count', '?')} messages)"
    if name == "response":
        calls = [c.get("name") for c in event.get("tool_calls") or []]
        return f"{pfx}response finish={event.get('finish_reason', '?')} calls={calls}"
    if name == "cache":
        return (
            f"{pfx}cache hit={event.get('hit', 0)} create={event.get('create', 0)} "
            f"new={event.get('new', 0)} / total={event.get('total', 0)}"
        )
    if name == "tool_result":
        tool_name = event.get("name", "?")
        args = format_call_args(tool_name, event.get("arguments") or {})
        if "text" in event:
            result = format_call_result(tool_name, event["text"])
        else:
            result = brief_content(event.get("blocks") or [])
        return f"{pfx}{tool_name}({args}) → {result}"
    if name == "tool_invalid_args":
        return f"{pfx}{event.get('name', '?')} invalid args: {brief(event.get('error', ''), 200)}"
    if name == "tool_unknown":
        return f"{pfx}{event.get('name', '?')} unknown tool"
    if name == "tool_error":
        return f"{pfx}{event.get('name', '?')} failed: {brief(event.get('error', ''), 200)}"
    if name == "violations":
        return f"{pfx}violations {event.get('codes') or []}"
    if name == "log_append":
        return f"{pfx}log: {brief(event.get('entry', ''), 200)}"
    if name == "memory_save":
        return f"{pfx}memory: {brief(event.get('text', ''), 200)}"
    if name == "sentinel":
        return f"{pfx}SENTINEL {event.get('name', '?')} — {event.get('recap', '')}"
    if name == "done":
        return (
            f"OUTCOME: {event.get('sentinel') or '(none)'} — {event.get('recap', '')}"
        )
    if name == "crashed":
        return "CRASHED"
    if name == "provider_failed":
        return f"{pfx}provider failed: {brief(event.get('error', ''), 200)}"
    if name == "prefix_drift":
        return (
            f"{pfx}!! PREFIX DRIFT "
            f"expected={event.get('expected', '')[:12]}… "
            f"actual={event.get('actual', '')[:12]}…"
        )
    if name in _SILENT_EVENTS:
        return None
    # Fallback — compact repr so nothing disappears silently.
    return f"event {name}: {brief(repr(event), 200)}"


# Public alias — the `physiclaw logs` CLI renders a session's narrative
# from its events.jsonl with the same formatter the daily log uses.
summarize_event = _summarize


# ---------- RawLog: per-session structured capture ----------


class RawLog:
    """Per-session JSONL sink for later analysis.

    Emits `session_start` once, then one line per provider round-trip
    (request OR response). Open inside the engine's try/finally — call
    `close()` on session end.
    """

    def __init__(self, session_id: str):
        d = _session_dir(session_id)
        self._image_dir = d / "images"
        self._image_dir.mkdir(parents=True, exist_ok=True)
        _purge_old()
        self.session_id = session_id
        self.path = d / "wire.jsonl"
        self._f = open(self.path, "a", encoding="utf-8", newline="\n")
        # The turn currently being scrubbed — set by write_request before
        # _scrub_images so extracted images carry their turn in the name.
        self._turn = -1

    def write_session_start(
        self,
        *,
        provider: str,
        model: str,
        prompt_hash: str,
        tools: list[dict],
    ) -> None:
        # Tools don't change mid-session (engine builds the registry once
        # at bootstrap), so logging them once at start is sufficient and
        # keeps per-turn records lean.
        self._emit(
            "session_start",
            provider=provider,
            model=model,
            prompt_hash=prompt_hash,
            tools=tools,
        )

    def write_request(self, turn: int, messages: list[dict]) -> None:
        self._turn = turn
        self._emit("request", turn=turn, messages=self._scrub_images(messages))

    def write_response(
        self,
        turn: int,
        raw: dict[str, Any],
        *,
        elapsed_ms: int,
    ) -> None:
        self._emit("response", turn=turn, elapsed_ms=elapsed_ms, raw=raw)

    def close(self) -> None:
        if not self._f.closed:
            self._f.close()

    def _emit(self, kind: str, **data: Any) -> None:
        obj = {"t": iso_now(), "kind": kind, **data}
        self._f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        self._f.flush()

    def _persist_image(self, mime: str, b64_data: str) -> str:
        """Decode `b64_data`, write to
        `sessions/<sid>/images/<HHMMSS>_<mmm>_t<turn><ext>`, return the path
        relative to the session dir (so wire.jsonl + images move together
        when the dir is copied). The `image_filename` stamp sorts the frames
        chronologically and its `_t<turn>` tag links each straight to its
        turn. Returns "" on decode failure so the caller can fall back to a
        byte-count stub."""
        name = save_image(self._image_dir, self._turn, mime, b64_data)
        return f"images/{name}" if name else ""

    def _scrub_images(self, messages: list[dict]) -> list[dict]:
        """Copy of `messages` with inline base64 image data replaced by
        a reference to an on-disk file under `images/<HHMMSS>_<mmm>_t<turn>.ext`
        in the session dir. No cross-request dedup, by design: a screen still
        in context is re-persisted each request it appears in (a fresh stamp
        each time), so the frames sort chronologically on disk for debugging.

        Handles two wire shapes (recognized at the block level, not the
        provider level):

          - OpenAI: `{"type": "image_url", "image_url": {"url": "data:..."}}`
          - Anthropic: `{"type": "image", "source": {"type": "base64",
            "media_type": "...", "data": "..."}}`

        On decode failure, falls back to a byte-count stub so the raw
        log still distinguishes an image from a tap result."""
        out: list[dict] = []
        for m in messages:
            c = m.get("content")
            if not isinstance(c, list):
                out.append(m)
                continue
            new_c: list[dict] = []
            for b in c:
                if isinstance(b, dict):
                    new_c.append(self._scrub_block(b))
                else:
                    new_c.append(b)
            out.append({**m, "content": new_c})
        return out

    def _scrub_block(self, b: dict) -> dict:
        """Scrub one content block; handles OpenAI `image_url`, Anthropic
        `image`, and Anthropic `tool_result` (whose nested `content` may
        itself contain image blocks). Pass-through for everything else
        (text, tool_use, …)."""
        bt = b.get("type")
        if bt == "image_url":
            url = (b.get("image_url") or {}).get("url", "")
            if not url.startswith("data:"):
                return b
            head, _, data = url.partition(",")
            mime = head[5:].partition(";")[0]
            rel = self._persist_image(mime, data) if data else ""
            scrubbed = rel or f"{head},<{len(data)}b unreadable>"
            return {"type": "image_url", "image_url": {"url": scrubbed}}
        if bt == "image":
            src = b.get("source") or {}
            if src.get("type") != "base64":
                return b
            data = src.get("data") or ""
            mime = src.get("media_type") or "image/jpeg"
            rel = self._persist_image(mime, data) if data else ""
            scrubbed_src = (
                {"type": "ref", "ref": rel}
                if rel
                else {"type": "base64", "byte_count": len(data)}
            )
            return {"type": "image", "source": scrubbed_src}
        if bt == "tool_result":
            inner = b.get("content")
            if isinstance(inner, list):
                scrubbed_inner = [
                    self._scrub_block(x) if isinstance(x, dict) else x for x in inner
                ]
                return {**b, "content": scrubbed_inner}
            return b
        return b


def _purge_old(
    *,
    days: int = _RETENTION_DAYS,
    log_days: int = _LOG_RETENTION_DAYS,
) -> None:
    """Session-bootstrap retention sweep, three targets:

      1. legacy files under `log/engine/raw/` (pre-sessions layout —
         purge-only until they age out),
      2. `sessions/<sid>/` dirs whose newest file is older than `days`,
      3. daily `engine-*.log` files older than `log_days`.

    mtime beats filename-date parsing because it tolerates clock skew
    and handles files appended to long after creation. Fail-open
    throughout — retention must never take down a session."""
    cutoff = time.time() - days * 86400
    removed = 0
    try:
        entries = list(_RAW_DIR.rglob("*"))
    except OSError:
        entries = []
    for path in entries:
        try:
            if path.is_file() and path.stat().st_mtime < cutoff:
                path.unlink()
                removed += 1
        except OSError:
            pass
    if removed:
        log.info("purged %d legacy raw log file(s) older than %d days", removed, days)

    removed = purge_old_sessions(_SESSIONS_DIR, days=days)
    if removed:
        log.info("purged %d session dir(s) older than %d days", removed, days)

    purge_daily_logs(_LOG_DIR, "engine", log_days)
