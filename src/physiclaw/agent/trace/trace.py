"""`Trace` — the per-day human log + per-session events.jsonl/summary.json."""

import json
import logging
import time
from collections import Counter
from pathlib import Path
from typing import Any

from physiclaw.agent.trace import store
from physiclaw.agent.trace.format import (
    _end_footer,
    brief_content,
    summarize_event,
)
from physiclaw.common import paths
from physiclaw.common.logger import (
    DailyLogWriter,
    SessionLogSidecars,
    append_stats,
    build_summary,
    ensure_readme,
    env_snapshot,
    iso_now,
    write_json_atomic,
)

log = logging.getLogger(__name__)


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
        self._daily = DailyLogWriter(store._log_dir(), "engine")
        d = store._session_dir(session_id)
        d.mkdir(parents=True, exist_ok=True)
        ensure_readme(store._sessions_dir(), store.SESSIONS_README)
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
        msg = summarize_event(event)
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
            write_json_atomic(
                store._session_dir(self.session_id) / "summary.json", summary
            )
            append_stats(paths.LOG_DIR, summary)
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
        self.conductor_turns = 0
        self.micro_calls = 0
        self.provider_time_ms = 0
        self.tool_time_ms = 0
        self.verdicts: Counter[str] = Counter()
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
            # A synthesized response is a conductor turn — no request was
            # sent, so it must not inflate provider calls or timing.
            if event.get("synthesized"):
                self.conductor_turns += 1
            else:
                self.provider_calls += 1
                self.provider_time_ms += int(event.get("elapsed_ms") or 0)
        elif name == "micro_call":
            # A real provider round-trip on the conductor's side channel:
            # counted apart from turn-loop calls, tokens folded into the
            # session's spend (that's the whole point of measuring).
            self.micro_calls += 1
            self.input_tokens += int(event.get("prompt_tokens") or 0)
            self.output_tokens += int(event.get("completion_tokens") or 0)
        elif name == "cache":
            self.input_tokens += int(event.get("total") or 0)
            self.output_tokens += int(event.get("out") or 0)
            self.cache_read += int(event.get("hit") or 0)
            self.cache_creation += int(event.get("create") or 0)
        elif name == "tool_result":
            self.tool_calls[event.get("name") or "?"] += 1
            self.tool_time_ms += int(event.get("elapsed_ms") or 0)
            if "changed" in event:
                # Three buckets, deliberately: absent (text tool) is not
                # counted at all — null means "camera couldn't decide",
                # which a miss-rate denominator must keep apart.
                c = event["changed"]
                key = (
                    "changed"
                    if c is True
                    else "unchanged"
                    if c is False
                    else "no_verdict"
                )
                self.verdicts[key] += 1
        elif name in _BLOCKED_KEYS:
            self.errors[_BLOCKED_KEYS[name]] += 1
        elif name == "tool_invalid_args":
            self.errors["invalid_args"] += 1
        elif name == "tool_unknown":
            self.errors["unknown_tool"] += 1
        elif name == "tool_error":
            self.errors["tool_errors"] += 1
            self.tool_time_ms += int(event.get("elapsed_ms") or 0)
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
            conductor_turns=self.conductor_turns,
            micro_calls=self.micro_calls,
            provider_time_ms=self.provider_time_ms,
            input_tokens=self.input_tokens,
            output_tokens=self.output_tokens,
            cache_read_tokens=self.cache_read,
            cache_creation_tokens=self.cache_creation,
            tool_calls=self.tool_calls,
            tool_time_ms=self.tool_time_ms,
            verdicts=self.verdicts,
            errors=self.errors,
            stuck_events=self.stuck_events,
            images=images,
            env=self.env,
        )


def _count_images(sid: str) -> int:
    try:
        return sum(1 for _ in (store._session_dir(sid) / "images").glob("*"))
    except OSError:
        return 0
