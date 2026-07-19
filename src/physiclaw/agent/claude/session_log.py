"""Per-wake logging for the claude engine: daily narrative + session dir.

One `_SessionLog` per `claude -p` wake writes two surfaces:
  - the human-readable daily narrative, `log/claude/claude-YYYY-MM-DD.log`
    (every stream-json event summarized, images elided);
  - a per-session artifact dir, `sessions/<sid>/` with `summary.json`
    (engine schema v1 — one `physiclaw logs` / `jq` reads both engines'
    sessions) and `images/` (the screenshots the model actually saw).

Extracted from `spawn.py`, which keeps only the subprocess lifecycle;
this module owns everything written to disk about a wake.
"""

import base64
import datetime as dt
import logging
import time
from collections import Counter

# Shared session-artifact helpers — reused verbatim so the claude sessions'
# summary.json + images/ stay byte-compatible with the engine's (one
# `physiclaw logs` / `jq` reads both).
from physiclaw.agent.engine.trace import (
    _env_snapshot,
    _write_json_atomic,
    image_filename,
    purge_old_sessions,
)
from physiclaw.agent.runtime.hook import Trigger
from physiclaw.agent.runtime.sentinel import STATUSES, parse_sentinel
from physiclaw.common import paths
from physiclaw.common.config import CONFIG
from physiclaw.common.logger import SessionLogSidecars
from physiclaw.common.logger.retention import purge_daily_logs
from physiclaw.common.text import read_text, write_text

log = logging.getLogger(__name__)

LOG_DIR = paths.claude_log_dir()


def _redact_images(content):
    """Replace base64 image data with a length placeholder so logs stay readable."""
    if not isinstance(content, list):
        return content
    out = []
    for item in content:
        if isinstance(item, dict) and item.get("type") == "image":
            src = item.get("source") or {}
            data = src.get("data", "")
            out.append({**item, "source": {**src, "data": f"<{len(data)}b elided>"}})
        else:
            out.append(item)
    return out


# Per-session artifact dir docs — a claude-specific README (the engine's
# `SESSIONS_README` also documents events.jsonl / wire.jsonl, which this
# engine doesn't emit). summary.json shares the engine's schema v1, so the
# same `physiclaw logs` / `jq` reads both engines' sessions.
_CLAUDE_SESSIONS_README = """\
# PhysiClaw claude-code session logs

One directory per `claude -p` wake, `YYYYMMDD-HHMMSS-<6 hex digits>`. The
human-readable narrative for all wakes of a day lives alongside, in
`../claude-YYYY-MM-DD.log`. All files are UTF-8 with LF newlines on every
platform.

- `summary.json` — session metrics, schema v1 (shared with the engine's
  sessions): sid, started/ended, duration_s, model_ref, provider, triggers,
  outcome{sentinel,recap,crashed}, turns, usage{tokens,cache_hit_pct},
  cost_usd, tool_calls{name:count}, errors, images, env. Missing = the
  session was killed. Cross-session: `jq .usage.cache_hit_pct */summary.json`.
- `images/<HHMMSS>_<mmm>_t<turn>.<ext>` — screenshots the model saw
  (typically .jpg). Name = local capture time (hour-minute-second `_`
  milliseconds) + `_t<turn>` (the turn it belongs to), so they sort in
  capture order. Example: `104542_123_t20.jpg` = 10:45:42.123, turn 20.
- `runtime.log` — a mirror of the runtime process log stream for this
  wake (uncolored, full DEBUG for `physiclaw.*` plus INFO+ from other
  loggers): warnings and full tracebacks the daily narrative elides.
- `mcp.log` — the MCP-server process's log for this wake (camera,
  exposure/tune, gesture execution), written live by the server via an
  active-session marker. Where to look when a view came back wrong.
  Absent if the server logged nothing or was not running.

Underperforming wake? Check `images/` first — bad runs usually trace to
what the model actually saw, not how it reasoned. Look for blur, glare,
or a cropped/off-frame screen before blaming the prompt or the model.

Privacy: images/ are phone screenshots. Treat a session dir as sensitive.
"""


def _ensure_claude_sessions_readme() -> None:
    """Keep the format doc current — rewritten whenever the shipped
    constant changed, so existing installs don't keep documenting a
    retired format. Fail-open, cheap."""
    path = paths.claude_sessions_dir() / "README.md"
    try:
        if not path.exists() or read_text(path) != _CLAUDE_SESSIONS_README:
            path.parent.mkdir(parents=True, exist_ok=True)
            write_text(path, _CLAUDE_SESSIONS_README)
    except OSError:
        log.debug("claude sessions README write failed", exc_info=True)


class _ClaudeSummary:
    """Accumulate a wake's metrics from the stream-json into the engine's
    summary.json schema (v1), so both engines' sessions read with one tool.
    Token fields sum the per-assistant `usage` — matching the engine's
    per-turn accumulation — and `cost_usd` (claude-only) rides along."""

    def __init__(
        self,
        sid: str,
        triggers: list[Trigger],
        *,
        model_ref: str,
        prompt_hash: str,
    ):
        self.sid = sid
        self.started_at = dt.datetime.now().isoformat(timespec="milliseconds")
        self._start_mono = time.monotonic()
        self.model_ref = model_ref
        self.prompt_hash = prompt_hash
        self.triggers = [
            {"source": t.source, "description": t.description} for t in triggers
        ]
        self.sentinel: str | None = None
        self.recap = ""
        self.crashed = False
        # Tokens/cost/time come from the `result` event — the single
        # authoritative cumulative record. Per-`assistant`-event `usage` is
        # PARTIAL (streamed) and REPEATS the same `message.id` across a
        # message's content-block events, so summing it double-counts.
        self.provider_time_ms = 0
        self.cost_usd = 0.0
        self.usage: dict = {}
        self._msg_ids: set[str] = set()  # distinct assistant messages = calls
        self.tool_calls: Counter[str] = Counter()
        self.tool_errors = 0
        self.env = _env_snapshot()

    def observe(self, data: dict) -> None:
        t = data.get("type")
        if t == "assistant":
            msg = data.get("message", {})
            mid = msg.get("id")
            if mid:
                self._msg_ids.add(mid)
            # Each content block appears in exactly one assistant event, so
            # counting tool_use across events doesn't double-count.
            for b in msg.get("content", []):
                if b.get("type") == "tool_use":
                    self.tool_calls[b.get("name") or "?"] += 1
        elif t == "user":
            for b in data.get("message", {}).get("content", []):
                if b.get("type") == "tool_result" and b.get("is_error"):
                    self.tool_errors += 1
        elif t == "result":
            self.provider_time_ms = int(data.get("duration_api_ms") or 0)
            self.cost_usd = float(data.get("total_cost_usd") or 0.0)
            self.usage = data.get("usage") or {}

    def finalize(self, *, images: int) -> dict:
        u = self.usage
        cr = int(u.get("cache_read_input_tokens") or 0)
        cc = int(u.get("cache_creation_input_tokens") or 0)
        total_in = int(u.get("input_tokens") or 0) + cr + cc
        calls = len(self._msg_ids)
        return {
            "schema": 1,
            "sid": self.sid,
            "started_at": self.started_at,
            "ended_at": dt.datetime.now().isoformat(timespec="milliseconds"),
            "duration_s": round(time.monotonic() - self._start_mono, 1),
            "model_ref": self.model_ref,
            "provider": self.model_ref.partition("/")[0],
            "prompt_hash": self.prompt_hash,
            "triggers": self.triggers,
            "outcome": {
                "sentinel": self.sentinel,
                "recap": self.recap,
                "crashed": self.crashed,
            },
            # Distinct assistant messages = provider round-trips; on the
            # claude path one turn == one call, so both keys carry the same
            # value (engine parity — there a turn can span calls).
            "turns": calls,
            "provider_calls": calls,
            "provider_time_ms": self.provider_time_ms,
            "usage": {
                "input_tokens": total_in,
                "output_tokens": int(u.get("output_tokens") or 0),
                "cache_read_tokens": cr,
                "cache_creation_tokens": cc,
                "cache_hit_pct": (round(100 * cr / total_in, 1) if total_in else 0.0),
            },
            "cost_usd": round(self.cost_usd, 4),
            "tool_calls": dict(self.tool_calls),
            "errors": {
                # Engine-internal refusal counters don't apply to the claude
                # path (Claude Code owns its own loop); keep the keys for
                # schema parity, populate the two that are observable here.
                "blocked_plan": 0,
                "blocked_layout": 0,
                "blocked_stuck": 0,
                "invalid_args": 0,
                "unknown_tool": 0,
                "tool_errors": self.tool_errors,
                "correctives": 0,
                "provider_failures": 0,
            },
            "stuck_events": 0,
            "images": images,
            "env": self.env,
        }


class _SessionLog:
    """One claude wake's logs: the daily human narrative
    (`claude-YYYY-MM-DD.log`) plus a per-session artifact dir
    (`sessions/<sid>/summary.json` + `images/`), mirroring the engine."""

    def __init__(
        self,
        sid: str,
        triggers: list[Trigger],
        *,
        model_ref: str,
        prompt_hash: str,
    ):
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        purge_daily_logs(LOG_DIR, "claude", CONFIG.retention.log_days)
        self._date = dt.datetime.now().strftime("%Y-%m-%d")
        self._last_text = ""  # most recent assistant text block, for sentinel check
        self._f = open(
            LOG_DIR / f"claude-{self._date}.log",
            "a",
            encoding="utf-8",
            newline="\n",
        )
        self._f.write(f"\n{'=' * 60}\n")

        # Per-session artifact dir + running metrics.
        self._sdir = paths.claude_sessions_dir() / sid
        self._img_dir = self._sdir / "images"
        self._img_dir.mkdir(parents=True, exist_ok=True)
        # Per-session log sidecars: runtime.log (process log mirror) +
        # mcp.log (the server's log for this wake's window). Shared with
        # the engine writer so the two can't drift.
        self._sidecars = SessionLogSidecars(self._sdir)
        _ensure_claude_sessions_readme()
        purge_old_sessions(
            paths.claude_sessions_dir(), days=CONFIG.retention.trace_days
        )
        self._summary = _ClaudeSummary(
            sid, triggers, model_ref=model_ref, prompt_hash=prompt_hash
        )
        self._turn = 0  # advanced per assistant MESSAGE; tags extracted images
        self._seen_msg_ids: set[str] = set()  # dedup streamed content-block events
        self._image_counter = 0
        self._closed = False

        sources = [t.source or "?" for t in triggers]
        self._write(f"WAKE triggers={sources}")

    def event(self, data: dict) -> dict | None:
        """Log a stream-json event. Returns the data if it's a result.

        Every event is summarized to the daily file, feeds the session
        summary, and (for tool-result screenshots) is extracted to
        images/. Assistant text additionally narrates to the runtime log.
        """
        if data.get("type") == "assistant":
            # One assistant MESSAGE streams as several assistant EVENTS
            # sharing message.id; advance the turn only on a new message,
            # so image tags match summary.json's message-id-deduped turn
            # count (summary counts `len(_msg_ids)` — mirror it here).
            # Fall back to per-event advance only when there is no id to
            # dedup on (never in a real Claude stream).
            mid = data.get("message", {}).get("id")
            if not mid:
                self._turn += 1
            elif mid not in self._seen_msg_ids:
                self._seen_msg_ids.add(mid)
                self._turn += 1
            # Sentinel evidence for done(): capture assistant text here
            # (not in the render-only _summarize) — the LAST non-empty
            # text block wins, whichever event carried it.
            for b in data.get("message", {}).get("content", []):
                if b.get("type") == "text" and b.get("text", "").strip():
                    self._last_text = b["text"]
        summary = self._summarize(data)
        if summary:
            self._write(summary)
        self._summary.observe(data)
        self._extract_images(data)
        self._forward_to_runtime(data)
        return data if data.get("type") == "result" else None

    def raw(self, text: str) -> None:
        self._write(f"raw: {text[:500]}")

    def done(self, returncode: int | str) -> str:
        """Write OUTCOME + EXIT bookends, record them on the summary, and
        return the OUTCOME status.

        Trust the sentinel only when the process exited cleanly (code 0);
        otherwise the run crashed even if the agent claimed DONE earlier.
        """
        last_line = next(
            (line for line in reversed(self._last_text.splitlines()) if line.strip()),
            "",
        )
        status, recap = parse_sentinel(last_line) if returncode == 0 else (None, "")
        if not status:
            status = "UNDONE"
            recap = (last_line or "(no text)").strip()[:200]
        self._summary.sentinel = status if status in STATUSES else None
        self._summary.recap = recap
        self._summary.crashed = returncode != 0
        self._write(f"OUTCOME: {status} - {recap}")
        self._write(f"EXIT code={returncode}")
        self._f.write(f"{'=' * 60}\n\n")
        return status

    def close(self) -> None:
        """Finalize summary.json, then close the daily-log handle. Idempotent
        and OSError-safe — a full disk must not turn a DONE wake into a crash."""
        if self._closed:
            return
        self._closed = True
        try:
            _write_json_atomic(
                self._sdir / "summary.json",
                self._summary.finalize(images=self._image_counter),
            )
        except OSError:
            log.warning("claude session summary write failed", exc_info=True)
        finally:
            self._sidecars.close()
            if not self._f.closed:
                self._f.close()

    def _extract_images(self, data: dict) -> None:
        """Decode base64 screenshots from tool_result blocks to
        images/<HHMMSS>_<mmm>_t<turn>.<ext> — the screenshots the model
        actually saw, the biggest post-mortem win over the elided daily log."""
        if data.get("type") != "user":
            return
        for b in data.get("message", {}).get("content", []):
            if b.get("type") != "tool_result":
                continue
            content = b.get("content")
            if not isinstance(content, list):
                continue
            for c in content:
                if not isinstance(c, dict) or c.get("type") != "image":
                    continue
                src = c.get("source") or {}
                if src.get("type") == "base64" and src.get("data"):
                    self._save_image(src.get("media_type") or "image/jpeg", src["data"])

    def _save_image(self, mime: str, b64: str) -> None:
        try:
            raw = base64.b64decode(b64, validate=False)
        except (ValueError, TypeError):
            return
        n = self._image_counter + 1
        name = image_filename(self._turn, mime)
        try:
            (self._img_dir / name).write_bytes(raw)
        except OSError:
            return  # don't advance the counter for a write that didn't land
        self._image_counter = n

    def _write(self, msg: str) -> None:
        now = dt.datetime.now()
        today = now.strftime("%Y-%m-%d")
        if today != self._date:
            # Crossed midnight — close current file, continue in today's file.
            # Markers in both files let a reader follow the session across days.
            self._f.write(f"[{now:%H:%M:%S}] ROLLOVER → claude-{today}.log\n")
            self._f.flush()
            self._f.close()
            self._date = today
            self._f = open(
                LOG_DIR / f"claude-{today}.log",
                "a",
                encoding="utf-8",
                newline="\n",
            )
            self._f.write(
                f"\n[{now:%H:%M:%S}] ROLLOVER ← continued from previous day\n"
            )
        self._f.write(f"[{now:%H:%M:%S}] {msg}\n")
        self._f.flush()

    def _forward_to_runtime(self, data: dict) -> None:
        """Forward the high-signal subset of events to runtime stderr so
        the daemon log is followable without tailing the detail file.

        Only assistant TEXT blocks are forwarded — tool_use / tool_result
        are already visible in the MCP server's own log, and the final
        `result` event is logged from `spawn_claude`'s exit path.
        """
        if data.get("type") != "assistant":
            return
        for b in data.get("message", {}).get("content", []):
            if b.get("type") == "text" and b.get("text", "").strip():
                first = b["text"].strip().splitlines()[0][:200]
                log.info("claude: %s", first)
                return

    def _summarize(self, data: dict) -> str | None:
        """Render one stream-json event as a daily-log line (or None).
        Pure rendering — sentinel evidence is captured in `event()`."""
        t = data.get("type", "")

        if t == "assistant":
            parts = []
            for b in data.get("message", {}).get("content", []):
                if b.get("type") == "tool_use":
                    parts.append(
                        f"tool_use: {b['name']} {str(b.get('input', ''))[:1000]}"
                    )
                elif b.get("type") == "text" and b.get("text", "").strip():
                    parts.append(f"text: {b['text'][:1000]}")
                elif b.get("type") == "thinking" and b.get("thinking", "").strip():
                    parts.append(f"thinking: {b['thinking'][:2000]}")
            return " | ".join(parts) if parts else None

        if t == "user":
            for b in data.get("message", {}).get("content", []):
                if b.get("type") == "tool_result":
                    return f"tool_result: {str(_redact_images(b.get('content', '')))[:1000]}"

        if t == "result":
            return f"result: turns={data.get('num_turns', '?')} {str(data.get('result', ''))[:2000]}"

        return None
