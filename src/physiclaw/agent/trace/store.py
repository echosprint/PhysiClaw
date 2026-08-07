"""Where a session's artifacts live, how a session is named, and how
long its artifacts are kept.

Dir resolution delegates to `common.paths` per call — the same live
seam the CLI reads through, so every sink (`Trace`, `RawLog`, the
retention sweep) and every test re-point agrees on one source of
truth. `SESSIONS_README` is the format documentation shipped WITH the
data: written once to sessions/README.md and embedded in every
``physiclaw logs --save`` zip, so an analyst — human or AI agent — can
bootstrap from the artifacts alone, no source access needed. Update
alongside any schema change.
"""

import datetime as dt
import logging
import secrets
import time
from pathlib import Path

from physiclaw.common import paths
from physiclaw.common.config import CONFIG
from physiclaw.common.logger.retention import purge_daily_logs, purge_old_sessions

log = logging.getLogger(__name__)


def _log_dir() -> Path:
    return paths.engine_log_dir()


def _raw_dir() -> Path:
    return _log_dir() / "raw"  # legacy layout — purge-only, no new writes


def _sessions_dir() -> Path:
    return paths.engine_sessions_dir()


def _session_dir(sid: str) -> Path:
    return _sessions_dir() / sid


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


# Purge session dirs (and legacy raw files) older than this on session
# bootstrap. One week is generous for post-mortem debugging while
# keeping disk usage bounded for long-running operators.
_RETENTION_DAYS = CONFIG.retention.trace_days
_LOG_RETENTION_DAYS = CONFIG.retention.log_days


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
        entries = list(_raw_dir().rglob("*"))
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

    removed = purge_old_sessions(_sessions_dir(), days=days)
    if removed:
        log.info("purged %d session dir(s) older than %d days", removed, days)

    purge_daily_logs(_log_dir(), "engine", log_days)


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
