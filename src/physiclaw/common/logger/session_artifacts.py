"""Shared session-artifact surface for the two agent engines.

The engine (`agent/engine/trace.py`) and the claude subprocess path
(`agent/claude/session_log.py`) each write a daily human narrative plus
a per-session artifact dir whose `summary.json` shares one schema (v1)
— one `physiclaw logs` / `jq` reads both engines' sessions. This module
owns the pieces that must stay identical between the two writers:

  - `DailyLogWriter` — the daily narrative file, session banner/footer,
    and midnight rollover;
  - `build_summary` — the summary.json shape, one constructor;
  - `iso_now` / `env_snapshot` / `write_json_atomic` / `image_filename`
    / `save_image` / `ensure_readme` — the helpers both artifact dirs
    are built from.

Each engine keeps its own event accumulation (their event streams are
genuinely different); only the on-disk shape is centralized, so a schema
change is one edit here instead of a lockstep pair.
"""

import base64
import datetime as dt
import json
import logging
from pathlib import Path
from typing import Any, Mapping

from physiclaw.common.logger.logger import _open_log_file, daily_log_path
from physiclaw.common.text import read_text, write_text

log = logging.getLogger(__name__)


def iso_now() -> str:
    """Local time as ISO with millisecond precision — the one timestamp
    format across the session artifacts (`started_at`/`ended_at`,
    events.jsonl / wire.jsonl `t`). ms precision makes per-turn latency
    analysis possible without correlating against the engine log."""
    return dt.datetime.now().isoformat(timespec="milliseconds")


def env_snapshot() -> dict[str, Any]:
    """The session's environment — versions, OS, rig identity, and the
    behavior-relevant config, captured once per session (OTel calls these
    resource attributes). This is what makes a shared session dir
    self-describing: "which rig / version / camera settings produced
    this?" without asking.

    Deliberately NOT the whole CONFIG — the provider section holds API
    keys. Only secret-free sections that vary across rigs and change
    what the agent sees (camera exposure/format, image compaction)."""
    import dataclasses
    import platform as _platform
    import sys

    from physiclaw import __version__
    from physiclaw.common.config import CONFIG

    return {
        "physiclaw": __version__,
        "python": _platform.python_version(),
        "os": sys.platform,
        "platform": _platform.platform(),
        "host": _platform.node(),
        "utc_offset": dt.datetime.now().astimezone().strftime("%z"),
        "config": {
            "camera": dataclasses.asdict(CONFIG.camera),
            "compact": dataclasses.asdict(CONFIG.compact),
        },
    }


def write_json_atomic(path: Path, obj: dict[str, Any]) -> None:
    """tmp + rename so a crash mid-write can't leave a truncated summary."""
    tmp = path.with_suffix(".json.tmp")
    write_text(tmp, json.dumps(obj, ensure_ascii=False, indent=2, default=repr) + "\n")
    tmp.replace(path)


def ensure_readme(dir: Path, content: str) -> None:
    """Keep the format doc at `<dir>/README.md` current — rewritten
    whenever the shipped constant changed, so existing installs don't
    keep documenting a retired format forever. Fail-open, cheap (one
    small read per session start)."""
    path = dir / "README.md"
    try:
        if not path.exists() or read_text(path) != content:
            dir.mkdir(parents=True, exist_ok=True)
            write_text(path, content)
    except OSError:
        log.debug("sessions README write failed", exc_info=True)


# mime → filename suffix for images extracted from data-URLs. Everything
# we actually serve is JPEG via compact.scale_image_bytes, but keep the
# fallback open for PNG / WebP in case an upstream tool starts emitting
# them.
_MIME_EXT = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
}


def image_filename(turn: int, mime: str) -> str:
    """Name a captured screenshot `<HHMMSS>_<mmm>_t<turn>.<ext>`: a
    local-time stamp (hour-minute-second + milliseconds, so the names sort
    chronologically within a session) plus the turn whose request carried
    it. Shared by both engines so the `images/` layout stays identical.
    `.bin` is the fallback for an unknown mime."""
    now = dt.datetime.now()
    ext = _MIME_EXT.get(mime, ".bin")
    return f"{now:%H%M%S}_{now.microsecond // 1000:03d}_t{turn}{ext}"


def save_image(img_dir: Path, turn: int, mime: str, b64: str) -> str | None:
    """Decode a base64 screenshot and write it to
    `<img_dir>/<image_filename(turn, mime)>`; returns the filename, or
    None on undecodable data. A failed write raises OSError — each
    caller picks its own fail-open policy (RawLog stubs the reference,
    the claude writer skips its counter)."""
    try:
        raw = base64.b64decode(b64, validate=False)
    except (ValueError, TypeError):
        return None
    name = image_filename(turn, mime)
    (img_dir / name).write_bytes(raw)
    return name


# The session banner/footer rule — one width, so "operators scan either
# runtime the same way" can't drift between the two dailies.
_RULE = "=" * 60


class DailyLogWriter:
    """Append-only daily narrative log: `<dir>/<prefix>-YYYY-MM-DD.log`.

    Opens with the session banner, and owns the midnight rollover: a
    `line()` that lands past midnight closes the day's file (with a →
    marker), reopens today's (with a ← marker), and continues — markers
    in both files let a reader follow one session across days. `close()`
    is idempotent and OSError-safe: a full disk must not turn a finished
    session into a crash.
    """

    def __init__(self, dir: Path, prefix: str):
        dir.mkdir(parents=True, exist_ok=True)
        self._dir = dir
        self._prefix = prefix
        self._date = dt.datetime.now().strftime("%Y-%m-%d")
        self._f = _open_log_file(daily_log_path(dir, prefix, self._date))
        self._f.write(f"\n{_RULE}\n")
        self._f.flush()

    def line(self, msg: str) -> None:
        """Append `[HH:MM:SS] msg`, rolling to the new day's file first
        when midnight was crossed."""
        now = dt.datetime.now()
        today = now.strftime("%Y-%m-%d")
        if today != self._date:
            new_path = daily_log_path(self._dir, self._prefix, today)
            self._f.write(f"[{now:%H:%M:%S}] ROLLOVER → {new_path.name}\n")
            self._f.flush()
            self._f.close()
            self._date = today
            self._f = _open_log_file(new_path)
            self._f.write(
                f"\n[{now:%H:%M:%S}] ROLLOVER ← continued from previous day\n"
            )
        self._f.write(f"[{now:%H:%M:%S}] {msg}\n")
        self._f.flush()

    def footer(self) -> None:
        """Close the session's block in the daily narrative."""
        self._f.write(f"{_RULE}\n\n")
        self._f.flush()

    def close(self) -> None:
        try:
            if not self._f.closed:
                self._f.close()
        except OSError:
            log.debug("daily log close failed", exc_info=True)


# summary.json `errors` — the full engine-internal counter set. Both
# engines render every key (absent counters as 0) so `jq` never has to
# null-guard a field across engines.
_ERROR_KEYS = (
    "blocked_plan",
    "blocked_layout",
    "blocked_stuck",
    "invalid_args",
    "unknown_tool",
    "tool_errors",
    "correctives",
    "provider_failures",
)


def build_summary(
    *,
    sid: str,
    started_at: str,
    duration_s: float,
    model_ref: str,
    prompt_hash: str,
    triggers: list[dict],
    sentinel: str | None,
    recap: str,
    crashed: bool,
    turns: int,
    provider_calls: int,
    provider_time_ms: int,
    input_tokens: int,
    output_tokens: int,
    cache_read_tokens: int,
    cache_creation_tokens: int,
    cost_usd: float | None = None,
    tool_calls: Mapping[str, int],
    errors: Mapping[str, int],
    stuck_events: int,
    images: int,
    env: dict[str, Any],
) -> dict[str, Any]:
    """The schema-v1 summary.json dict — the single constructor both
    engines' `finalize` paths call, so the shape can't drift. `ended_at`
    is stamped here (finalize time). `errors` may carry any subset of
    `_ERROR_KEYS`; the rest render as 0. `cost_usd` (claude-only — the
    CLI reports it) is dropped when None, keeping engine summaries
    byte-stable."""
    unknown = set(errors) - set(_ERROR_KEYS)
    if unknown:
        # A counter incremented somewhere without a matching _ERROR_KEYS
        # entry would otherwise silently never reach summary.json.
        log.warning("summary errors dropped unknown counter(s): %s", sorted(unknown))
    summary: dict[str, Any] = {
        "schema": 1,
        "sid": sid,
        "started_at": started_at,
        "ended_at": iso_now(),
        "duration_s": round(duration_s, 1),
        "model_ref": model_ref,
        "provider": model_ref.partition("/")[0],
        "prompt_hash": prompt_hash,
        "triggers": triggers,
        "outcome": {
            "sentinel": sentinel,
            "recap": recap,
            "crashed": crashed,
        },
        "turns": turns,
        "provider_calls": provider_calls,
        "provider_time_ms": provider_time_ms,
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cache_read_tokens": cache_read_tokens,
            "cache_creation_tokens": cache_creation_tokens,
            "cache_hit_pct": (
                round(100 * cache_read_tokens / input_tokens, 1)
                if input_tokens
                else 0.0
            ),
        },
        "cost_usd": round(cost_usd, 4) if cost_usd is not None else None,
        "tool_calls": dict(tool_calls),
        "errors": {key: int(errors.get(key, 0)) for key in _ERROR_KEYS},
        "stuck_events": stuck_events,
        "images": images,
        "env": env,
    }
    if cost_usd is None:
        del summary["cost_usd"]
    return summary
