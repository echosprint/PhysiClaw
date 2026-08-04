"""Per-macro run counters — ``~/.physiclaw/macros/stats.json``.

One machine-written JSON file keyed by macro name (the job-stats
convention: total_runs / total_successes / total_aborts, last_run_at /
last_success_at, and consecutive_aborts — the re-rehearse signal: a
lifetime ratio says the macro is good, a consecutive streak says the app
layout changed under it). Every run records through
`runner.run_and_record` — engine and CLI rehearsal alike; read only by
`physiclaw macros stats` — never rendered into the prompt, which must
stay byte-stable across wakes.

Never fatal in either direction: a missing or corrupt file loads as
empty and is rewritten whole (atomic tmp+rename) on the next record.
Keys whose macro directory is gone are pruned at write time. Additive
fields load as absent — no schema versioning (alpha convention).
"""

import datetime as dt
import json
import logging
from pathlib import Path

from physiclaw.agent.macros.model import REASON_BAD_INPUT
from physiclaw.common import paths
from physiclaw.common.logger import write_json_atomic
from physiclaw.common.text import read_text

log = logging.getLogger(__name__)

_COUNTERS = {
    "total_runs": 0,
    "total_successes": 0,
    "total_aborts": 0,
    "consecutive_aborts": 0,
}


def stats_file() -> Path:
    return paths.macros_dir() / "stats.json"


def load() -> dict[str, dict]:
    """All recorded stats, keyed by macro name. Missing/corrupt → {}."""
    path = stats_file()
    if not path.exists():
        return {}
    try:
        data = json.loads(read_text(path))
    except (OSError, ValueError):
        log.warning("macro stats unreadable; starting fresh", exc_info=True)
        return {}
    return data if isinstance(data, dict) else {}


def record(
    name: str,
    *,
    ok: bool,
    known_names: set[str],
    step: int | None = None,
    reason: str | None = None,
    detail: str = "",
    run_id: str | None = None,
) -> None:
    """Fold one run outcome into the file. `known_names` (the currently
    existing macro dirs) prunes orphan keys so deleted macros don't leave
    stats behind. Best-effort: a write failure logs and moves on — stats
    must never take down a session."""
    data = load()
    data = {k: v for k, v in data.items() if k in known_names or k == name}
    entry = data.setdefault(name, dict(_COUNTERS))
    if not isinstance(entry, dict):
        # A hand-edited or truncated file can leave a scalar under a macro
        # key. Recording happens AFTER the gestures physically ran, so an
        # AttributeError here would swallow the step log and the screen the
        # agent needs — start that key over instead.
        log.warning("macro stats entry for %r was not a mapping; reset", name)
        entry = data[name] = dict(_COUNTERS)
    now = dt.datetime.now().isoformat(timespec="seconds")
    entry["total_runs"] = entry.get("total_runs", 0) + 1
    entry["last_run_at"] = now
    if ok:
        entry["total_successes"] = entry.get("total_successes", 0) + 1
        entry["consecutive_aborts"] = 0
        entry["last_success_at"] = now
    else:
        entry["total_aborts"] = entry.get("total_aborts", 0) + 1
        # `consecutive_aborts` is the re-rehearse signal — "the app layout
        # changed under this macro". A `bad_input` run never reached the
        # screen (a missing input, an unknown `start_at`), so it says
        # nothing about layout: count it as a run and an abort, but leave
        # the streak alone rather than let three typos in a row make a
        # healthy macro read as decayed.
        if reason != REASON_BAD_INPUT:
            entry["consecutive_aborts"] = entry.get("consecutive_aborts", 0) + 1
        entry["last_abort"] = {
            "ts": now,
            "step": step,
            "reason": reason,
            "detail": detail,
            # The forensic pointer: `physiclaw macros runs <id>` shows the
            # per-step trail of exactly this abort.
            "run": run_id,
        }
    try:
        paths.macros_dir().mkdir(parents=True, exist_ok=True)
        write_json_atomic(stats_file(), data)
    except OSError:
        log.warning("macro stats write failed", exc_info=True)
