"""Daily-log primitives — ``memory/YYYY-MM-DD.md``, the shared spelling.

The append-only daily log is the agent's activity journal (PERSISTENCE
doctrine: `[HH:MM] app: … — what you did`, one file per day). Two
packages write and read it — the engine (the model's `append_log` /
`read_logs` tools, the bootstrap memory slot, the external-stop
breadcrumb) and the conductor (a walk's payment and suspension lines,
the activation's recent-activity context) — and the two may not import
each other, so the file format lives HERE, the `verdict.py` precedent:
one spelling below the seam, both sides consume it.

`agent.engine.memory` keeps its public API by delegating; policy (what
to log, when, config-tuned window sizes) stays with the callers.
"""

import datetime as dt
import re
from pathlib import Path

from physiclaw.common import paths
from physiclaw.common.text import append_text, read_text

# Hard ceiling on how far back to scan when collecting entries. Guards
# against an indefinite loop on a near-empty memory dir; missing days
# are skipped cheaply so the cost is just calendar arithmetic.
_LOOKBACK_DAYS_CEILING = 365

# `append_log` writes `[HH:MM] …`. `load_recent_entries` rewrites the
# prefix to `[YYYY-MM-DD HH:MM] …` using the file's date so a merged-day
# view stays unambiguous about cross-day order.
_TIME_PREFIX_RE = re.compile(r"^\[(\d{2}:\d{2})\]\s*(.*)$")


def stamped(entry: str) -> str:
    """The doctrine's `[HH:MM]` prefix on one entry — callers that
    compose lines in code (the engine's external-stop breadcrumb, the
    conductor's walk lines) share this spelling instead of each
    formatting their own clock."""
    return f"[{dt.datetime.now():%H:%M}] {entry}"


def append_log(entry: str, root: Path | None = None) -> None:
    """Append a log line to today's daily file. Creates the file (with
    its `# YYYY-MM-DD` header) if needed. The caller supplies the
    `[HH:MM]` prefix (`stamped`) — the model's tool passes its own.
    `root` overrides the memory dir (the engine threads its own
    import-frozen path through; everyone else takes the live one)."""
    entry = entry.strip()
    if not entry:
        return
    mem = root if root is not None else paths.memory_dir()
    mem.mkdir(parents=True, exist_ok=True)
    today = dt.date.today().isoformat()
    path = mem / f"{today}.md"
    header = f"# {today}\n\n" if not path.exists() else ""
    append_text(path, header + entry + "\n")


def load_recent_entries(n: int, root: Path | None = None) -> str:
    """Last N log entries across daily files, most recent first.

    Walks `memory/YYYY-MM-DD.md` from today backward, accumulating
    non-empty content lines. If today's file has fewer than N entries,
    yesterday's is read too, and so on, up to the lookback ceiling.
    The `# YYYY-MM-DD` header line and blank lines are skipped — only
    actual entries count toward N.

    Each line's `[HH:MM]` prefix is rewritten to `[YYYY-MM-DD HH:MM]`
    using the file's date so the merged stream is unambiguous about
    when each entry happened. Lines without a time prefix pass through
    unchanged. Returns "" if no entries are found within the ceiling.
    """
    mem = root if root is not None else paths.memory_dir()
    today = dt.date.today()
    collected: list[str] = []
    for i in range(_LOOKBACK_DAYS_CEILING):
        if len(collected) >= n:
            break
        d = today - dt.timedelta(days=i)
        p = mem / f"{d.isoformat()}.md"
        try:
            text = read_text(p)
        except FileNotFoundError:
            continue
        # Files are append-order (oldest line first). Reverse before
        # taking from a single file so the most recent line in that
        # file appears first in the merged output.
        per_file: list[str] = []
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            per_file.append(_stamp_date(stripped, d))
        for line in reversed(per_file):
            collected.append(line)
            if len(collected) >= n:
                break
    if not collected:
        return ""
    return "\n".join(collected)


def _stamp_date(line: str, d: dt.date) -> str:
    m = _TIME_PREFIX_RE.match(line)
    if not m:
        return line
    return f"[{d.isoformat()} {m.group(1)}] {m.group(2)}"
