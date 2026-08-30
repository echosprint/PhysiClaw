"""Memory retrieval + update for the engine.

Read paths:
  - `load_user` — `memory/USER.md`. Routed through the doctrine renderer
    so it appears as a slot inside `# Doctrine`.
  - `load_persistent` — `memory/memory.md`. Auto-injected at session start.
  - `load_recent_entries` — last N entries across `memory/YYYY-MM-DD.md`
    daily logs, walking back through prior days when the latest file
    has fewer than N. Fetched on demand via `read_logs`; injecting
    them every wake would burn context.

Write paths used by the local tool handlers: `append_log`, `save_fact`,
`update_fact`. After a save/update, `render_result` builds the tool result —
the action line plus the full current store (so the agent sees the whole
updated memory, since the injected snapshot goes stale mid-session).

The daily-log FILE FORMAT (append shape, date-stamp rewrite, lookback)
lives in `common.daylog` — the conductor writes and reads the same
files and may not import this package — this module keeps the engine's
policy: the config-tuned window sizes and the tool-facing API.
"""

import re

from physiclaw.common import daylog, paths
from physiclaw.common.config import CONFIG
from physiclaw.common.text import append_text, read_text, write_text

MEMORY_DIR = paths.memory_dir()
MEMORY_FILE = paths.memory_file()
USER_FILE = MEMORY_DIR / "USER.md"

# `read_logs` default. Counts entry lines (one log step), not files —
# a low-activity recent day spills into older days until N is reached.
# User-tunable via `[memory] default_log_entries` in config.toml.
DEFAULT_LOG_ENTRIES = CONFIG.memory.default_log_entries

# How many log entries to preload into the memory slot at session
# bootstrap. Smaller than `DEFAULT_LOG_ENTRIES` because every wake pays
# this in prompt tokens whether the agent needs the history or not;
# the agent can always call `read_logs` for a deeper window. User-
# tunable via `[memory] bootstrap_log_entries` in config.toml.
BOOTSTRAP_LOG_ENTRIES = CONFIG.memory.bootstrap_log_entries


def load_user() -> str:
    """`memory/USER.md` body, or "" if missing/empty. Called by the
    doctrine renderer for the USER.md slot."""
    if not USER_FILE.exists():
        return ""
    return read_text(USER_FILE).strip()


def load_persistent() -> str:
    """`memory/memory.md` body, or "" if missing/empty. Auto-injected into
    the SYSTEM prompt under the engine-rendered `## memory.md` heading
    — no inner wrapper here. (Spec lives separately in the
    PERSISTENCE.md doctrine slot.)"""
    if not MEMORY_FILE.exists():
        return ""
    return read_text(MEMORY_FILE).strip()


def load_recent_entries(n: int = DEFAULT_LOG_ENTRIES) -> str:
    """Last N log entries across daily files, most recent first — the
    shared reader (`common.daylog`), with the engine's config-tuned
    default window and its import-frozen memory dir."""
    return daylog.load_recent_entries(n, root=MEMORY_DIR)


def append_log(entry: str) -> None:
    """Append a log line to today's daily file — the shared writer
    (`common.daylog`), against the engine's import-frozen memory dir."""
    daylog.append_log(entry, root=MEMORY_DIR)


def save_fact(text: str) -> None:
    """Append a persistent fact to memory.md."""
    text = text.strip()
    if not text:
        return
    MEMORY_DIR.mkdir(parents=True, exist_ok=True)
    append_text(MEMORY_FILE, text + "\n")


# Soft size budget for the always-injected semantic store (memory.md). Over
# this, the save handler nudges the agent to consolidate — soft, not enforced:
# unbounded growth turns memory.md into a token-heavy "junk drawer" that
# degrades recall, but hard auto-pruning risks dropping a rare essential fact,
# so the agent judges what to merge/keep (via `update_memory`).
SOFT_CAP_CHARS = CONFIG.memory.soft_cap_chars


def over_soft_cap() -> int | None:
    """The current memory.md length if it exceeds `SOFT_CAP_CHARS`, else None."""
    n = len(load_persistent())
    return n if n > SOFT_CAP_CHARS else None


def render_result(action: str) -> str:
    """A save/update tool result: the `action` line + the full CURRENT
    memory.md, so the agent always sees the whole updated store without a
    re-read (the injected `## memory.md` is a wake snapshot and goes stale after
    a mid-session write). Appends the soft-cap curation nudge when over budget."""
    store = load_persistent() or "(empty)"
    msg = f"{action}\n\n## memory.md (now)\n{store}"
    over = over_soft_cap()
    if over is not None:
        # memory.md rides in every prompt, so keep it lean.
        msg += (
            f"\n\n~{over} chars, over the {SOFT_CAP_CHARS} soft cap — "
            "use `update_memory` to merge or prune stale facts."
        )
    return msg


# ---------- memory cues (pre-close save enforcement) ----------

# "Please remember this" signals in the agent's own note / scratchpad / plan
# text. Scanned every turn (robust to compaction folding the turn away) and
# accumulated on the session; at close, an unaddressed cue forces one
# `save_memory` nudge. Latin alternatives are case-folded via IGNORECASE;
# the CJK ones match verbatim. Kept precision-leaning — a false positive
# only costs a single fail-open nudge the agent can decline.
_MEMORY_CUE_RE = re.compile(
    r"remember\s+(?:this|that|to|:|,)"
    r"|don'?t\s+forget|do\s+not\s+forget"
    r"|keep\s+in\s+mind"
    r"|for\s+(?:future|next)\s+(?:reference|time)"
    r"|save\s+(?:this\s+)?to\s+memory"
    r"|记住|记一下|记下来|记下|别忘|不要忘|务必记|以后记得|下次记得",
    re.IGNORECASE,
)


def scan_cues(text: str, *, window: int = 50, limit: int = 5) -> list[str]:
    """Snippets of `text` around any 'remember this' / '记住' cue — the match
    plus `window` chars either side, so the agent gets the cue and its
    context. Deduped (case-folded), capped at `limit`, order preserved."""
    out: list[str] = []
    seen: set[str] = set()
    for m in _MEMORY_CUE_RE.finditer(text or ""):
        lo = max(0, m.start() - window)
        hi = min(len(text), m.end() + window)
        snippet = " ".join(text[lo:hi].split())
        key = snippet.casefold()
        if not snippet or key in seen:
            continue
        seen.add(key)
        out.append(snippet)
        if len(out) >= limit:
            break
    return out


def cue_corrective(snippets: list[str]) -> str:
    """The pre-close corrective the engine injects when the session flagged
    something to remember but never called `save_memory`. Hands the model
    the matched cue(s) + context and names the exact fix."""
    joined = "\n".join(f'- "{s}"' for s in snippets)
    return (
        "Rejected — you flagged something to remember but haven't saved it. "
        "Matched cue(s):\n"
        f"{joined}\n"
        "If any is a durable fact/preference, `save_memory` it (one line); "
        "one-off detail goes to `append_log`. Then re-issue `end_session`."
    )


def update_fact(old: str, new: str) -> None:
    """Replace the single occurrence of `old` with `new` in memory.md.
    Empty `new` deletes the line containing `old`.

    Raises FileNotFoundError if memory.md is absent. Raises ValueError if
    `old` is not found OR appears more than once — the caller must pick a
    substring specific enough to match exactly one place.
    """
    if not MEMORY_FILE.exists():
        raise FileNotFoundError(f"{MEMORY_FILE} does not exist")
    text = read_text(MEMORY_FILE)
    count = text.count(old)
    if count == 0:
        raise ValueError(f"old text not found in memory.md: {old!r}")
    if count > 1:
        raise ValueError(
            f"old text matched {count} places in memory.md — "
            f"narrow the string so it matches exactly once"
        )
    if new:
        updated = text.replace(old, new, 1)
    else:
        if "\n" in old:
            # The line loop below can never match a substring spanning a
            # newline — without this guard the rewrite is a byte-identical
            # no-op that still reports "memory.md updated".
            raise ValueError(
                "old spans multiple lines — delete one line at a time "
                "(pass a single-line substring)"
            )
        lines = text.splitlines(keepends=True)
        out: list[str] = []
        removed = False
        for line in lines:
            if not removed and old in line:
                removed = True
                continue
            out.append(line)
        updated = "".join(out)
    write_text(MEMORY_FILE, updated)
