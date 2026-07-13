"""Learned pitfalls — a single, always-on list of turn-wasting traps.

An agent that authors a whole *working flow* self-congratulates (it thinks the
flow worked even when it didn't), so such notes rot. Pitfalls don't have that
failure mode — "X wasted turns → avoid it" is grounded in real friction, not
self-praise. So the agent's *only* power here is to **append ≤3 traps** it hit;
it never writes a how-to, and never edits or prunes.

Shape: one flat file `learned/pitfalls/pitfalls.md`, one `- item` per line,
**newest on top**. Always injected into SYSTEM right after user skills
(`render_section`), so there is no `Skill()` load to forget. Three bounds keep
it prompt-lean: ≤3 per session (the tool), ≤`max_item_chars` per line (clamped
here), ≤`max_items` total (the curator consolidates toward it; `add`/`replace`
hard-cut from the bottom/oldest as a backstop). Every write appends a full
snapshot to `history.jsonl`, so a curator prune is always recoverable.

This module owns the write (`add`, `replace`), read (`read`, `render_section`),
and the pre-close gate (`should_capture`, `corrective`) that forces an
`add_pitfall(...)` before a long DONE. The curator (`curate.py`) is the only
caller of `replace`; `policy.PitfallCheckpoint` is the only caller of the gate.
"""

import datetime as dt
import json
import re
from typing import TYPE_CHECKING

from physiclaw.agent.runtime.sentinel import DONE
from physiclaw.common import paths
from physiclaw.common.config import CONFIG
from physiclaw.common.text import append_text, read_text, write_text

if TYPE_CHECKING:
    from physiclaw.agent.engine.session import Session

_FILE = "pitfalls.md"
_HISTORY = "history.jsonl"
_HEADER = "## Learned pitfalls"
_DOCTRINE = (
    "Traps past runs hit — don't repeat a listed one; ignore ones for other apps."
)
_BULLET_RE = re.compile(r"^\s*-\s+(.*\S)\s*$")


def _file():
    return paths.pitfalls_dir() / _FILE


def _norm(text: str) -> str:
    """Dedup key: whitespace-collapsed, case-folded."""
    return " ".join(text.split()).casefold()


def _clean(item: str) -> str:
    """One-line, whitespace-collapsed, clamped to `max_item_chars`. "" if blank."""
    s = " ".join((item or "").split())
    cap = CONFIG.pitfalls.max_item_chars
    if len(s) > cap:
        s = s[: cap - 1].rstrip() + "…"
    return s


def _dedup(items: list[str]) -> list[str]:
    """Drop blanks and normalized-duplicates, first wins, order preserved."""
    out: list[str] = []
    seen: set[str] = set()
    for raw in items:
        s = _clean(raw)
        if not s or _norm(s) in seen:
            continue
        seen.add(_norm(s))
        out.append(s)
    return out


def read() -> list[str]:
    """The current list, newest first (file order). [] if none."""
    path = _file()
    if not path.exists():
        return []
    out: list[str] = []
    for line in read_text(path).splitlines():
        m = _BULLET_RE.match(line)
        if m:
            out.append(m.group(1))
    return out


def _append_history(op: str, items: list[str]) -> None:
    path = paths.pitfalls_dir() / _HISTORY
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "ts": dt.datetime.now().isoformat(timespec="seconds"),
        "op": op,
        "items": items,
    }
    append_text(path, json.dumps(record, ensure_ascii=False) + "\n")


def _commit(op: str, items: list[str]) -> None:
    """Persist `items` (already deduped + capped) as the new list + snapshot."""
    path = _file()
    path.parent.mkdir(parents=True, exist_ok=True)
    _append_history(op, items)
    write_text(path, "".join(f"- {x}\n" for x in items))


def add(items: list[str]) -> dict:
    """Prepend up to 3 new pitfalls (newest on top), deduped against the batch
    and the existing list, each clamped. Hard-cut to `max_items` from the
    bottom (oldest) as a backstop — the curator does the real consolidation.
    Append-only for the agent: it can never edit or drop an existing item.
    Returns {added, total}."""
    fresh = _dedup(items or [])[:3]
    existing = read()
    existing_keys = {_norm(x) for x in existing}
    fresh = [x for x in fresh if _norm(x) not in existing_keys]
    if not fresh:
        return {"added": 0, "total": len(existing)}
    merged = (fresh + existing)[: CONFIG.pitfalls.max_items]
    _commit("add", merged)
    return {"added": len(fresh), "total": len(merged)}


def replace(items: list[str]) -> dict:
    """Curator write path: replace the whole list with `items`, deduped and
    hard-cut to the top `max_items` (drop oldest). No-op guard: an empty result
    never wipes a non-empty list. Returns {total, dropped}."""
    cleaned = _dedup(items or [])
    if not cleaned:
        return {"total": len(read()), "dropped": 0}
    dropped = max(0, len(cleaned) - CONFIG.pitfalls.max_items)
    kept = cleaned[: CONFIG.pitfalls.max_items]
    _commit("curate", kept)
    return {"total": len(kept), "dropped": dropped}


def render_section(items: list[str] | None = None) -> str:
    """The `## Learned pitfalls` block for SYSTEM (after user skills), or "" when
    the list is empty. Built once at bootstrap; pass an already-read list to
    avoid re-reading the file (the caller also needs the count)."""
    if items is None:
        items = read()
    if not items:
        return ""
    return "\n".join([_HEADER, "", _DOCTRINE, "", *(f"- {x}" for x in items)])


# --- Pre-close gate: force `add_pitfall(...)` before a long DONE ------------
#
# A pitfall is `trap → fix`, and you can only write the fix if you actually got
# PAST the trap — so we capture only on a long DONE (over `capture_turn_floor`
# turns): the task succeeded, and a run that long almost always stumbled over
# something worth banking. The agent judges what (it may add 0). A STUCK / FAIL
# session never escaped the trap (the "fix" would be a guess) and a short DONE
# sailed through — neither captures. `policy.PitfallCheckpoint` intercepts the
# closing `end_session`, rejects it once, and injects `corrective(seed)`.


def should_capture(status: str, turn: int, session: "Session") -> tuple[bool, str]:
    """Whether to force `add_pitfall(...)` before this close, plus a diagnostic
    seed. Capture only on a DONE that ran past `capture_turn_floor` turns — the
    task succeeded and the length says it hit friction; the agent decides what
    (0–3 traps). STUCK / FAIL never escaped the trap; a short DONE sailed
    through; IDLE / WAIT wakes never do."""
    if not CONFIG.pitfalls.capture_enabled:
        return False, ""
    if status != DONE or turn < CONFIG.pitfalls.capture_turn_floor:
        return False, ""
    seed = f"{turn} turns"
    if session.stuck_events:
        seed += f", loop-guard fired {session.stuck_events}×"
    seed += ", closing DONE"
    return True, seed


def corrective(seed: str, trajectory: str = "") -> str:
    """The pre-close corrective injected on a rejected long-DONE `end_session`.
    Names the exact re-issue shape, the measured-waste seed, and — when present
    — the turn-marked plan/scratchpad trajectory to mine for the real traps."""
    msg = (
        "Rejected — before closing, `add_pitfall(...)` with the traps that "
        f"wasted turns this session (up to 3; session: {seed}). Each: lead with "
        "the app, then `trap → the fix that worked`, terse. If nothing was worth "
        "banking, pass an empty list. Then re-issue `end_session`."
    )
    if trajectory:
        msg += (
            "\n\nMine your run for the turn-wasters — where the plan churned or "
            "the same action kept failing:"
            f"\n{trajectory}"
        )
    return msg
