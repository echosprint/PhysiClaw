"""Session trajectory — the turn-tagged history of plan + scratchpad states.

The plan and scratchpad tails only ever hold the CURRENT value; a rewrite
overwrites the old one. But reflection (`reflect.py`) wants the *evolution* —
how the plan churned, which scratchpad approaches were tried and abandoned —
because that churn IS the turn-wasting-trap signal the agent should bank.

So the engine calls `record(session, turn, plan_updated=..., scratchpad_written=...)`
once per turn (after dispatch): the plan is snapshotted on EVERY `update_progress`
call and the scratchpad on EVERY `note(scratchpad=...)` write — each is an
explicit agent act (a re-plan, a re-write), even one that repeats the prior
content, and worth its own turn-tagged entry. A safety net also logs a drafted
plan / non-blank scratchpad that changed without the corresponding call (e.g.
the closing turn). At a hard close the pre-close gate injects `render(session)`
— plan re-plans and scratchpad writes merged into ONE `[t1]..[tN]` timeline
(oldest→newest, labelled by kind) — into the reflect corrective, so the model
reflects on the whole run as it unfolded.

In-memory only (like `Session`), never in `messages[]` and never on disk: it is
read exactly once, at reflect time. Both the retained-snapshot count and the
rendered char count are capped (`CONFIG.engine`), so a long session with a
full (8KB) scratchpad stays bounded in memory and in the prompt.
"""
from physiclaw.config import CONFIG

# Per-entry framing charged against the render budget (the `[tNN] <kind>\n`
# header line) so `_select` accounts for framing, not just the payload text.
_MARKER_OVERHEAD = 16


def record(
    session, turn: int, *, plan_updated: bool = False, scratchpad_written: bool = False
) -> None:
    """Snapshot the plan + scratchpad onto the session's turn-tagged logs.
    `turn` is 0-based; stored 1-based to match the human/agent turn count. Each
    log is capped to the newest `trajectory_max_snapshots` (evict oldest).

    Plan: logged on EVERY `update_progress` (`plan_updated`); scratchpad on
    EVERY `note(scratchpad=...)` write (`scratchpad_written`) — each is an
    explicit agent act worth capturing, even one that repeats the prior content.
    As a safety net, a drafted plan / non-blank scratchpad that changed WITHOUT
    the corresponding call (e.g. the closing turn) is logged once, deduped. A
    blank scratchpad is never logged."""
    cap = CONFIG.engine.trajectory_max_snapshots

    snap = session.plan.snapshot()
    changed = not session.plan_log or session.plan_log[-1][1] != snap
    if plan_updated or (changed and session.plan.is_drafted()):
        session.plan_log.append((turn + 1, snap))
        del session.plan_log[:-cap]

    sp = session.scratchpad
    changed_sp = not session.scratchpad_log or session.scratchpad_log[-1][1] != sp
    if sp.strip() and (scratchpad_written or changed_sp):
        session.scratchpad_log.append((turn + 1, sp))
        del session.scratchpad_log[:-cap]


def _select(entries: list, budget: int, max_count: int) -> tuple[list, int]:
    """Walk entries newest-first (pass ascending) and accumulate WHOLE entries
    until adding the next would exceed EITHER the char `budget` or `max_count`
    — then stop. Always keep at least the newest, and never trim an entry's
    text: the edge (oldest kept) entry is included in full or not at all. Text
    is the LAST tuple element. Returns (kept_newest_first, elided_count)."""
    kept: list = []
    used = 0
    for entry in reversed(entries):
        cost = len(entry[-1]) + _MARKER_OVERHEAD
        if kept and (used + cost > budget or len(kept) >= max_count):
            break
        kept.append(entry)
        used += cost
    return kept, len(entries) - len(kept)


def render(session, budget: int | None = None, max_count: int | None = None) -> str:
    """A turn-marked `<session-trajectory>` block for the reflect corrective, or
    "" when nothing was logged. Plan re-plans and scratchpad writes are merged
    into ONE timeline sorted `[t1]..[tN]` (oldest→newest), each entry labelled by
    kind, so the run reads as it unfolded — regardless of type. Selection applies
    BOTH caps (chars `budget`, count `max_count`) from the newest backward,
    keeping whole entries with an explicit elision note — never a silent or
    mid-entry truncation."""
    if budget is None:
        budget = CONFIG.engine.trajectory_budget
    if max_count is None:
        max_count = CONFIG.engine.trajectory_max_snapshots

    # (turn, kind, text); stable sort keeps plan before scratchpad on a shared
    # turn (plan_log built first) — reads "re-planned, then wrote notes".
    entries = (
        [(t, "plan", text) for t, text in session.plan_log]
        + [(t, "scratchpad", text) for t, text in session.scratchpad_log]
    )
    if not entries:
        return ""
    entries.sort(key=lambda e: e[0])

    kept, elided = _select(entries, budget, max_count)
    kept.reverse()  # back to ascending for display

    lines = ["<session-trajectory>", "(the run oldest→newest — plan re-plans + scratchpad writes)"]
    if elided:
        lines.append(f"({elided} earlier entr{'y' if elided == 1 else 'ies'} elided to fit)")
    for t, kind, text in kept:
        lines.append(f"[t{t}] {kind}\n{text}")

    lines.append("</session-trajectory>")
    return "\n".join(lines)
