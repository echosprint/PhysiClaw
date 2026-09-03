"""Per-walk outcome log — ``playbooks/runs.jsonl``, the escalation KPI's data.

One machine-written JSONL line per walk, appended at the walk's FIRST
terminal moment (engine wakes and CLI rehearsals alike — recording lives
in `Program`, so callers can't diverge; the `session` field tells them
apart via the active-session marker, the `macros.runlog` convention)::

    {"ts", "session", "app", "playbook", "outcome", "node", "idx",
     "nodes", "reason", "micros", "rescues"}

``outcome`` is one of completed | suspended | handover | crashed |
abandoned. ``rescues`` counts the walk's recovery actions (the JSON key
keeps its original spelling).
``node``/``idx`` locate where the walk ended (``idx`` is the 0-based
cursor; ``node`` is null once past the last node); ``reason`` is the
handover reason or suspend recap, clipped; ``micros`` counts the
decision outcomes brokered to the walk. First terminal moment wins: a
suspension whose end_session is then blocked stays recorded "suspended"
— the one-line-per-walk shape beats a second row.

Completed lines additionally carry the walk's `values` (the resolved
inputs, clipped) and `total` (the ¥ amount actually paid, when a
payment fired). TELEMETRY ONLY: the human-readable record of a walk's
purchases and suspensions goes to the agent's own daily log
(`common.daylog`, written by the Program at those moments), which is
what parse_task's activation context reads — the conductor follows the
agent's memory convention, never a private prompt store.

Read by `physiclaw playbooks stats` / `propose` — never into the
SYSTEM prompt. Never fatal in either direction: a write
failure logs and moves on, unparseable lines are skipped at read.
Append-only with no cap (alpha convention — readers take the tail).
"""

import json
import logging
from collections import Counter
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

from physiclaw.common import paths
from physiclaw.common.logger import iso_now
from physiclaw.common.text import append_text, clip, read_text

log = logging.getLogger(__name__)

RUNS_FILENAME = "runs.jsonl"


# The valid outcome spellings — `record` refuses anything else loudly (a
# typo'd outcome at a call site must fail tests, never skew the KPI).
# "abandoned" = the SESSION ended around a mid-flight walk (Ctrl-C,
# wall-clock budget) — recorded from plugin teardown (`Program.abandon`,
# `log_external_stop`'s twin), and deliberately NOT an escalation: the
# next wake starts the route over, no model session was spent.
class Outcome(StrEnum):
    """How a walk ended — the one vocabulary the walk records, the KPI
    counts, and a replay or a stepping driver reports."""

    COMPLETED = "completed"
    SUSPENDED = "suspended"
    HANDOVER = "handover"
    CRASHED = "crashed"
    ABANDONED = "abandoned"


# The KPI's numerator, one home: suspensions are the walk working as
# designed and abandonments spent no model session — neither escalates.
ESCALATION_OUTCOMES = frozenset({Outcome.HANDOVER, Outcome.CRASHED})

_TRUNCATE_REASON = 200


def runs_file() -> Path:
    return paths.playbooks_dir() / RUNS_FILENAME


def record(
    *,
    app: str,
    playbook: str,
    outcome: Outcome,
    idx: int,
    nodes: int,
    node: str | None,
    reason: str = "",
    micros: int = 0,
    rescues: int = 0,
    values: dict | None = None,
    total: float | None = None,
) -> None:
    """Append one walk's terminal line. Best-effort: a write failure logs
    and moves on — telemetry must never take down a session. `outcome`
    is coerced through `Outcome`, so a typo'd spelling at a call site
    raises rather than skewing the KPI."""
    try:
        outcome = Outcome(outcome)
    except ValueError:
        raise ValueError(f"unknown walk outcome {outcome!r}") from None
    line = {
        "ts": iso_now(),
        "session": paths.live_session_id(),
        "app": app,
        "playbook": playbook,
        "outcome": outcome,
        "node": node,
        "idx": idx,
        "nodes": nodes,
        "reason": _clip(reason),
        "micros": micros,
        "rescues": rescues,
        "values": {str(k): _clip(str(v)) for k, v in (values or {}).items()},
        "total": total,
    }
    try:
        paths.playbooks_dir().mkdir(parents=True, exist_ok=True)
        append_text(runs_file(), json.dumps(line, ensure_ascii=False) + "\n")
    except Exception:
        log.warning("walk run log write failed", exc_info=True)


def load() -> list[dict]:
    """Every recorded run, oldest first. Missing file → []; a line that
    won't parse is skipped (fail-open read, like macro stats)."""
    p = runs_file()
    if not p.exists():
        return []
    out: list[dict] = []
    for line in read_text(p).splitlines():
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except ValueError:
            log.warning("skipping unparseable runs.jsonl line")
            continue
        if isinstance(rec, dict):
            out.append(rec)
    return out


@dataclass
class WalkStats:
    """One playbook's aggregate — what `playbooks stats` renders."""

    runs: int = 0
    completed: int = 0
    suspended: int = 0
    handover: int = 0
    crashed: int = 0
    abandoned: int = 0
    micros: int = 0
    rescues: int = 0
    # Handover counts per node id ("?" for a row with none), plus the
    # latest reason seen there — the example the stats line quotes.
    handover_nodes: Counter = field(default_factory=Counter)
    last_reason: dict = field(default_factory=dict)

    @property
    def escalation_rate(self) -> float:
        """`ESCALATION_OUTCOMES` over all runs — the KPI."""
        if not self.runs:
            return 0.0
        return sum(getattr(self, o) for o in ESCALATION_OUTCOMES) / self.runs

    def hot_nodes(self, top: int = 3) -> list[tuple[str, int, str]]:
        """The most-escalated nodes: (node, count, latest reason)."""
        return [
            (node, count, str(self.last_reason.get(node, "")))
            for node, count in self.handover_nodes.most_common(top)
        ]


def summarize(rows: list[dict]) -> dict[str, WalkStats]:
    """Fold run rows into per-playbook stats, keyed ``app/playbook``.
    Unknown outcome spellings (a future field, a hand-edit) count into
    `runs` only — the KPI never silently reclassifies them."""
    out: dict[str, WalkStats] = {}
    for rec in rows:
        key = f"{rec.get('app', '?')}/{rec.get('playbook', '?')}"
        st = out.setdefault(key, WalkStats())
        st.runs += 1
        st.micros += int(rec.get("micros") or 0)
        st.rescues += int(rec.get("rescues") or 0)
        outcome = rec.get("outcome")
        if outcome == "completed":
            st.completed += 1
        elif outcome == "suspended":
            st.suspended += 1
        elif outcome == "handover":
            st.handover += 1
        elif outcome == "crashed":
            st.crashed += 1
        elif outcome == "abandoned":
            st.abandoned += 1
        if outcome in ESCALATION_OUTCOMES:
            node = str(rec.get("node") or "?")
            st.handover_nodes[node] += 1
            st.last_reason[node] = str(rec.get("reason") or "")
    return out


@dataclass(frozen=True)
class EscalationSite:
    """One hot escalation site — `playbooks propose`'s triage unit."""

    app: str
    playbook: str
    node: str
    count: int
    reason: str  # the latest one seen there
    sessions: tuple[str, ...]  # recent session ids to mine, oldest first


def escalation_sites(rows: list[dict], top: int = 3) -> list[EscalationSite]:
    """The hottest escalation sites across all playbooks, most-escalated
    first — every one is authoring work the author can now see (a page
    that keeps missing, a popup no hand clears, a macro whose
    rehearsal drifted)."""
    counts: Counter = Counter()
    latest: dict = {}
    sessions: dict = {}
    for r in rows:
        if r.get("outcome") not in ESCALATION_OUTCOMES:
            continue
        key = (
            str(r.get("app") or "?"),
            str(r.get("playbook") or "?"),
            str(r.get("node") or "?"),
        )
        counts[key] += 1
        latest[key] = str(r.get("reason") or "")
        if r.get("session"):
            sessions.setdefault(key, []).append(str(r["session"]))
    out: list[EscalationSite] = []
    for key, count in counts.most_common(top):
        app, playbook, node = key
        out.append(
            EscalationSite(
                app=app,
                playbook=playbook,
                node=node,
                count=count,
                reason=latest[key],
                sessions=tuple(sessions.get(key, [])[-3:]),
            )
        )
    return out


def _clip(text: str) -> str:
    return clip(text, _TRUNCATE_REASON)
