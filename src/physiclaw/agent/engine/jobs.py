"""Engine-side job integration.

In-process replacement for the bash-CLI loopback `claude -p` used. This
module:
  - extracts cron-fired job ids from Triggers,
  - formats firings into the SYSTEM prompt,
  - creates new jobs when the model's response includes `create_job`.

Outcome marking is the agent's responsibility via `finish_job` — there
is no engine-side fallback. Jobs forgotten in `fired` status remain
there until the agent explicitly closes them.

Data model + jobs.md I/O lives in `physiclaw.agent.engine.job_store`. The hook
that actually ticks and fires is `physiclaw.agent.hooks.cron`.
"""

import datetime as dt
import logging

from physiclaw.agent.engine.job_store import (
    ID_RE,
    JOBS_PATH,
    KIND_ONE_TIME,
    KIND_PERIODIC,
    NEVER,
    STATUS_DONE,
    STATUS_FAIL,
    STATUS_PEND,
    TERMINAL_STATUSES,
    Job,
    format_minute,
    load_jobs,
    min_fire_gap,
    next_fire,
    update_fields,
    validate_schedule,
)
from physiclaw.agent.runtime.hook import Trigger

log = logging.getLogger(__name__)

_CRON_PREFIX = "cron:"

# Every periodic fire wakes the full agent (camera + LLM turns), and
# done/fail re-arms the job — a `*/5` "reply watcher" therefore loops
# forever at enormous cost once its purpose is gone. Below this floor,
# force a one-time follow-up instead.
PERIODIC_MIN_GAP = dt.timedelta(minutes=30)

AUTO_WAIT_JOB_ID = "wait-check-auto"
_AUTO_WAIT_DESCRIPTION = "Auto follow-up after WAIT with no explicit create_job."
_AUTO_WAIT_CONTEXT = (
    "Previous session ended with WAIT. Re-check IM / state and continue the task."
)


def fired_job_ids(triggers: list[Trigger]) -> list[str]:
    """Extract cron job ids from Trigger.source values like 'cron:a,b'."""
    ids: list[str] = []
    for t in triggers:
        if not t.source or not t.source.startswith(_CRON_PREFIX):
            continue
        ids.extend(x for x in t.source[len(_CRON_PREFIX) :].split(",") if x)
    return ids


def format_fired(triggers: list[Trigger]) -> str:
    """Build a SYSTEM-prompt section describing the cron jobs firing now."""
    ids = fired_job_ids(triggers)
    if not ids:
        return ""
    try:
        jobs = {j.id: j for j in load_jobs()}
    except Exception:
        log.exception("jobs: failed to load jobs.md")
        return ""

    blocks: list[str] = []
    for jid in ids:
        j = jobs.get(jid)
        if j is None:
            continue
        blocks.append(f"### {j.id}\n{j.description}\n\nContext: {j.context}")
    if not blocks:
        return ""
    return "## Scheduled jobs firing now\n\n" + "\n\n".join(blocks)


def create_job(
    *,
    id: str,
    description: str,
    schedule: str,
    context: str,
    kind: str = KIND_ONE_TIME,
) -> None:
    """Append a new job section to jobs.md. Pure append — never edits or
    overwrites an existing entry.

    Id format: `<user>-<topic>-<YYYY-MM-DD>` — see JOBS.md § Id format.

    Raises ValueError on duplicate id (even if the existing entry is
    terminal — the agent must pick a fresh id), invalid kind, invalid
    schedule, a periodic schedule firing more often than
    PERIODIC_MIN_GAP, missing description, or context shorter than
    10 chars.
    """
    if not ID_RE.match(id):
        raise ValueError(f"invalid job id {id!r} (lowercase + digits + hyphens)")
    if kind not in (KIND_ONE_TIME, KIND_PERIODIC):
        raise ValueError(f"kind must be one-time or periodic, got {kind!r}")
    validate_schedule(schedule)
    if kind == KIND_PERIODIC:
        gap = min_fire_gap(schedule, dt.datetime.now())
        if gap is not None and gap < PERIODIC_MIN_GAP:
            raise ValueError(
                f"periodic schedule {schedule!r} fires every "
                f"{int(gap.total_seconds() // 60)} min — the minimum for "
                f"periodic jobs is {int(PERIODIC_MIN_GAP.total_seconds() // 60)} "
                "min (each fire wakes the whole agent). For a reply check or "
                "follow-up, create a ONE-TIME job at a specific minute instead, "
                "and create another one then if it's still unresolved."
            )
    if len(context.strip()) < 10:
        raise ValueError("context must be at least 10 characters")
    description = description.strip()
    if not description:
        raise ValueError("description is required")

    if any(j.id == id for j in load_jobs()):
        raise ValueError(f"job id already exists: {id!r}")

    now = dt.datetime.now()
    nxt = next_fire(schedule, now)
    stamp_now = format_minute(now)
    stamp_next = format_minute(nxt) if nxt else NEVER

    block = (
        f"\n## {id}\n"
        f"{description}\n\n"
        f"- Type: {kind}\n"
        f"- Status: {STATUS_PEND}\n"
        f"- Schedule: `{schedule}`\n"
        f"- Context: {context.strip()}\n"
        f"- Create time: {stamp_now}\n"
        f"- Next fire time: {stamp_next}\n"
        f"- Last fire time: {NEVER}\n"
        f"- Execution time: {NEVER}\n"
        f"- Execution result: {NEVER}\n"
    )
    JOBS_PATH.parent.mkdir(parents=True, exist_ok=True)
    existed = JOBS_PATH.exists()
    with open(JOBS_PATH, "a", encoding="utf-8") as f:
        if not existed:
            f.write("# Jobs\n")
        f.write(block)
    log.info("jobs: created %s (schedule=%r, next=%s)", id, schedule, stamp_next)


def upsert_auto_wait_check(at: dt.datetime) -> None:
    """Singleton auto-follow-up after WAIT. Reuses one canonical id so
    jobs.md doesn't grow one entry per close; resets prior fire/exec
    history on reschedule so the row reads as a fresh pending job."""
    schedule = f"{at.minute} {at.hour} {at.day} {at.month} *"
    if AUTO_WAIT_JOB_ID in {j.id for j in load_jobs()}:
        update_fields(
            JOBS_PATH,
            {
                AUTO_WAIT_JOB_ID: {
                    "Schedule": f"`{schedule}`",
                    "Status": STATUS_PEND,
                    "Next fire time": format_minute(at),
                    "Last fire time": NEVER,
                    "Execution time": NEVER,
                    "Execution result": NEVER,
                }
            },
        )
        log.info(
            "jobs: rescheduled %s (schedule=%r, next=%s)",
            AUTO_WAIT_JOB_ID,
            schedule,
            format_minute(at),
        )
    else:
        create_job(
            id=AUTO_WAIT_JOB_ID,
            description=_AUTO_WAIT_DESCRIPTION,
            schedule=schedule,
            context=_AUTO_WAIT_CONTEXT,
        )


def job_ids() -> set[str] | None:
    """Ids currently in jobs.md, or None when the file won't parse (a
    missing file parses as no jobs). Two of these snapshots feed
    `created_since` — used by the claude path's WAIT-close detection,
    which has no in-process created-a-job signal. jobs.md does persist
    a `Create time` field, but it isn't surfaced on `Job`, and an
    id-set diff needs no clock at all."""
    try:
        return {j.id for j in load_jobs()}
    except Exception:
        log.warning("jobs.md unreadable for job-id snapshot", exc_info=True)
        return None


def created_since(before: set[str] | None) -> bool:
    """True when jobs.md gained a job since the `before` snapshot — e.g.
    a WAIT close's own resume job, which suppresses the outcome
    contract's generic follow-up. Unknown (either snapshot failed)
    counts as NOT created: a redundant auto-scheduled check-in beats a
    dead WAIT.

    Known coarseness: ANY new id counts, so an unrelated job created in
    the same wake (a reminder the user also asked for) suppresses the
    follow-up for a jobless WAIT. An id diff can't tell the two apart;
    accepted because the doctrine drills "pair a WAIT close with a
    resume job", making the created-but-unrelated + jobless-WAIT combo
    rare. Callers wanting tighter scope must snapshot closer to the
    close (the claude path snapshots per attempt for this reason)."""
    if before is None:
        return False
    after = job_ids()
    if after is None:
        return False
    return bool(after - before)


def get_job(id: str) -> Job:
    """Return the full Job for `id`. Raises ValueError on unknown id."""
    existing = {j.id: j for j in load_jobs()}
    if id not in existing:
        raise ValueError(f"no job with id {id!r}")
    return existing[id]


def finish_job(*, id: str, status: str, recap: str) -> str:
    """Mark a job as terminated. `status` is one of done / fail / cancel.

    Sets Status (or pend reset for periodic done/fail), Execution time
    (now), and Execution result (recap). Raises ValueError on unknown
    id, invalid status, or already-terminal status (re-finishing a
    closed job is a bug, not idempotent).

    For periodic jobs, done/fail reset to pend so the next firing still
    happens; cancel is permanent (to revive, create_job with a new id).
    Returns the tool-reply line; when a periodic job re-arms, the reply
    says so explicitly — agents read "done" as "over" and the job
    fires again minutes later.
    """
    if status not in TERMINAL_STATUSES:
        raise ValueError(
            f"status must be one of {sorted(TERMINAL_STATUSES)}, got {status!r}"
        )
    existing = {j.id: j for j in load_jobs()}
    if id not in existing:
        raise ValueError(f"no job with id {id!r}")
    j = existing[id]
    if j.status in TERMINAL_STATUSES:
        raise ValueError(f"job {id!r} is already in terminal status {j.status!r}")
    new_status = (
        STATUS_PEND
        if j.kind == KIND_PERIODIC and status in (STATUS_DONE, STATUS_FAIL)
        else status
    )
    update_fields(
        JOBS_PATH,
        {
            id: {
                "Status": new_status,
                "Execution time": format_minute(dt.datetime.now()),
                "Execution result": recap.strip() or status,
            }
        },
    )
    log.info("jobs: finished %s as %s", id, status)
    if new_status == STATUS_PEND:
        nxt = j.next_fire_time or "its next scheduled minute"
        return (
            f"finished job {id!r} as {status} — PERIODIC job, so it is "
            f"RE-ARMED and will fire again at {nxt}. If its purpose is "
            "over, call finish_job(status='cancel') now to stop it for good"
        )
    return f"finished job {id!r} as {status}"
