"""jobs.md — the durable job model.

Shared between `physiclaw.agent.hooks.cron` (fires Triggers when jobs are due) and
`physiclaw.agent.engine.jobs` (formats firings into the SYSTEM prompt, persists
outcomes). This module holds:

- the `Job` dataclass + field constants
- `load_jobs` / `update_fields` — read and mutate jobs.md in place
- `validate_schedule` / `matches_now` / `next_fire` / `find_due` — cron helpers
- `purge_stale` — housekeeping for terminal jobs older than 7 days

See `.claude/skills/cron/SKILL.md` for the full format spec and workflows.

Schedule syntax: standard 5-field cron (min hour dom mon dow). Supports `*`,
`N`, `N,M`, `*/N`, `A-B`. Day-of-week: 0=Sun, 6=Sat. All times are naive
local time, ISO 8601 truncated to minute.
"""

import datetime as dt
import logging
import re
from dataclasses import dataclass
from pathlib import Path

from croniter import croniter

from physiclaw.common import paths
from physiclaw.common.config import CONFIG
from physiclaw.common.text import read_text, write_text

log = logging.getLogger(__name__)

JOBS_PATH = paths.jobs_file()

KIND_PERIODIC = "periodic"
KIND_ONE_TIME = "one-time"
_VALID_KINDS = {KIND_PERIODIC, KIND_ONE_TIME}

ID_RE = re.compile(r"^[a-z0-9][a-z0-9-]*$")
_HEADING_RE = re.compile(r"^##\s+(.+?)\s*$", re.MULTILINE)
# Fields are markdown list items: `- Key: value`. The leading dash is
# required so a description line that happens to contain a colon is
# never confused with a field.
_FIELD_RE = re.compile(r"^\s*-\s+([A-Z][A-Za-z ]*?)\s*:\s*(.*)$")
_KNOWN_FIELDS = {
    "Type",
    "Status",
    "Schedule",
    "Context",
    "Create time",
    "Next fire time",
    "Last fire time",
    "Execution time",
    "Execution result",
}
NEVER = "(never)"
_NEVER_VALUES = {"", NEVER, "never", "-"}

STATUS_PEND = "pend"
STATUS_FIRED = "fired"
STATUS_CANCEL = "cancel"
STATUS_DONE = "done"
STATUS_FAIL = "fail"
_VALID_STATUS = {STATUS_PEND, STATUS_FIRED, STATUS_CANCEL, STATUS_DONE, STATUS_FAIL}

_PURGE_AFTER = dt.timedelta(days=CONFIG.retention.trace_days)
TERMINAL_STATUSES = frozenset({STATUS_CANCEL, STATUS_DONE, STATUS_FAIL})


@dataclass(frozen=True)
class Job:
    id: str
    kind: str  # "periodic" or "one-time"
    schedule: str
    description: str
    status: str = STATUS_PEND  # "pend", "fired", "cancel", "done", or "fail"
    context: str = ""
    next_fire_time: str = ""  # ISO minute or ""
    last_fire_time: str = ""  # ISO minute, or ""
    execution_time: str = ""  # ISO minute, or ""
    execution_result: str = ""  # description of execution outcome


# ---------- parser ----------


def load_jobs(path: Path | None = None) -> list[Job]:
    """Parse jobs from `jobs.md`.

    Raises ValueError on malformed job sections (bad id, missing required
    fields, invalid kind/schedule). Documentation-style `## ...` sections
    whose headings aren't valid ids are skipped silently. A missing file
    returns an empty list.
    """
    if path is None:
        path = JOBS_PATH  # read at call time, not def time
    if not path.exists():
        return []
    text = read_text(path)

    parts = _HEADING_RE.split(text)
    jobs: list[Job] = []
    seen_ids: set[str] = set()

    for i in range(1, len(parts), 2):
        heading = parts[i].strip()
        body = parts[i + 1] if i + 1 < len(parts) else ""
        if not ID_RE.match(heading):
            continue  # documentation section
        description, fields = _parse_section(heading, body)
        if not fields:
            continue  # not a job (e.g. docs section with no fields)
        if heading in seen_ids:
            raise ValueError(f"duplicate job id: {heading!r}")

        if not description:
            raise ValueError(f"{heading}: missing description line below the heading")

        kind = fields["Type"].strip().lower()
        if kind not in _VALID_KINDS:
            raise ValueError(
                f"{heading}: Type must be 'periodic' or 'one-time', got {kind!r}"
            )

        status = fields["Status"].strip().lower()
        if status not in _VALID_STATUS:
            raise ValueError(
                f"{heading}: Status must be one of "
                f"{sorted(_VALID_STATUS)}, got {status!r}"
            )

        schedule = fields["Schedule"].strip("`").strip()
        validate_schedule(schedule)

        context = fields["Context"].strip()
        if len(context) < 10:
            raise ValueError(f"{heading}: Context too short (min 10 chars)")

        last_fire_time = fields["Last fire time"].strip()
        last_fire_time = "" if last_fire_time in _NEVER_VALUES else last_fire_time

        execution_time = fields["Execution time"].strip()
        execution_time = "" if execution_time in _NEVER_VALUES else execution_time

        execution_result = fields["Execution result"].strip()
        execution_result = "" if execution_result in _NEVER_VALUES else execution_result

        next_raw = fields["Next fire time"].strip()
        next_fire_time = "" if next_raw in _NEVER_VALUES else next_raw

        # Validate Next fire time based on status.
        if status == STATUS_PEND:
            if not next_fire_time:
                raise ValueError(f"{heading}: Next fire time is required for pend jobs")
            try:
                nft = dt.datetime.fromisoformat(next_fire_time)
            except (ValueError, TypeError) as e:
                raise ValueError(
                    f"{heading}: Next fire time {next_raw!r} is not a "
                    f"valid ISO timestamp"
                ) from e
            if not matches_now(schedule, nft):
                raise ValueError(
                    f"{heading}: Next fire time {next_fire_time} does "
                    f"not match Schedule {schedule!r}"
                )
        elif next_fire_time:
            try:
                dt.datetime.fromisoformat(next_fire_time)
            except (ValueError, TypeError) as e:
                raise ValueError(
                    f"{heading}: Next fire time {next_raw!r} is not a valid "
                    f"ISO timestamp"
                ) from e

        seen_ids.add(heading)
        jobs.append(
            Job(
                id=heading,
                kind=kind,
                schedule=schedule,
                description=description,
                status=status,
                context=context,
                next_fire_time=next_fire_time,
                last_fire_time=last_fire_time,
                execution_time=execution_time,
                execution_result=execution_result,
            )
        )
    return jobs


_REQUIRED_FIELDS = {
    "Type",
    "Schedule",
    "Context",
    "Create time",
    "Next fire time",
    "Last fire time",
    "Execution time",
    "Execution result",
    "Status",
}


def _parse_section(heading: str, body: str) -> tuple[str, dict[str, str]]:
    """Parse a job section: description line + field list.

    Sections with zero recognized fields are treated as documentation and
    returned as empty — load_jobs skips them. Sections with at least one
    field are treated as jobs: unexpected lines and missing required
    fields both raise ValueError.
    """
    lines = [line for line in body.splitlines() if line.strip()]
    if not lines:
        return "", {}

    description = ""
    start = 0
    first = lines[0].strip()
    m = _FIELD_RE.match(first)
    if not m or m.group(1).strip() not in _KNOWN_FIELDS:
        description = first
        start = 1

    fields: dict[str, str] = {}
    unexpected: list[str] = []
    for line in lines[start:]:
        m = _FIELD_RE.match(line.rstrip())
        if m and m.group(1).strip() in _KNOWN_FIELDS:
            fields.setdefault(m.group(1).strip(), m.group(2).strip())
        else:
            unexpected.append(line.strip())

    if not fields:
        return description, {}  # docs section, not a job

    if unexpected:
        raise ValueError(f"{heading}: unexpected line: {unexpected[0]!r}")

    missing = _REQUIRED_FIELDS - fields.keys()
    if missing:
        raise ValueError(
            f"{heading}: missing required field(s): {', '.join(sorted(missing))}"
        )

    return description, fields


# ---------- cron helpers ----------


def validate_schedule(schedule: str) -> None:
    if not croniter.is_valid(schedule):
        raise ValueError(f"invalid cron expression: {schedule!r}")


def matches_now(schedule: str, now: dt.datetime) -> bool:
    """True if `schedule` matches `now` at minute granularity."""
    return croniter.match(schedule, now.replace(second=0, microsecond=0))


def next_fire(schedule: str, after: dt.datetime) -> dt.datetime | None:
    """Next datetime after `after` that matches `schedule`."""
    return croniter(schedule, after).get_next(dt.datetime)


def format_minute(t: dt.datetime) -> str:
    return t.replace(second=0, microsecond=0).isoformat(timespec="minutes")


def min_fire_gap(
    schedule: str, start: dt.datetime, samples: int = 5
) -> dt.timedelta | None:
    """Smallest gap between consecutive fires over the next `samples` fires.

    Sampling (rather than parsing the cron fields) handles mixed
    schedules like `0,10 9 * * *` whose gaps vary."""
    prev = next_fire(schedule, start)
    if prev is None:
        return None
    smallest: dt.timedelta | None = None
    for _ in range(samples):
        nxt = next_fire(schedule, prev)
        if nxt is None:
            break
        gap = nxt - prev
        if smallest is None or gap < smallest:
            smallest = gap
        prev = nxt
    return smallest


def _parse_minute(ts_str: str) -> dt.datetime | None:
    """ISO-minute timestamp → datetime, or None on empty / garbage."""
    if not ts_str:
        return None
    try:
        return dt.datetime.fromisoformat(ts_str)
    except ValueError:
        return None


# ---------- due check ----------


def find_due(jobs: list[Job], now: dt.datetime) -> list[Job]:
    """Return jobs whose Next fire time has arrived.

    Only `Status: pend` jobs with a valid Next fire time are eligible. Uses
    the precomputed Next fire time rather than matching the schedule against
    `now`, so delayed ticks don't miss the job.
    """
    due: list[Job] = []
    for job in jobs:
        if job.status != STATUS_PEND:
            continue
        nft = _parse_minute(job.next_fire_time)
        if nft is None:
            continue
        if nft <= now:
            due.append(job)
    return due


# ---------- value sanitation ----------


def one_line(value: str) -> str:
    """Fold free text onto a single line — the format's core invariant.

    jobs.md is line-oriented: one heading, one description line, one
    `- Field: value` line each. A newline inside a written value becomes
    a stray line the parser rejects, and one malformed section kills
    `load_jobs()` for EVERY job — the cron scheduler stays dead until a
    human repairs the file. Model-supplied strings (context, recap,
    description) routinely carry newlines, so writers fold instead of
    reject: intent survives, the file never breaks.

    Uses `str.splitlines()` — the exact line-boundary set (\\n, \\r,
    \\x85, \\u2028, …) the parser's own `splitlines()` splits on, so the
    writer and parser can never disagree on what "one line" means.
    """
    return " ".join(part.strip() for part in value.splitlines() if part.strip())


def parses_as_field(line: str) -> bool:
    """True when `line` would be consumed as a known `- Field: value`
    row by the parser. A description with this shape doesn't corrupt the
    file — worse, it silently becomes the field (``setdefault`` keeps
    the first occurrence), poisoning e.g. Status for the whole section.
    `create_job` rejects such descriptions instead of rewriting them."""
    m = _FIELD_RE.match(line)
    return bool(m and m.group(1).strip() in _KNOWN_FIELDS)


# ---------- in-place field updates ----------


def _update_field(text: str, job_id: str, field_name: str, value: str) -> str:
    """Replace a single `- Field name:` list item in a job section.

    Raises ValueError if the field line doesn't exist — all fields are
    required and must be present in the file. `value` is folded to one
    line (see `one_line`) — this is the single choke point every update
    path (finish_job recap, cron stamps, zombie expiry) writes through.
    """
    value = one_line(value)
    # `[^\n]*` for the trailing line — `.*` would be greedy across
    # newlines under DOTALL and eat the rest of the file.
    pattern = re.compile(
        rf"(^##\s+{re.escape(job_id)}\s*$\n(?:(?!^##\s).)*?)^(\s*-\s+){re.escape(field_name)}:[^\n]*",
        re.MULTILINE | re.DOTALL,
    )
    new_text, count = pattern.subn(
        lambda m: m.group(1) + f"{m.group(2)}{field_name}: {value}",
        text,
        count=1,
    )
    if count == 0:
        raise ValueError(f"{job_id}: field '- {field_name}:' not found in jobs.md")
    return new_text


def update_fields(path: Path, updates: dict[str, dict[str, str]]) -> None:
    """Apply `{job_id: {field: value, ...}}` updates to `path` in place.

    Preserves the rest of the file byte-for-byte where possible.
    """
    if not updates:
        return
    text = read_text(path)
    for job_id, fields in updates.items():
        for field_name, value in fields.items():
            text = _update_field(text, job_id, field_name, value)
    write_text(path, text)


# ---------- auto-purge stale jobs ----------


def _latest_timestamp(job: Job) -> dt.datetime | None:
    """Most recent activity timestamp for a job, or None."""
    for ts_str in (job.execution_time, job.last_fire_time):
        ts = _parse_minute(ts_str)
        if ts is not None:
            return ts
    return None


def _remove_sections(path: Path, job_ids: set[str]) -> None:
    """Delete entire `## <id>` sections from the file."""
    if not job_ids:
        return
    text = read_text(path)
    for job_id in job_ids:
        pattern = re.compile(
            rf"^##\s+{re.escape(job_id)}\s*$\n(?:(?!^##\s)[\s\S])*?(?=^##\s|\Z)",
            re.MULTILINE,
        )
        text = pattern.sub("", text)
    # Clean up any resulting triple+ blank lines.
    text = re.sub(r"\n{3,}", "\n\n", text)
    write_text(path, text)


# A fired job no session ever closed sits outside both find_due (not
# pend) and purge_stale (not terminal) — a permanent zombie. After this
# long it is auto-closed so the miss surfaces in the file: one-time →
# fail (terminal, purges later); periodic → pend, mirroring
# finish_job's fail-on-periodic semantics — terminal would silently
# kill a living routine over one unhandled fire.
FIRED_EXPIRE = dt.timedelta(hours=CONFIG.retention.fired_expire_hours)


def expire_stale_fired(
    *,
    now: dt.datetime,
    jobs: list[Job] | None = None,
    path: Path | None = None,
) -> list[str]:
    """Auto-close jobs stuck in `fired` for FIRED_EXPIRE or longer.

    One-time zombies are failed; periodic zombies re-arm to `pend` (the
    stale Next fire time makes them fire again promptly — worst case one
    wake per FIRED_EXPIRE until a session closes them properly). Returns
    the expired ids. Jobs with a missing/unparsable Last fire time are
    left alone (they surface via `cron verify`)."""
    if path is None:
        path = JOBS_PATH
    if jobs is None:
        jobs = load_jobs(path)

    updates: dict[str, dict[str, str]] = {}
    for j in jobs:
        if j.status != STATUS_FIRED:
            continue
        fired_at = _parse_minute(j.last_fire_time)
        if fired_at is None:
            continue
        if now - fired_at >= FIRED_EXPIRE:
            if j.kind == KIND_PERIODIC:
                new_status, verdict = STATUS_PEND, "re-armed"
            else:
                new_status, verdict = STATUS_FAIL, "failed"
            updates[j.id] = {
                "Status": new_status,
                "Execution time": format_minute(now),
                "Execution result": (
                    f"auto-{verdict}: fired {j.last_fire_time} but never finished"
                ),
            }

    if updates:
        update_fields(path, updates)
        log.warning(
            "jobs: auto-closed %d zombie fired job(s): %s",
            len(updates),
            sorted(updates),
        )
    return sorted(updates)


def purge_stale(
    path: Path | None = None,
    now: dt.datetime | None = None,
    jobs: list[Job] | None = None,
) -> list[str]:
    """Remove jobs in terminal status (cancel/done/fail) inactive for 7+
    days (or with no parseable timestamp). Returns purged job ids.

    Pass `jobs` to reuse an already-loaded list (the cron hook does this to
    avoid re-parsing jobs.md every tick).
    """
    if path is None:
        path = JOBS_PATH
    if now is None:
        now = dt.datetime.now()
    if jobs is None:
        if not path.exists():
            return []
        try:
            jobs = load_jobs(path)
        except Exception:
            return []

    to_remove: set[str] = set()
    for j in jobs:
        if j.status not in TERMINAL_STATUSES:
            continue
        ts = _latest_timestamp(j)
        if ts is None or now - ts >= _PURGE_AFTER:
            to_remove.add(j.id)

    if to_remove:
        _remove_sections(path, to_remove)
        log.info("jobs: purged %d stale job(s): %s", len(to_remove), sorted(to_remove))
    return sorted(to_remove)
