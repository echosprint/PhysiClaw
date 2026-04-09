"""Cron hook — fires Triggers for scheduled jobs in `jobs/jobs.md`.

Each tick: parse jobs.md → find jobs whose status is `pend` and
Next fire time has arrived → fire a Trigger → update Last/Next fire time
in place. The agent marks completion via `done <id>` or `fail <id>` CLI.
Terminal jobs (cancel/done/fail) are auto-purged after 7 days.

See `.claude/skills/cron/SKILL.md` for the full format spec, field
ownership table, and workflows.

Schedule syntax: standard 5-field cron (min hour dom mon dow).
Supports `*`, `N`, `N,M`, `*/N`, `A-B`. Day-of-week: 0=Sun, 6=Sat.
All times are naive local time, ISO 8601 truncated to minute.
"""

import datetime as dt
import logging
import re
from dataclasses import dataclass
from pathlib import Path

from croniter import croniter

from physiclaw.runtime.hook import Trigger, register

log = logging.getLogger(__name__)

JOBS_PATH = Path("jobs/jobs.md")

KIND_PERIODIC = "periodic"
KIND_ONE_TIME = "one-time"
_VALID_KINDS = {KIND_PERIODIC, KIND_ONE_TIME}

_ID_RE = re.compile(r"^[a-z0-9][a-z0-9-]*$")
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
_NEVER_VALUES = {"", "(never)", "never", "-"}

STATUS_PEND = "pend"
STATUS_FIRED = "fired"
STATUS_CANCEL = "cancel"
STATUS_DONE = "done"
STATUS_FAIL = "fail"
_VALID_STATUS = {STATUS_PEND, STATUS_FIRED, STATUS_CANCEL, STATUS_DONE, STATUS_FAIL}

# Jobs in a terminal status (cancel/done/fail) are auto-deleted from
# jobs.md after this many days since their last activity timestamp.
_PURGE_AFTER = dt.timedelta(days=7)
_TERMINAL_STATUSES = {STATUS_CANCEL, STATUS_DONE, STATUS_FAIL}


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

    Raises ValueError on malformed job sections (bad id, missing
    required fields, invalid kind/schedule). Documentation-style
    `## ...` sections whose headings aren't valid ids are skipped
    silently. A missing file returns an empty list.
    """
    if path is None:
        path = JOBS_PATH  # read at call time, not def time
    if not path.exists():
        return []
    text = path.read_text()

    parts = _HEADING_RE.split(text)
    jobs: list[Job] = []
    seen_ids: set[str] = set()

    for i in range(1, len(parts), 2):
        heading = parts[i].strip()
        body = parts[i + 1] if i + 1 < len(parts) else ""
        if not _ID_RE.match(heading):
            continue  # documentation section
        description, fields = _parse_section(heading, body)
        if not fields:
            continue  # not a job (e.g. docs section with no fields)
        if heading in seen_ids:
            raise ValueError(f"duplicate job id: {heading!r}")

        if not description:
            raise ValueError(
                f"{heading}: missing description line below the heading"
            )

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
        _validate_schedule(schedule)

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
                raise ValueError(
                    f"{heading}: Next fire time is required for pend jobs"
                )
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


_REQUIRED_FIELDS = {"Type", "Schedule", "Context", "Create time",
                     "Next fire time", "Last fire time", "Execution time",
                     "Execution result", "Status"}


def _parse_section(heading: str, body: str) -> tuple[str, dict[str, str]]:
    """Parse a job section: description line + field list.

    Sections with zero recognized fields are treated as documentation
    and returned as empty — load_jobs skips them. Sections with at
    least one field are treated as jobs: unexpected lines and missing
    required fields both raise ValueError.
    """
    lines = [l for l in body.splitlines() if l.strip()]
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


# ---------- cron helpers (via croniter) ----------


def _validate_schedule(schedule: str) -> None:
    if not croniter.is_valid(schedule):
        raise ValueError(f"invalid cron expression: {schedule!r}")


def matches_now(schedule: str, now: dt.datetime) -> bool:
    """True if `schedule` matches `now` at minute granularity."""
    return croniter.match(schedule, now.replace(second=0, microsecond=0))


def next_fire(schedule: str, after: dt.datetime) -> dt.datetime | None:
    """Next datetime after `after` that matches `schedule`."""
    return croniter(schedule, after).get_next(dt.datetime)


def _format_minute(t: dt.datetime) -> str:
    return t.replace(second=0, microsecond=0).isoformat(timespec="minutes")


# ---------- due check ----------


def find_due(jobs: list[Job], now: dt.datetime) -> list[Job]:
    """Return jobs whose Next fire time has arrived.

    Only `Status: pend` jobs with a valid Next fire time are eligible.
    Uses the precomputed Next fire time rather than matching the schedule
    against `now`, so delayed ticks don't miss the job.
    """
    due: list[Job] = []
    for job in jobs:
        if job.status != STATUS_PEND:
            continue
        if not job.next_fire_time:
            continue
        try:
            nft = dt.datetime.fromisoformat(job.next_fire_time)
        except ValueError:
            continue
        if nft <= now:
            due.append(job)
    return due


# ---------- in-place field updates ----------


def _update_field(text: str, job_id: str, field_name: str, value: str) -> str:
    """Replace a single `- Field name:` list item in a job section.

    Raises ValueError if the field line doesn't exist — all fields are
    required and must be present in the file.
    """
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
        raise ValueError(
            f"{job_id}: field '- {field_name}:' not found in jobs.md"
        )
    return new_text


def _update_fields(path: Path, updates: dict[str, dict[str, str]]) -> None:
    """Apply `{job_id: {field: value, ...}}` updates to `path` in place.

    Preserves the rest of the file byte-for-byte where possible.
    """
    if not updates:
        return
    text = path.read_text()
    for job_id, fields in updates.items():
        for field_name, value in fields.items():
            text = _update_field(text, job_id, field_name, value)
    path.write_text(text)


# ---------- auto-purge stale jobs ----------


def _latest_timestamp(job: Job) -> dt.datetime | None:
    """Most recent activity timestamp for a job, or None."""
    for ts_str in (job.execution_time, job.last_fire_time):
        if ts_str:
            try:
                return dt.datetime.fromisoformat(ts_str)
            except ValueError:
                continue
    return None


def _remove_sections(path: Path, job_ids: set[str]) -> None:
    """Delete entire `## <id>` sections from the file."""
    if not job_ids:
        return
    text = path.read_text()
    for job_id in job_ids:
        pattern = re.compile(
            rf"^##\s+{re.escape(job_id)}\s*$\n(?:(?!^##\s)[\s\S])*?(?=^##\s|\Z)",
            re.MULTILINE,
        )
        text = pattern.sub("", text)
    # Clean up any resulting triple+ blank lines.
    text = re.sub(r"\n{3,}", "\n\n", text)
    path.write_text(text)


def purge_stale(
    path: Path | None = None,
    now: dt.datetime | None = None,
) -> list[str]:
    """Remove jobs in terminal status (cancel/done/fail) that have been
    inactive for 7+ days or have no parseable timestamp. Returns the
    list of purged job ids.
    """
    if path is None:
        path = JOBS_PATH
    if now is None:
        now = dt.datetime.now()
    if not path.exists():
        return []
    try:
        jobs = load_jobs(path)
    except Exception:
        return []

    to_remove: set[str] = set()
    for j in jobs:
        if j.status not in _TERMINAL_STATUSES:
            continue
        ts = _latest_timestamp(j)
        if ts is None or now - ts >= _PURGE_AFTER:
            to_remove.add(j.id)

    if to_remove:
        _remove_sections(path, to_remove)
        log.info("cron: purged %d stale job(s): %s", len(to_remove), sorted(to_remove))
    return sorted(to_remove)


# ---------- prompt building ----------


def _build_trigger_description(due: list[Job]) -> str:
    """Format due jobs into a Trigger description for spawn_claude."""
    blocks: list[str] = []
    for j in due:
        block = [f"job: {j.id}", f"description: {j.description}"]
        block.append(f"context: {j.context}")
        blocks.append("\n".join(block))

    body = "\n\n".join(blocks)
    done_lines = "\n".join(
        f"  uv run python -m physiclaw.hooks.cron done {j.id} <one-line result summary>" for j in due
    )
    fail_lines = "\n".join(
        f"  uv run python -m physiclaw.hooks.cron fail {j.id} <what went wrong>" for j in due
    )
    return (
        f"{body}\n\n"
        f"When you finish each job, mark it:\n"
        f"  success:\n{done_lines}\n"
        f"  failure:\n{fail_lines}"
    )


# ---------- hook ----------


@register
async def cron() -> Trigger | None:
    try:
        jobs = load_jobs()
    except Exception:
        log.exception("cron: failed to load %s", JOBS_PATH)
        return None
    if not jobs:
        return None

    now = dt.datetime.now()

    # Housekeeping: remove terminal jobs (cancel/done/fail) older than 7 days.
    try:
        purge_stale(now=now)
    except Exception:
        log.exception("cron: purge_stale failed")

    due = find_due(jobs, now)
    if not due:
        return None

    last_stamp = _format_minute(now)
    updates: dict[str, dict[str, str]] = {}
    for j in due:
        fields: dict[str, str] = {
            "Last fire time": last_stamp,
            "Status": STATUS_FIRED,
        }
        if j.kind == KIND_ONE_TIME:
            fields["Next fire time"] = "(never)"
        else:
            nxt = next_fire(j.schedule, now)
            fields["Next fire time"] = _format_minute(nxt)
        updates[j.id] = fields

    try:
        _update_fields(JOBS_PATH, updates)
    except Exception:
        log.exception("cron: failed to write fire times to %s", JOBS_PATH)
        # Still fire — better to double-fire next tick than to miss entirely.

    description = _build_trigger_description(due)
    if len(due) == 1:
        source = f"cron:{due[0].id}"
    else:
        source = "cron:" + ",".join(j.id for j in due)
    return Trigger(description=description, source=source)


# ---------- CLI (used by the /cron skill and the agent) ----------


def _cli() -> int:
    import sys

    args = sys.argv[1:]
    cmd = args[0] if args else "verify"

    if cmd == "verify":
        if not JOBS_PATH.exists():
            print(f"OK: {JOBS_PATH} does not exist yet (no jobs)")
            return 0
        try:
            jobs = load_jobs()
        except Exception as e:
            print(f"PARSE ERROR: {e}")
            return 1
        print(f"OK: {len(jobs)} job(s) parsed from {JOBS_PATH}")
        for j in jobs:
            print(f"  [{j.kind:8s}] [{j.status:6s}] {j.id}")
            print(f"    {j.description}")
            print(f"    schedule: {j.schedule}")
            print(f"    context: {j.context}")
            print(f"    next: {j.next_fire_time or '(never)'}")
            print(f"    last: {j.last_fire_time or '(never)'}")
            print(f"    exec: {j.execution_time or '(never)'}")
            print(f"    result: {j.execution_result or '(never)'}")
        return 0

    if cmd == "jobs-to-do":
        try:
            jobs = load_jobs()
        except Exception as e:
            print(f"PARSE ERROR: {e}")
            return 1
        fired = [j for j in jobs if j.status == STATUS_FIRED]
        if not fired:
            print("no jobs to do")
            return 0
        print(f"{len(fired)} job(s) fired, awaiting agent execution:")
        for j in fired:
            print(f"  [{j.kind}] {j.id}: {j.description}")
            print(f"    fired: {j.last_fire_time}")
        return 0

    # --- job mutation commands (done/fail/cancel) ---

    if cmd in ("done", "fail", "cancel"):
        if len(args) < 2:
            print(f"usage: python -m physiclaw.hooks.cron {cmd} <job-id> [result description]")
            return 2
        job_id = args[1]
        try:
            jobs = load_jobs()
        except Exception as e:
            print(f"PARSE ERROR: {e}")
            return 1
        job = next((j for j in jobs if j.id == job_id), None)
        if job is None:
            print(f"ERROR: no job named {job_id!r} in {JOBS_PATH}")
            return 1

        now = dt.datetime.now()
        updates: dict[str, str] = {}

        if cmd in ("done", "fail"):
            result_desc = " ".join(args[2:]) if len(args) > 2 else ""
            updates["Execution time"] = _format_minute(now)
            updates["Execution result"] = result_desc or cmd
            if job.kind == KIND_ONE_TIME:
                updates["Status"] = STATUS_DONE if cmd == "done" else STATUS_FAIL
            else:
                updates["Status"] = STATUS_PEND
        elif cmd == "cancel":
            updates["Status"] = STATUS_CANCEL

        try:
            _update_fields(JOBS_PATH, {job_id: updates})
        except Exception as e:
            print(f"WRITE ERROR: {e}")
            return 1
        print(f"OK: {cmd} {job_id}")
        return 0

    if cmd == "purge":
        purged = purge_stale()
        if not purged:
            print("nothing to purge")
        else:
            print(f"purged {len(purged)} stale job(s): {', '.join(purged)}")
        return 0

    print("usage: python -m physiclaw.hooks.cron [verify|jobs-to-do|purge|done|fail|cancel] [<id>]")
    return 2


if __name__ == "__main__":
    raise SystemExit(_cli())
