"""Tests for `physiclaw.agent.engine.jobs` — engine-side job ops.

The module-level `JOBS_PATH` (imported from job_store) is captured at
session import. The autouse `_jobs_path` fixture re-points it on BOTH
modules so file I/O lands in tmp_path.

`freezegun` is used wherever the function reads `dt.datetime.now()`
internally (create_job, upsert_auto_wait_check, finish_job).
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from textwrap import dedent

import pytest
from freezegun import freeze_time

from physiclaw.agent.engine import job_store, jobs
from physiclaw.agent.engine.job_store import (
    KIND_PERIODIC,
    NEVER,
    STATUS_CANCEL,
    STATUS_DONE,
    STATUS_FAIL,
)
from physiclaw.agent.runtime.hook import Trigger


@pytest.fixture(autouse=True)
def _jobs_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Per-test jobs.md location."""
    p = tmp_path / "jobs.md"
    monkeypatch.setattr(jobs, "JOBS_PATH", p)
    monkeypatch.setattr(job_store, "JOBS_PATH", p)
    return p


def _job_text(
    *,
    job_id: str = "user-greet-2026-04-28",
    description: str = "Say hi to user",
    kind: str = "periodic",
    status: str = "pend",
    schedule: str = "0 7 * * *",
    context: str = "ten char minimum context",
    create_time: str = "2026-04-28T07:00",
    next_fire_time: str = "2026-04-29T07:00",
    last_fire_time: str = NEVER,
    execution_time: str = NEVER,
    execution_result: str = NEVER,
) -> str:
    return dedent(
        f"""\
        ## {job_id}
        {description}
        - Type: {kind}
        - Status: {status}
        - Schedule: `{schedule}`
        - Context: {context}
        - Create time: {create_time}
        - Next fire time: {next_fire_time}
        - Last fire time: {last_fire_time}
        - Execution time: {execution_time}
        - Execution result: {execution_result}
        """
    )


def _write_jobs(_jobs_path: Path, *sections: str) -> None:
    _jobs_path.parent.mkdir(parents=True, exist_ok=True)
    _jobs_path.write_text("\n".join(sections))


# ---------- fired_job_ids ----------


def test_fired_job_ids_extracts_single_id_from_cron_source() -> None:
    triggers = [Trigger(description="x", source="cron:user-greet")]

    assert jobs.fired_job_ids(triggers) == ["user-greet"]


def test_fired_job_ids_splits_comma_separated_ids() -> None:
    triggers = [Trigger(description="x", source="cron:a,b,c")]

    assert jobs.fired_job_ids(triggers) == ["a", "b", "c"]


def test_fired_job_ids_skips_non_cron_sources() -> None:
    triggers = [
        Trigger(description="x", source="phone"),
        Trigger(description="y", source="cron:job-a"),
        Trigger(description="z", source=""),
    ]

    assert jobs.fired_job_ids(triggers) == ["job-a"]


def test_fired_job_ids_skips_empty_segments_in_split() -> None:
    triggers = [Trigger(description="x", source="cron:a,,b,")]

    assert jobs.fired_job_ids(triggers) == ["a", "b"]


def test_fired_job_ids_returns_empty_for_no_triggers() -> None:
    assert jobs.fired_job_ids([]) == []


# ---------- format_fired ----------


def test_format_fired_returns_empty_when_no_cron_triggers() -> None:
    assert jobs.format_fired([Trigger(description="x", source="phone")]) == ""


def test_format_fired_returns_empty_when_load_jobs_raises(
    _jobs_path: Path, mocker, caplog: pytest.LogCaptureFixture
) -> None:
    import logging

    mocker.patch.object(jobs, "load_jobs", side_effect=RuntimeError("boom"))

    with caplog.at_level(logging.ERROR, logger="physiclaw.agent.engine.jobs"):
        out = jobs.format_fired([Trigger(description="x", source="cron:job-a")])

    assert out == ""
    assert any("failed to load jobs.md" in r.getMessage() for r in caplog.records)


def test_format_fired_returns_empty_when_no_matching_jobs(
    _jobs_path: Path,
) -> None:
    _write_jobs(_jobs_path, _job_text(job_id="other-job"))

    out = jobs.format_fired([Trigger(description="x", source="cron:not-found")])

    assert out == ""


def test_format_fired_renders_block_per_fired_job(_jobs_path: Path) -> None:
    _write_jobs(
        _jobs_path,
        _job_text(
            job_id="user-greet",
            description="Say hi",
            context="morning greeting context",
        ),
    )

    out = jobs.format_fired([Trigger(description="x", source="cron:user-greet")])

    assert out == (
        "## Scheduled jobs firing now\n\n"
        "### user-greet\n"
        "Say hi\n\n"
        "Context: morning greeting context"
    )


# ---------- create_job: validation ----------


def test_create_job_invalid_id_raises() -> None:
    with pytest.raises(
        ValueError, match=r"^invalid job id 'BadId' \(lowercase \+ digits \+ hyphens\)$"
    ):
        jobs.create_job(
            id="BadId",
            description="d",
            schedule="* * * * *",
            context="ten char minimum context",
        )


def test_create_job_invalid_kind_raises() -> None:
    with pytest.raises(
        ValueError, match=r"^kind must be one-time or periodic, got 'hourly'$"
    ):
        jobs.create_job(
            id="x",
            description="d",
            schedule="* * * * *",
            context="ten char minimum context",
            kind="hourly",
        )


def test_create_job_invalid_schedule_raises() -> None:
    with pytest.raises(ValueError, match=r"^invalid cron expression"):
        jobs.create_job(
            id="x",
            description="d",
            schedule="not cron",
            context="ten char minimum context",
        )


def test_create_job_short_context_raises() -> None:
    with pytest.raises(ValueError, match=r"^context must be at least 10 characters$"):
        jobs.create_job(
            id="x",
            description="d",
            schedule="* * * * *",
            context="short",
        )


def test_create_job_empty_description_raises() -> None:
    with pytest.raises(ValueError, match=r"^description is required$"):
        jobs.create_job(
            id="x",
            description="   ",
            schedule="* * * * *",
            context="ten char minimum context",
        )


def test_create_job_duplicate_id_raises(_jobs_path: Path) -> None:
    _write_jobs(_jobs_path, _job_text(job_id="dup-id"))

    with pytest.raises(ValueError, match=r"^job id already exists: 'dup-id'$"):
        jobs.create_job(
            id="dup-id",
            description="d",
            schedule="* * * * *",
            context="ten char minimum context",
        )


def test_create_job_terminal_duplicate_id_still_raises(_jobs_path: Path) -> None:
    # Even a terminal entry blocks the id — the agent must pick fresh.
    _write_jobs(
        _jobs_path,
        _job_text(
            job_id="closed-id",
            status="done",
            next_fire_time=NEVER,
            execution_time="2026-04-28T07:30",
            last_fire_time="2026-04-28T07:00",
        ),
    )

    with pytest.raises(ValueError, match=r"already exists"):
        jobs.create_job(
            id="closed-id",
            description="d",
            schedule="* * * * *",
            context="ten char minimum context",
        )


# ---------- create_job: happy path ----------


def test_create_job_writes_block_with_jobs_header_when_file_missing(
    _jobs_path: Path,
) -> None:
    with freeze_time("2026-04-28T07:00:00"):
        jobs.create_job(
            id="user-greet",
            description="Say hi",
            schedule="0 7 * * *",
            context="morning greeting context",
        )

    text = _jobs_path.read_text()
    assert text.startswith("# Jobs\n")
    assert "## user-greet" in text
    assert "- Type: one-time" in text  # default kind
    assert "- Schedule: `0 7 * * *`" in text


def test_create_job_appends_block_when_file_already_exists(
    _jobs_path: Path,
) -> None:
    _write_jobs(_jobs_path, _job_text(job_id="existing"))

    with freeze_time("2026-04-28T07:00:00"):
        jobs.create_job(
            id="new-job",
            description="Hi",
            schedule="0 8 * * *",
            context="another context here",
        )

    text = _jobs_path.read_text()
    # "# Jobs" header NOT re-added on append.
    assert text.count("# Jobs") == 0
    assert "## existing" in text
    assert "## new-job" in text


def test_create_job_records_create_time_from_now(_jobs_path: Path) -> None:
    with freeze_time("2026-04-28T14:30:00"):
        jobs.create_job(
            id="x",
            description="d",
            schedule="0 * * * *",
            context="ten chars at least",
        )

    text = _jobs_path.read_text()
    assert "- Create time: 2026-04-28T14:30" in text


def test_create_job_records_next_fire_from_cron_schedule(
    _jobs_path: Path,
) -> None:
    with freeze_time("2026-04-28T14:30:00"):
        jobs.create_job(
            id="x",
            description="d",
            schedule="0 * * * *",  # next minute :00
            context="ten chars at least",
        )

    text = _jobs_path.read_text()
    assert "- Next fire time: 2026-04-28T15:00" in text


def test_create_job_periodic_rejects_high_frequency_schedule(
    _jobs_path: Path,
) -> None:
    # A */5 periodic "reply watcher" loops forever at full-agent cost
    # (done re-arms it) — refused with a pointer to one-time.
    with (
        freeze_time("2026-04-28T07:00:00"),
        pytest.raises(ValueError, match="ONE-TIME"),
    ):
        jobs.create_job(
            id="watcher",
            description="d",
            schedule="*/5 * * * *",
            context="ten chars at least",
            kind=KIND_PERIODIC,
        )

    assert not _jobs_path.exists()


def test_create_job_one_time_allows_high_frequency_schedule(
    _jobs_path: Path,
) -> None:
    # The floor is periodic-only — a one-time job fires once no matter
    # what period its cron expression implies.
    with freeze_time("2026-04-28T07:00:00"):
        jobs.create_job(
            id="check-once",
            description="d",
            schedule="*/5 * * * *",
            context="ten chars at least",
        )

    assert "- Type: one-time" in _jobs_path.read_text()


def test_min_fire_gap_uses_smallest_gap_of_mixed_schedule() -> None:
    # `0,10 9 * * *` alternates 10 min and ~24 h gaps — the 10 min
    # burst is what the floor must catch.
    gap = job_store.min_fire_gap("0,10 9 * * *", dt.datetime(2026, 4, 28, 7, 0))
    assert gap == dt.timedelta(minutes=10)


def test_create_job_periodic_kind_records_periodic(_jobs_path: Path) -> None:
    with freeze_time("2026-04-28T07:00:00"):
        jobs.create_job(
            id="repeat-job",
            description="d",
            schedule="0 7 * * *",
            context="ten chars at least",
            kind=KIND_PERIODIC,
        )

    assert "- Type: periodic" in _jobs_path.read_text()


# ---------- upsert_auto_wait_check ----------


def test_upsert_auto_wait_creates_when_absent(_jobs_path: Path) -> None:
    target = dt.datetime(2026, 4, 28, 14, 30)

    with freeze_time("2026-04-28T14:00:00"):
        jobs.upsert_auto_wait_check(target)

    text = _jobs_path.read_text()
    assert f"## {jobs.AUTO_WAIT_JOB_ID}" in text
    assert "- Schedule: `30 14 28 4 *`" in text
    assert "- Next fire time: 2026-04-28T14:30" in text


def test_upsert_auto_wait_updates_existing_to_pend_with_new_schedule(
    _jobs_path: Path,
) -> None:
    # Pre-populate a stale auto-wait entry in done state.
    _write_jobs(
        _jobs_path,
        _job_text(
            job_id=jobs.AUTO_WAIT_JOB_ID,
            status="done",
            schedule="0 0 1 1 *",
            next_fire_time=NEVER,
            last_fire_time="2026-04-27T12:00",
            execution_time="2026-04-27T12:30",
            execution_result="prior run",
        ),
    )
    target = dt.datetime(2026, 4, 28, 14, 30)

    with freeze_time("2026-04-28T14:00:00"):
        jobs.upsert_auto_wait_check(target)

    text = _jobs_path.read_text()
    assert "- Status: pend" in text
    assert "- Schedule: `30 14 28 4 *`" in text
    assert "- Next fire time: 2026-04-28T14:30" in text
    assert "- Last fire time: (never)" in text
    assert "- Execution time: (never)" in text
    assert "- Execution result: (never)" in text


# ---------- get_job ----------


def test_get_job_returns_full_job_for_known_id(_jobs_path: Path) -> None:
    _write_jobs(_jobs_path, _job_text(job_id="found"))

    j = jobs.get_job("found")

    assert j.id == "found"
    assert j.description == "Say hi to user"


def test_get_job_raises_for_unknown_id(_jobs_path: Path) -> None:
    _write_jobs(_jobs_path, _job_text(job_id="other"))

    with pytest.raises(ValueError, match=r"^no job with id 'missing'$"):
        jobs.get_job("missing")


# ---------- finish_job ----------


@pytest.mark.parametrize("bad_status", ["pend", "fired", "running", "started"])
def test_finish_job_invalid_status_raises(_jobs_path: Path, bad_status: str) -> None:
    _write_jobs(_jobs_path, _job_text(job_id="x", status="fired"))

    with pytest.raises(
        ValueError,
        match=r"^status must be one of \['cancel', 'done', 'fail'\], got",
    ):
        jobs.finish_job(id="x", status=bad_status, recap="r")


def test_finish_job_unknown_id_raises(_jobs_path: Path) -> None:
    _write_jobs(_jobs_path, _job_text(job_id="other", status="fired"))

    with pytest.raises(ValueError, match=r"^no job with id 'missing'$"):
        jobs.finish_job(id="missing", status=STATUS_DONE, recap="r")


def test_finish_job_already_terminal_raises(_jobs_path: Path) -> None:
    _write_jobs(
        _jobs_path,
        _job_text(
            job_id="closed",
            status="done",
            next_fire_time=NEVER,
            execution_time="2026-04-28T07:30",
            last_fire_time="2026-04-28T07:00",
        ),
    )

    with pytest.raises(
        ValueError, match=r"^job 'closed' is already in terminal status 'done'$"
    ):
        jobs.finish_job(id="closed", status=STATUS_DONE, recap="r")


def test_finish_job_one_time_done_marks_done(_jobs_path: Path) -> None:
    _write_jobs(
        _jobs_path,
        _job_text(
            job_id="one-time",
            kind="one-time",
            status="fired",
        ),
    )

    with freeze_time("2026-04-28T08:00:00"):
        jobs.finish_job(id="one-time", status=STATUS_DONE, recap="all good")

    text = _jobs_path.read_text()
    assert "- Status: done" in text
    assert "- Execution time: 2026-04-28T08:00" in text
    assert "- Execution result: all good" in text


def test_finish_job_periodic_done_resets_to_pend(_jobs_path: Path) -> None:
    _write_jobs(
        _jobs_path,
        _job_text(
            job_id="periodic",
            kind="periodic",
            status="fired",
        ),
    )

    with freeze_time("2026-04-28T08:00:00"):
        jobs.finish_job(id="periodic", status=STATUS_DONE, recap="ok")

    text = _jobs_path.read_text()
    assert "- Status: pend" in text


def test_finish_job_periodic_done_reply_flags_rearm(_jobs_path: Path) -> None:
    # The re-fire trap: an agent reads "done" as "over" while the
    # periodic job re-arms and fires again minutes later. The tool
    # reply must state the re-arm and the permanent way out (cancel).
    _write_jobs(
        _jobs_path,
        _job_text(
            job_id="periodic",
            kind="periodic",
            status="fired",
            next_fire_time="2026-04-29T07:00",
        ),
    )

    with freeze_time("2026-04-28T08:00:00"):
        out = jobs.finish_job(id="periodic", status=STATUS_DONE, recap="ok")

    assert "RE-ARMED" in out
    assert "2026-04-29T07:00" in out
    assert "cancel" in out


def test_finish_job_one_time_reply_is_plain(_jobs_path: Path) -> None:
    _write_jobs(
        _jobs_path,
        _job_text(
            job_id="one-shot",
            kind="one-time",
            status="fired",
        ),
    )

    with freeze_time("2026-04-28T08:00:00"):
        out = jobs.finish_job(id="one-shot", status=STATUS_DONE, recap="ok")

    assert out == "finished job 'one-shot' as done"


def test_finish_job_periodic_fail_also_resets_to_pend(_jobs_path: Path) -> None:
    _write_jobs(
        _jobs_path,
        _job_text(
            job_id="periodic",
            kind="periodic",
            status="fired",
        ),
    )

    with freeze_time("2026-04-28T08:00:00"):
        jobs.finish_job(id="periodic", status=STATUS_FAIL, recap="boom")

    text = _jobs_path.read_text()
    assert "- Status: pend" in text


def test_finish_job_periodic_cancel_stays_canceled(_jobs_path: Path) -> None:
    # cancel is permanent even on periodic jobs.
    _write_jobs(
        _jobs_path,
        _job_text(
            job_id="periodic",
            kind="periodic",
            status="fired",
        ),
    )

    with freeze_time("2026-04-28T08:00:00"):
        jobs.finish_job(id="periodic", status=STATUS_CANCEL, recap="stop")

    text = _jobs_path.read_text()
    assert "- Status: cancel" in text


def test_finish_job_uses_status_as_recap_when_recap_blank(
    _jobs_path: Path,
) -> None:
    _write_jobs(
        _jobs_path,
        _job_text(
            job_id="x",
            kind="one-time",
            status="fired",
        ),
    )

    with freeze_time("2026-04-28T08:00:00"):
        jobs.finish_job(id="x", status=STATUS_DONE, recap="   ")

    text = _jobs_path.read_text()
    # Empty recap → status name as recap.
    assert "- Execution result: done" in text
