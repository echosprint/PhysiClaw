"""Tests for `physiclaw.agent.hooks.cron`."""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import pytest
from freezegun import freeze_time

from physiclaw.agent.engine import job_store
from physiclaw.agent.hooks import cron
from physiclaw.agent.hooks.cron import (
    _build_trigger_description,
    _cli,
    cron as cron_hook,
)
from physiclaw.agent.engine.job_store import (
    KIND_ONE_TIME,
    KIND_PERIODIC,
    NEVER,
    STATUS_FIRED,
    STATUS_PEND,
    Job,
)


@pytest.fixture(autouse=True)
def _jobs_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    p = tmp_path / "jobs.md"
    monkeypatch.setattr(cron, "JOBS_PATH", p)
    monkeypatch.setattr(job_store, "JOBS_PATH", p)
    return p


def _job_text(
    *,
    job_id: str,
    status: str = STATUS_PEND,
    kind: str = "periodic",
    schedule: str = "0 7 * * *",
    next_fire_time: str = "2026-04-29T07:00",
) -> str:
    return dedent(
        f"""\
        ## {job_id}
        Description text
        - Type: {kind}
        - Status: {status}
        - Schedule: `{schedule}`
        - Context: ten char minimum context
        - Create time: 2026-04-28T07:00
        - Next fire time: {next_fire_time}
        - Last fire time: {NEVER}
        - Execution time: {NEVER}
        - Execution result: {NEVER}
        """
    )


# ---------- _build_trigger_description ----------


def test_build_trigger_description_includes_job_block_and_done_fail_lines() -> None:
    j = Job(
        id="x",
        kind=KIND_PERIODIC,
        schedule="* * * * *",
        description="Do thing",
        context="some context",
    )

    out = _build_trigger_description([j])

    assert "job: x" in out
    assert "description: Do thing" in out
    assert "context: some context" in out
    assert "When you finish each job" in out
    assert "physiclaw.agent.hooks.cron done x" in out
    assert "physiclaw.agent.hooks.cron fail x" in out


def test_build_trigger_description_handles_multiple_jobs() -> None:
    a = Job(
        id="a",
        kind=KIND_PERIODIC,
        schedule="* * * * *",
        description="d-a",
        context="ctx",
    )
    b = Job(
        id="b",
        kind=KIND_ONE_TIME,
        schedule="* * * * *",
        description="d-b",
        context="ctx",
    )

    out = _build_trigger_description([a, b])

    assert "job: a" in out
    assert "job: b" in out


# ---------- cron hook ----------


@pytest.mark.asyncio
async def test_cron_returns_none_when_no_jobs(_jobs_path: Path) -> None:
    assert await cron_hook() is None


@pytest.mark.asyncio
async def test_cron_returns_none_when_load_jobs_raises(
    mocker, caplog: pytest.LogCaptureFixture
) -> None:
    import logging

    mocker.patch.object(cron, "load_jobs", side_effect=RuntimeError("boom"))

    with caplog.at_level(logging.ERROR, logger="physiclaw.agent.hooks.cron"):
        out = await cron_hook()

    assert out is None
    assert any("failed to load" in r.getMessage() for r in caplog.records)


@pytest.mark.asyncio
async def test_cron_returns_none_when_no_due_jobs(_jobs_path: Path) -> None:
    _jobs_path.parent.mkdir(parents=True, exist_ok=True)
    _jobs_path.write_text(_job_text(job_id="future", next_fire_time="2027-01-01T00:00"))

    with freeze_time("2026-04-28T07:00:00"):
        out = await cron_hook()

    assert out is None


@pytest.mark.asyncio
async def test_cron_fires_due_one_time_job_and_marks_fired(
    _jobs_path: Path,
) -> None:
    _jobs_path.parent.mkdir(parents=True, exist_ok=True)
    _jobs_path.write_text(
        _job_text(
            job_id="due-one",
            kind="one-time",
            schedule="0 7 * * *",
            next_fire_time="2026-04-28T07:00",
        )
    )

    with freeze_time("2026-04-28T07:00:00"):
        trigger = await cron_hook()

    assert trigger is not None
    assert trigger.source == "cron:due-one"
    assert "due-one" in trigger.description
    text = _jobs_path.read_text()
    assert "- Status: fired" in text
    # One-time job: next fire time set to NEVER after firing.
    assert "- Next fire time: (never)" in text


@pytest.mark.asyncio
async def test_cron_periodic_job_recomputes_next_fire(
    _jobs_path: Path,
) -> None:
    _jobs_path.parent.mkdir(parents=True, exist_ok=True)
    _jobs_path.write_text(
        _job_text(
            job_id="periodic",
            kind="periodic",
            schedule="0 * * * *",
            next_fire_time="2026-04-28T07:00",
        )
    )

    with freeze_time("2026-04-28T07:00:00"):
        await cron_hook()

    text = _jobs_path.read_text()
    # Periodic schedule "0 * * * *" → next fire is 08:00.
    assert "- Next fire time: 2026-04-28T08:00" in text


@pytest.mark.asyncio
async def test_cron_multiple_due_jobs_combined_in_source(_jobs_path: Path) -> None:
    _jobs_path.parent.mkdir(parents=True, exist_ok=True)
    _jobs_path.write_text(
        _job_text(
            job_id="job-a", schedule="0 7 * * *", next_fire_time="2026-04-28T07:00"
        )
        + _job_text(
            job_id="job-b", schedule="0 7 * * *", next_fire_time="2026-04-28T07:00"
        )
    )

    with freeze_time("2026-04-28T07:00:00"):
        trigger = await cron_hook()

    assert trigger is not None
    assert trigger.source == "cron:job-a,job-b"


@pytest.mark.asyncio
async def test_cron_logs_warning_when_purge_stale_raises(
    _jobs_path: Path, mocker, caplog: pytest.LogCaptureFixture
) -> None:
    import logging

    _jobs_path.parent.mkdir(parents=True, exist_ok=True)
    _jobs_path.write_text(
        _job_text(
            job_id="x",
            schedule="0 7 * * *",
            next_fire_time="2027-01-01T07:00",
        )
    )
    mocker.patch.object(cron, "purge_stale", side_effect=RuntimeError("nope"))

    with caplog.at_level(logging.ERROR, logger="physiclaw.agent.hooks.cron"):
        await cron_hook()

    assert any("purge_stale failed" in r.getMessage() for r in caplog.records)


@pytest.mark.asyncio
async def test_cron_still_fires_when_update_fields_raises(
    _jobs_path: Path, mocker, caplog: pytest.LogCaptureFixture
) -> None:
    import logging

    _jobs_path.parent.mkdir(parents=True, exist_ok=True)
    _jobs_path.write_text(
        _job_text(
            job_id="due",
            schedule="0 7 * * *",
            next_fire_time="2026-04-28T07:00",
        )
    )
    mocker.patch.object(cron, "update_fields", side_effect=OSError("disk full"))

    with caplog.at_level(logging.ERROR, logger="physiclaw.agent.hooks.cron"):
        with freeze_time("2026-04-28T07:00:00"):
            trigger = await cron_hook()

    # Trigger is still produced — better double-fire than miss.
    assert trigger is not None
    assert trigger.source == "cron:due"
    assert any("failed to write fire times" in r.getMessage() for r in caplog.records)


# ---------- CLI ----------


def _run_cli(monkeypatch: pytest.MonkeyPatch, *args: str) -> int:
    monkeypatch.setattr("sys.argv", ["cron", *args])
    return _cli()


def test_cli_verify_no_file(
    _jobs_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    rc = _run_cli(monkeypatch, "verify")

    assert rc == 0
    assert "does not exist yet" in capsys.readouterr().out


def test_cli_default_command_is_verify(
    _jobs_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    rc = _run_cli(monkeypatch)

    assert rc == 0
    assert "does not exist yet" in capsys.readouterr().out


def test_cli_verify_lists_jobs(
    _jobs_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    _jobs_path.parent.mkdir(parents=True, exist_ok=True)
    _jobs_path.write_text(
        _job_text(
            job_id="alpha",
            schedule="0 7 * * *",
            next_fire_time="2026-04-29T07:00",
        )
    )

    rc = _run_cli(monkeypatch, "verify")
    out = capsys.readouterr().out

    assert rc == 0
    assert "1 job(s) parsed" in out
    assert "alpha" in out
    assert "0 7 * * *" in out


def test_cli_verify_parse_error(
    _jobs_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    _jobs_path.parent.mkdir(parents=True, exist_ok=True)
    _jobs_path.write_text("## bad\n- Type: garbage\n")

    rc = _run_cli(monkeypatch, "verify")

    assert rc == 1
    assert "PARSE ERROR" in capsys.readouterr().out


def test_cli_jobs_to_do_none(
    _jobs_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    _jobs_path.parent.mkdir(parents=True, exist_ok=True)
    _jobs_path.write_text(
        _job_text(
            job_id="x",
            schedule="0 7 * * *",
            next_fire_time="2026-04-29T07:00",
        )
    )

    rc = _run_cli(monkeypatch, "jobs-to-do")

    assert rc == 0
    assert "no jobs to do" in capsys.readouterr().out


def test_cli_jobs_to_do_lists_fired(
    _jobs_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    _jobs_path.parent.mkdir(parents=True, exist_ok=True)
    _jobs_path.write_text(
        _job_text(
            job_id="firedjob",
            status=STATUS_FIRED,
            schedule="0 7 * * *",
            next_fire_time="2026-04-29T07:00",
        )
    )

    rc = _run_cli(monkeypatch, "jobs-to-do")
    out = capsys.readouterr().out

    assert rc == 0
    assert "1 job(s) fired" in out
    assert "firedjob" in out


def test_cli_jobs_to_do_parse_error(
    _jobs_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    _jobs_path.parent.mkdir(parents=True, exist_ok=True)
    _jobs_path.write_text("## bad\n- Type: garbage\n")

    rc = _run_cli(monkeypatch, "jobs-to-do")

    assert rc == 1
    assert "PARSE ERROR" in capsys.readouterr().out


def test_cli_done_missing_arg(
    _jobs_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    rc = _run_cli(monkeypatch, "done")

    assert rc == 2
    assert "usage:" in capsys.readouterr().out


def test_cli_done_unknown_id(
    _jobs_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    _jobs_path.parent.mkdir(parents=True, exist_ok=True)
    _jobs_path.write_text(
        _job_text(
            job_id="real",
            schedule="0 7 * * *",
            next_fire_time="2026-04-29T07:00",
        )
    )

    rc = _run_cli(monkeypatch, "done", "nope")

    assert rc == 1
    assert "no job named 'nope'" in capsys.readouterr().out


def test_cli_done_parse_error(
    _jobs_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    _jobs_path.parent.mkdir(parents=True, exist_ok=True)
    _jobs_path.write_text("## bad\n- Type: garbage\n")

    rc = _run_cli(monkeypatch, "done", "anything")

    assert rc == 1
    assert "PARSE ERROR" in capsys.readouterr().out


def test_cli_done_one_time_marks_done(
    _jobs_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    _jobs_path.parent.mkdir(parents=True, exist_ok=True)
    _jobs_path.write_text(
        _job_text(
            job_id="ot",
            kind="one-time",
            status=STATUS_FIRED,
            schedule="0 7 * * *",
            next_fire_time=NEVER,
        )
    )

    rc = _run_cli(monkeypatch, "done", "ot", "all", "good")

    assert rc == 0
    assert "OK: done ot" in capsys.readouterr().out
    text = _jobs_path.read_text()
    assert "- Status: done" in text
    assert "- Execution result: all good" in text


def test_cli_fail_one_time_marks_fail(
    _jobs_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    _jobs_path.parent.mkdir(parents=True, exist_ok=True)
    _jobs_path.write_text(
        _job_text(
            job_id="ot",
            kind="one-time",
            status=STATUS_FIRED,
            schedule="0 7 * * *",
            next_fire_time=NEVER,
        )
    )

    rc = _run_cli(monkeypatch, "fail", "ot")

    assert rc == 0
    text = _jobs_path.read_text()
    assert "- Status: fail" in text
    assert "- Execution result: fail" in text


def test_cli_done_periodic_resets_to_pend(
    _jobs_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    _jobs_path.parent.mkdir(parents=True, exist_ok=True)
    _jobs_path.write_text(
        _job_text(
            job_id="p",
            kind="periodic",
            status=STATUS_FIRED,
            schedule="0 * * * *",
            next_fire_time="2026-04-28T08:00",
        )
    )

    rc = _run_cli(monkeypatch, "done", "p", "ok")

    assert rc == 0
    text = _jobs_path.read_text()
    assert "- Status: pend" in text
    # The claude-code engine marks jobs through this CLI — it must get
    # the same re-arm warning the in-process finish_job tool gives.
    out = capsys.readouterr().out
    assert "PERIODIC" in out
    assert "2026-04-28T08:00" in out
    assert "cancel p" in out


def test_cli_done_one_time_prints_no_rearm_note(
    _jobs_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    _jobs_path.parent.mkdir(parents=True, exist_ok=True)
    _jobs_path.write_text(
        _job_text(
            job_id="o",
            kind="one-time",
            status=STATUS_FIRED,
            schedule="0 7 28 4 *",
            next_fire_time=NEVER,
        )
    )

    rc = _run_cli(monkeypatch, "done", "o", "ok")

    assert rc == 0
    assert "PERIODIC" not in capsys.readouterr().out


def test_cli_cancel_marks_cancel(
    _jobs_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    _jobs_path.parent.mkdir(parents=True, exist_ok=True)
    _jobs_path.write_text(
        _job_text(
            job_id="c",
            schedule="0 7 * * *",
            next_fire_time="2026-04-29T07:00",
        )
    )

    rc = _run_cli(monkeypatch, "cancel", "c")

    assert rc == 0
    text = _jobs_path.read_text()
    assert "- Status: cancel" in text


def test_cli_done_write_error(
    _jobs_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mocker,
    capsys: pytest.CaptureFixture,
) -> None:
    _jobs_path.parent.mkdir(parents=True, exist_ok=True)
    _jobs_path.write_text(
        _job_text(
            job_id="x",
            kind="one-time",
            status=STATUS_FIRED,
            schedule="0 7 * * *",
            next_fire_time=NEVER,
        )
    )
    mocker.patch.object(cron, "update_fields", side_effect=OSError("boom"))

    rc = _run_cli(monkeypatch, "done", "x")

    assert rc == 1
    assert "WRITE ERROR" in capsys.readouterr().out


def test_cli_purge_nothing(
    _jobs_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mocker,
    capsys: pytest.CaptureFixture,
) -> None:
    mocker.patch.object(cron, "purge_stale", return_value=[])

    rc = _run_cli(monkeypatch, "purge")

    assert rc == 0
    assert "nothing to purge" in capsys.readouterr().out


def test_cli_purge_with_results(
    _jobs_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mocker,
    capsys: pytest.CaptureFixture,
) -> None:
    mocker.patch.object(cron, "purge_stale", return_value=["a", "b"])

    rc = _run_cli(monkeypatch, "purge")
    out = capsys.readouterr().out

    assert rc == 0
    assert "purged 2 stale job(s)" in out
    assert "a, b" in out


def test_cli_unknown_command_returns_2(
    _jobs_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    rc = _run_cli(monkeypatch, "unknownsubcmd")

    assert rc == 2
    assert "usage:" in capsys.readouterr().out
