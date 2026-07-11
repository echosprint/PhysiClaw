"""Tests for `physiclaw.agent.claude.skills.jobs.jobs` — Claude-side jobs CLI."""

from __future__ import annotations

import argparse

import pytest

from physiclaw.agent.claude.skills.jobs import jobs as jobs_cli
from physiclaw.agent.engine.job_store import (
    KIND_ONE_TIME,
    KIND_PERIODIC,
    STATUS_DONE,
    STATUS_PEND,
    Job,
)


def _ns(**kw) -> argparse.Namespace:
    return argparse.Namespace(**kw)


# ---------- _cmd_create ----------


def test_cmd_create_happy_path(mocker, capsys: pytest.CaptureFixture) -> None:
    spy = mocker.patch.object(jobs_cli.jobs, "create_job")
    args = _ns(
        id="u-x-2026-04-28",
        description="d",
        schedule="0 7 * * *",
        context="some context here",
        kind=KIND_ONE_TIME,
    )

    rc = jobs_cli._cmd_create(args)

    assert rc == 0
    assert "created u-x-2026-04-28" in capsys.readouterr().out
    spy.assert_called_once_with(
        id="u-x-2026-04-28",
        description="d",
        schedule="0 7 * * *",
        context="some context here",
        kind=KIND_ONE_TIME,
    )


def test_cmd_create_value_error_prints_to_stderr_and_returns_2(
    mocker, capsys: pytest.CaptureFixture
) -> None:
    mocker.patch.object(jobs_cli.jobs, "create_job", side_effect=ValueError("bad id"))
    args = _ns(
        id="bad",
        description="d",
        schedule="0 7 * * *",
        context="ctx",
        kind=KIND_ONE_TIME,
    )

    rc = jobs_cli._cmd_create(args)
    err = capsys.readouterr().err

    assert rc == 2
    assert "error: bad id" in err


# ---------- _cmd_list ----------


def _job(**kw) -> Job:
    base = dict(
        id="j",
        kind=KIND_ONE_TIME,
        schedule="0 7 * * *",
        description="something",
        status=STATUS_PEND,
        context="c",
        next_fire_time="2026-04-29T07:00",
    )
    base.update(kw)
    return Job(**base)


def test_cmd_list_all_lists_jobs(mocker, capsys: pytest.CaptureFixture) -> None:
    mocker.patch.object(
        jobs_cli,
        "load_jobs",
        return_value=[
            _job(id="a", description="alpha"),
            _job(id="b", description="beta"),
        ],
    )

    rc = jobs_cli._cmd_list(_ns(status="all"))
    out = capsys.readouterr().out

    assert rc == 0
    assert "a" in out and "b" in out
    assert "alpha" in out and "beta" in out


def test_cmd_list_filters_by_status(mocker, capsys: pytest.CaptureFixture) -> None:
    mocker.patch.object(
        jobs_cli,
        "load_jobs",
        return_value=[
            _job(id="a", status=STATUS_PEND),
            _job(id="b", status=STATUS_DONE),
        ],
    )

    rc = jobs_cli._cmd_list(_ns(status=STATUS_DONE))
    out = capsys.readouterr().out

    assert rc == 0
    assert "b" in out
    assert "a  [pend]" not in out


def test_cmd_list_empty_message(mocker, capsys: pytest.CaptureFixture) -> None:
    mocker.patch.object(jobs_cli, "load_jobs", return_value=[])

    rc = jobs_cli._cmd_list(_ns(status="all"))
    out = capsys.readouterr().out

    assert rc == 0
    assert "(no jobs with status=all)" in out


def test_cmd_list_load_value_error_returns_2(
    mocker, capsys: pytest.CaptureFixture
) -> None:
    mocker.patch.object(jobs_cli, "load_jobs", side_effect=ValueError("malformed"))

    rc = jobs_cli._cmd_list(_ns(status="all"))
    err = capsys.readouterr().err

    assert rc == 2
    assert "error loading jobs.md: malformed" in err


def test_cmd_list_truncates_description_to_80_chars(
    mocker, capsys: pytest.CaptureFixture
) -> None:
    long_desc = "x" * 200
    mocker.patch.object(
        jobs_cli,
        "load_jobs",
        return_value=[_job(id="long", description=long_desc)],
    )

    jobs_cli._cmd_list(_ns(status="all"))
    out = capsys.readouterr().out

    # Truncated to 80, full 200-char string not present.
    assert "x" * 80 in out
    assert "x" * 81 not in out


def test_cmd_list_uses_first_line_of_multiline_description(
    mocker, capsys: pytest.CaptureFixture
) -> None:
    mocker.patch.object(
        jobs_cli,
        "load_jobs",
        return_value=[_job(id="m", description="first line\nsecond line")],
    )

    jobs_cli._cmd_list(_ns(status="all"))
    out = capsys.readouterr().out

    assert "first line" in out
    assert "second line" not in out


def test_cmd_list_renders_dash_when_no_next_fire_time(
    mocker, capsys: pytest.CaptureFixture
) -> None:
    mocker.patch.object(
        jobs_cli,
        "load_jobs",
        return_value=[_job(id="nox", next_fire_time="")],
    )

    jobs_cli._cmd_list(_ns(status="all"))
    out = capsys.readouterr().out

    assert "next=-" in out


# ---------- _cmd_get ----------


def test_cmd_get_prints_all_fields(mocker, capsys: pytest.CaptureFixture) -> None:
    j = _job(
        id="g",
        description="get-desc",
        context="ctx-here",
        last_fire_time="2026-04-28T07:00",
        execution_time="2026-04-28T07:05",
        execution_result="ok",
    )
    mocker.patch.object(jobs_cli.jobs, "get_job", return_value=j)

    rc = jobs_cli._cmd_get(_ns(id="g"))
    out = capsys.readouterr().out

    assert rc == 0
    for needle in [
        "id:               g",
        "kind:             one-time",
        "status:           pend",
        "description:      get-desc",
        "context:          ctx-here",
        "next fire time:   2026-04-29T07:00",
        "last fire time:   2026-04-28T07:00",
        "execution time:   2026-04-28T07:05",
        "execution result: ok",
    ]:
        assert needle in out, f"expected {needle!r} in output"


def test_cmd_get_renders_dashes_for_empty_optional_fields(
    mocker, capsys: pytest.CaptureFixture
) -> None:
    j = _job(
        id="g",
        next_fire_time="",
        last_fire_time="",
        execution_time="",
        execution_result="",
    )
    mocker.patch.object(jobs_cli.jobs, "get_job", return_value=j)

    jobs_cli._cmd_get(_ns(id="g"))
    out = capsys.readouterr().out

    assert "next fire time:   -" in out
    assert "last fire time:   -" in out
    assert "execution time:   -" in out
    assert "execution result: -" in out


def test_cmd_get_value_error_returns_2(mocker, capsys: pytest.CaptureFixture) -> None:
    mocker.patch.object(
        jobs_cli.jobs,
        "get_job",
        side_effect=ValueError("no such job"),
    )

    rc = jobs_cli._cmd_get(_ns(id="nope"))
    err = capsys.readouterr().err

    assert rc == 2
    assert "error: no such job" in err


# ---------- _cmd_finish ----------


def test_cmd_finish_happy_path(mocker, capsys: pytest.CaptureFixture) -> None:
    spy = mocker.patch.object(jobs_cli.jobs, "finish_job")
    args = _ns(id="g", status=STATUS_DONE, recap="all good")

    rc = jobs_cli._cmd_finish(args)

    assert rc == 0
    assert "finished g as done" in capsys.readouterr().out
    spy.assert_called_once_with(id="g", status=STATUS_DONE, recap="all good")


def test_cmd_finish_value_error_returns_2(
    mocker, capsys: pytest.CaptureFixture
) -> None:
    mocker.patch.object(
        jobs_cli.jobs,
        "finish_job",
        side_effect=ValueError("bad transition"),
    )

    rc = jobs_cli._cmd_finish(_ns(id="g", status=STATUS_DONE, recap="r"))
    err = capsys.readouterr().err

    assert rc == 2
    assert "error: bad transition" in err


# ---------- main() argparse wiring ----------


def test_main_dispatches_create(mocker, monkeypatch: pytest.MonkeyPatch) -> None:
    spy = mocker.patch.object(jobs_cli.jobs, "create_job")
    monkeypatch.setattr(
        "sys.argv",
        [
            "jobs.py",
            "create",
            "--id",
            "u-x-2026-04-28",
            "--schedule",
            "0 7 * * *",
            "--description",
            "do thing",
            "--context",
            "ten char ctx",
        ],
    )

    with pytest.raises(SystemExit) as exc:
        jobs_cli.main()

    assert exc.value.code == 0
    spy.assert_called_once()


def test_main_dispatches_list(
    mocker, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    mocker.patch.object(jobs_cli, "load_jobs", return_value=[])
    monkeypatch.setattr("sys.argv", ["jobs.py", "list"])

    with pytest.raises(SystemExit) as exc:
        jobs_cli.main()

    assert exc.value.code == 0
    assert "no jobs" in capsys.readouterr().out


def test_main_dispatches_get(mocker, monkeypatch: pytest.MonkeyPatch) -> None:
    j = _job(id="x")
    spy = mocker.patch.object(jobs_cli.jobs, "get_job", return_value=j)
    monkeypatch.setattr("sys.argv", ["jobs.py", "get", "--id", "x"])

    with pytest.raises(SystemExit) as exc:
        jobs_cli.main()

    assert exc.value.code == 0
    spy.assert_called_once_with("x")


def test_main_dispatches_finish(mocker, monkeypatch: pytest.MonkeyPatch) -> None:
    spy = mocker.patch.object(jobs_cli.jobs, "finish_job")
    monkeypatch.setattr(
        "sys.argv",
        [
            "jobs.py",
            "finish",
            "--id",
            "x",
            "--status",
            STATUS_DONE,
            "--recap",
            "r",
        ],
    )

    with pytest.raises(SystemExit) as exc:
        jobs_cli.main()

    assert exc.value.code == 0
    spy.assert_called_once_with(id="x", status=STATUS_DONE, recap="r")


def test_main_requires_subcommand(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("sys.argv", ["jobs.py"])

    with pytest.raises(SystemExit) as exc:
        jobs_cli.main()

    # argparse exits 2 when a required argument is missing.
    assert exc.value.code == 2


def test_main_rejects_invalid_kind(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "jobs.py",
            "create",
            "--id",
            "x",
            "--schedule",
            "* * * * *",
            "--description",
            "d",
            "--context",
            "c",
            "--kind",
            "weird",
        ],
    )

    with pytest.raises(SystemExit) as exc:
        jobs_cli.main()

    assert exc.value.code == 2


def test_main_create_default_kind_is_one_time(
    mocker, monkeypatch: pytest.MonkeyPatch
) -> None:
    spy = mocker.patch.object(jobs_cli.jobs, "create_job")
    monkeypatch.setattr(
        "sys.argv",
        [
            "jobs.py",
            "create",
            "--id",
            "x",
            "--schedule",
            "* * * * *",
            "--description",
            "d",
            "--context",
            "ten char ctx",
        ],
    )

    with pytest.raises(SystemExit):
        jobs_cli.main()

    assert spy.call_args.kwargs["kind"] == KIND_ONE_TIME


def test_main_create_accepts_periodic(mocker, monkeypatch: pytest.MonkeyPatch) -> None:
    spy = mocker.patch.object(jobs_cli.jobs, "create_job")
    monkeypatch.setattr(
        "sys.argv",
        [
            "jobs.py",
            "create",
            "--id",
            "x",
            "--schedule",
            "* * * * *",
            "--description",
            "d",
            "--context",
            "ten char ctx",
            "--kind",
            KIND_PERIODIC,
        ],
    )

    with pytest.raises(SystemExit):
        jobs_cli.main()

    assert spy.call_args.kwargs["kind"] == KIND_PERIODIC
