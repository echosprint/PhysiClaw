"""Tests for `physiclaw debug` — the one-command debug runner: task
seeding + staged replies, the armed wake, the reset, and the
server-start path (no live server → become one, in debug mode). An
autouse stub fakes a live server so no test ever starts a real one."""

from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from physiclaw.agent.hooks.debug import wake_path
from physiclaw.cli import app
from physiclaw.common import runtime_state
from physiclaw.common.config import DEBUG_ENV_VAR, MACRO_FAILURE_ENV_VAR
from physiclaw.debug import thread as vthread

runner = CliRunner()


@pytest.fixture(autouse=True)
def _live_server(monkeypatch: pytest.MonkeyPatch) -> None:
    """Default every test to "a server is already running" so `--task`
    never starts a real one; the server-start tests override this."""
    monkeypatch.setattr(runtime_state, "read_live", lambda: {"pid": 1})


def test_task_with_live_server_prints_the_debug_hint() -> None:
    result = runner.invoke(app, ["debug", "--task", "buy milk", "--no-wake"])

    assert result.exit_code == 0
    assert "physiclaw debug" in result.output  # the debug-mode hint


def test_task_with_no_live_server_starts_one_in_debug_mode(
    mocker, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(runtime_state, "read_live", lambda: None)
    mocker.patch.dict("os.environ")  # snapshot: the command exports env flags
    server_spy = mocker.patch("physiclaw.cli.server.server")

    result = runner.invoke(app, ["debug", "--task", "buy milk"])

    assert result.exit_code == 0
    server_spy.assert_called_once_with(hot_start=True)
    import os

    assert os.environ.get(DEBUG_ENV_VAR) == "1"
    assert os.environ.get(MACRO_FAILURE_ENV_VAR) == "1"  # the debug default


def test_no_macro_failure_leaves_the_halt_unarmed(
    mocker, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(runtime_state, "read_live", lambda: None)
    monkeypatch.delenv(MACRO_FAILURE_ENV_VAR, raising=False)
    mocker.patch.dict("os.environ")
    mocker.patch("physiclaw.cli.server.server")

    result = runner.invoke(app, ["debug", "--task", "buy milk", "--no-macro-failure"])

    assert result.exit_code == 0
    import os

    assert os.environ.get(DEBUG_ENV_VAR) == "1"
    assert os.environ.get(MACRO_FAILURE_ENV_VAR) is None


def test_no_flags_is_an_error() -> None:
    result = runner.invoke(app, ["debug"])

    assert result.exit_code != 0


def test_task_seeds_thread_stages_replies_and_arms_a_wake() -> None:
    result = runner.invoke(app, ["debug", "--task", "buy milk", "--reply", "ok"])

    assert result.exit_code == 0
    thread = vthread.load()
    assert thread.bubbles == [vthread.Bubble(sender=vthread.USER, text="buy milk")]
    assert thread.staged == ["ok"]
    assert json.loads(wake_path().read_text(encoding="utf-8"))["description"]


def test_task_no_wake_leaves_the_trigger_unarmed() -> None:
    result = runner.invoke(app, ["debug", "--task", "buy milk", "--no-wake"])

    assert result.exit_code == 0
    assert not wake_path().exists()


def test_task_resets_a_previous_script() -> None:
    runner.invoke(app, ["debug", "--task", "buy milk", "--reply", "ok"])

    runner.invoke(app, ["debug", "--task", "buy eggs"])

    thread = vthread.load()
    assert thread.bubbles == [vthread.Bubble(sender=vthread.USER, text="buy eggs")]
    assert thread.staged == []


def test_reply_alone_stages_without_arming_a_wake() -> None:
    runner.invoke(app, ["debug", "--task", "buy milk", "--no-wake"])

    result = runner.invoke(app, ["debug", "--reply", "ok"])

    assert result.exit_code == 0
    assert vthread.load().staged == ["ok"]
    assert not wake_path().exists()


def test_reply_wakes_a_suspended_walk() -> None:
    from physiclaw.common.logger import write_json_atomic
    from physiclaw.conductor.suspension import SUSPENDED_SCHEMA, suspended_path

    suspended_path().parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(
        suspended_path(),
        {"schema": SUSPENDED_SCHEMA, "app": "demo", "playbook": "flow", "idx": 0},
    )

    result = runner.invoke(app, ["debug", "--reply", "ok"])

    assert result.exit_code == 0
    assert wake_path().exists()


def test_status_reports_the_script_state() -> None:
    runner.invoke(app, ["debug", "--task", "buy milk", "--reply", "ok"])

    result = runner.invoke(app, ["debug", "--status"])

    assert "buy milk" in result.output
    assert "ok" in result.output
    assert "wake armed: True" in result.output


def test_clear_drops_thread_and_wake() -> None:
    runner.invoke(app, ["debug", "--task", "buy milk"])

    result = runner.invoke(app, ["debug", "--clear"])

    assert result.exit_code == 0
    assert vthread.load() == vthread.Thread() and not wake_path().exists()
