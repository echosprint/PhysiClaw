"""Tests for `physiclaw.cli.studio` — a frontend only: it finds the
running server and never starts one."""

from __future__ import annotations

import importlib

import httpx
import pytest
import typer

studio_mod = importlib.import_module("physiclaw.cli.studio")

BASE = "http://127.0.0.1:8048"


def _ready_script(monkeypatch, answers: list):
    """`check_ready_once` answers from a script — a value is returned,
    an exception is raised; the last entry repeats."""
    from physiclaw.common import ready

    def fake(base, *, timeout=2.0):
        a = answers.pop(0) if len(answers) > 1 else answers[0]
        if isinstance(a, Exception):
            raise a
        return a

    monkeypatch.setattr(ready, "check_ready_once", fake)


def _no_live(monkeypatch, live=None):
    from physiclaw.common import config, runtime_state

    monkeypatch.setattr(runtime_state, "read_live", lambda: live)
    monkeypatch.setattr(config, "server_url", lambda: BASE)


def test_live_server_record_wins_without_probing(monkeypatch) -> None:
    _no_live(monkeypatch, {"pid": 1, "host": "0.0.0.0", "port": 9999})
    _ready_script(monkeypatch, [httpx.ConnectError("never asked")])

    assert studio_mod._server_base() == "http://127.0.0.1:9999"


def test_a_listening_server_is_driven_even_before_ready(monkeypatch) -> None:
    _no_live(monkeypatch)
    _ready_script(monkeypatch, [False])

    assert studio_mod._server_base() == BASE


def test_nothing_listening_is_the_start_it_first_error(monkeypatch, capsys) -> None:
    _no_live(monkeypatch)
    _ready_script(monkeypatch, [httpx.ConnectError("refused")])

    with pytest.raises(typer.Exit):
        studio_mod._server_base()

    assert "physiclaw mcp" in capsys.readouterr().err


def test_listening_tells_a_refused_connect_from_any_answer(monkeypatch) -> None:
    _ready_script(monkeypatch, [httpx.ConnectError("refused")])
    assert studio_mod._listening(BASE) is False

    _ready_script(
        monkeypatch, [httpx.HTTPStatusError("503", request=None, response=None)]
    )
    assert studio_mod._listening(BASE) is True  # slow or unhappy, but there


def test_recorded_session_flag_reports_a_missing_recording(tmp_path) -> None:
    with pytest.raises(typer.BadParameter, match="wire.jsonl"):
        studio_mod._recorded_session(str(tmp_path))
