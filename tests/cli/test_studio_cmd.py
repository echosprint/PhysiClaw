"""Tests for `physiclaw.cli.studio` — finding or starting the server the
studio drives, and the readiness wait."""

from __future__ import annotations

import importlib

import httpx
import pytest
import typer

studio_mod = importlib.import_module("physiclaw.cli.studio")

BASE = "http://127.0.0.1:8048"


class FakeProc:
    def __init__(self, exit_after: int | None = None):
        self.returncode = None
        self._polls = 0
        self._exit_after = exit_after

    def poll(self):
        self._polls += 1
        if self._exit_after is not None and self._polls > self._exit_after:
            self.returncode = 3
        return self.returncode


def _ready_script(monkeypatch, answers: list):
    """`_ready` answers from a script — a value is returned, an exception
    is raised; the last entry repeats."""

    def fake(base):
        a = answers.pop(0) if len(answers) > 1 else answers[0]
        if isinstance(a, Exception):
            raise a
        return a

    monkeypatch.setattr(studio_mod, "_ready", fake)
    monkeypatch.setattr(studio_mod.time, "sleep", lambda s: None)


def _no_live(monkeypatch, live=None):
    from physiclaw.common import config, runtime_state

    monkeypatch.setattr(runtime_state, "read_live", lambda: live)
    monkeypatch.setattr(config, "server_url", lambda: BASE)


def test_live_server_record_wins_and_is_never_doubled(monkeypatch) -> None:
    _no_live(monkeypatch, {"pid": 1, "host": "0.0.0.0", "port": 9999})
    _ready_script(monkeypatch, [True])
    monkeypatch.setattr(studio_mod, "_spawn_server", lambda b: pytest.fail("spawned"))

    assert studio_mod._server_base() == "http://127.0.0.1:9999"


def test_a_listening_server_is_reused_even_before_ready(monkeypatch) -> None:
    _no_live(monkeypatch)
    _ready_script(monkeypatch, [False, False, True])
    monkeypatch.setattr(studio_mod, "_spawn_server", lambda b: pytest.fail("spawned"))

    assert studio_mod._server_base() == BASE


def test_nothing_listening_spawns_at_the_configured_address(monkeypatch) -> None:
    _no_live(monkeypatch)
    _ready_script(
        monkeypatch, [httpx.ConnectError("refused"), httpx.ConnectError("x"), True]
    )
    proc = FakeProc()
    spawned: list[str] = []
    monkeypatch.setattr(
        studio_mod, "_spawn_server", lambda b: spawned.append(b) or proc
    )
    registered: list = []
    monkeypatch.setattr(
        studio_mod.atexit, "register", lambda f, *a: registered.append((f, a))
    )

    assert studio_mod._server_base() == BASE
    assert spawned == [BASE]
    assert registered == [(studio_mod.terminate_child, (proc, 10))]


def test_listening_tells_a_refused_connect_from_any_answer(monkeypatch) -> None:
    _ready_script(monkeypatch, [httpx.ConnectError("refused")])
    assert studio_mod._listening(BASE) is False

    _ready_script(
        monkeypatch, [httpx.HTTPStatusError("503", request=None, response=None)]
    )
    assert studio_mod._listening(BASE) is True  # slow or unhappy, but there


def test_child_exiting_early_is_fatal(monkeypatch) -> None:
    _ready_script(monkeypatch, [httpx.ConnectError("refused")])

    with pytest.raises(typer.Exit):
        studio_mod._wait_ready(BASE, FakeProc(exit_after=1), timeout=5)


def test_wait_gives_up_at_the_deadline_without_failing(monkeypatch) -> None:
    _ready_script(monkeypatch, [False])
    clock = iter([0.0, 0.1, 0.2, 9.0])
    monkeypatch.setattr(studio_mod.time, "monotonic", lambda: next(clock))

    assert studio_mod._wait_ready(BASE, None, timeout=1.0) is False


def test_spawn_targets_the_probed_address(monkeypatch) -> None:
    cmds: list[list[str]] = []
    monkeypatch.setattr(
        studio_mod.subprocess, "Popen", lambda cmd: cmds.append(cmd) or FakeProc()
    )

    studio_mod._spawn_server("http://127.0.0.1:9050")

    (cmd,) = cmds
    assert cmd[1:] == [
        "-m",
        "physiclaw.cli",
        "mcp",
        "-H",
        "--host",
        "127.0.0.1",
        "--port",
        "9050",
    ]
