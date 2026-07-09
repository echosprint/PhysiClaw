"""Tests for `physiclaw.cli.update` — the two-phase startup update check
(Phase B stages/warms, Phase A notifies) plus the `_tool_version` parser.

Everything that touches uv / PyPI is patched at the module seams (`_uv`,
`_tool_version`, `_run`, `_fetch_pypi_version`) so no test shells out or hits
the network. The autouse `physiclaw_home` fixture isolates the version-check
cache per test.
"""
from __future__ import annotations

import importlib
import json
import os
import subprocess
from pathlib import Path

import pytest

up = importlib.import_module("physiclaw.cli.update")


def _proc(returncode: int = 0, stderr: str = "") -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(args=[], returncode=returncode, stdout="", stderr=stderr)


def _cached_version(home: Path) -> str | None:
    p = home / "run" / "version-check.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())["latest_version"]


# ---------- `physiclaw update` (print-only signpost) ----------


def test_update_command_points_at_uv(capsys) -> None:
    # We don't self-install; `physiclaw update` just shows the uv command.
    up.update()

    out = capsys.readouterr().out
    assert "uv tool upgrade physiclaw" in out
    assert "stop" in out.lower()  # server must be stopped first


# ---------- _tool_version parsing ----------


def test_tool_version_parses_uv_tool_list(mocker) -> None:
    out = _proc(0)
    out.stdout = "physiclaw v0.1.13\n- physiclaw\nruff v0.15.9\n- ruff\n"
    mocker.patch.object(up.subprocess, "run", return_value=out)

    assert up._tool_version("uv") == "0.1.13"


def test_tool_version_none_when_absent(mocker) -> None:
    out = _proc(0)
    out.stdout = "ruff v0.15.9\n- ruff\n"
    mocker.patch.object(up.subprocess, "run", return_value=out)

    assert up._tool_version("uv") is None


def test_tool_version_none_on_uv_failure(mocker) -> None:
    mocker.patch.object(up.subprocess, "run", return_value=_proc(2))
    assert up._tool_version("uv") is None

    mocker.patch.object(up.subprocess, "run", side_effect=OSError("gone"))
    assert up._tool_version("uv") is None


# ---------- notify_staged_update / maybe_stage_update ----------


@pytest.fixture
def auto_env(monkeypatch: pytest.MonkeyPatch, mocker):
    """Clean slate for the startup update hooks: kill switches off, os.environ
    restored after the test, CONFIG.update.check on."""
    monkeypatch.delenv("PHYSICLAW_DISABLE_UPDATE_CHECK", raising=False)
    monkeypatch.delenv("CI", raising=False)  # the suite itself may run in CI
    mocker.patch.dict(os.environ)
    monkeypatch.setattr(up.CONFIG.update, "check", True)


# --- Phase A: notify_staged_update (notice only — never installs) ---


@pytest.fixture
def staged_ready(auto_env, monkeypatch, mocker):
    """A valid stage-and-notify setup: current 1.0.0, a 1.1.0 marker, uv present.
    Tests override `_tool_version` for their specific branch."""
    monkeypatch.setattr(up, "_pkg_version", "1.0.0")
    up._write_staged("1.1.0")
    mocker.patch.object(up, "_uv", return_value="/usr/bin/uv")


def test_notify_noop_when_nothing_staged(auto_env, mocker) -> None:
    # No marker → return before touching uv at all.
    run = mocker.patch.object(up, "_run")

    up.notify_staged_update()

    run.assert_not_called()


def test_notify_skips_when_config_off(auto_env, monkeypatch, mocker) -> None:
    monkeypatch.setattr(up.CONFIG.update, "check", False)
    up._write_staged("1.1.0")
    run = mocker.patch.object(up, "_run")

    up.notify_staged_update()

    run.assert_not_called()


def test_notify_skips_when_env_disabled(auto_env, monkeypatch, mocker) -> None:
    monkeypatch.setenv("PHYSICLAW_DISABLE_UPDATE_CHECK", "1")
    up._write_staged("1.1.0")
    run = mocker.patch.object(up, "_run")

    up.notify_staged_update()

    run.assert_not_called()


def test_notify_skips_in_ci(auto_env, monkeypatch, mocker) -> None:
    monkeypatch.setenv("CI", "true")
    up._write_staged("1.1.0")
    run = mocker.patch.object(up, "_run")

    up.notify_staged_update()

    run.assert_not_called()


def test_notify_clears_stale_marker_when_not_newer(
    auto_env, monkeypatch, mocker,
) -> None:
    # Staged version is not newer than current (we already updated past it).
    monkeypatch.setattr(up, "_pkg_version", "1.1.0")
    up._write_staged("1.1.0")
    uv = mocker.patch.object(up, "_uv")

    up.notify_staged_update()

    assert up._read_staged() is None  # stale marker cleared
    uv.assert_not_called()            # returned before the uv check


def test_notify_stays_silent_for_dev_and_pip_installs(
    staged_ready, mocker, capsys,
) -> None:
    # Not a uv-tool install → nothing to point the user at; say nothing, and
    # leave the marker (a real uv-tool run elsewhere may still want it).
    mocker.patch.object(up, "_tool_version", return_value=None)

    up.notify_staged_update()

    assert capsys.readouterr().out == ""
    assert up._read_staged() == "1.1.0"


def test_notify_prints_ready_notice_and_keeps_marker(
    staged_ready, mocker, capsys,
) -> None:
    # The real case: a newer version is staged and we're a uv-tool install.
    # Print a one-line notice pointing at `uv tool upgrade physiclaw`; install
    # nothing, raise nothing, and keep the marker so it repeats until applied.
    mocker.patch.object(up, "_tool_version", return_value="1.0.0")

    up.notify_staged_update()  # must NOT raise

    out = capsys.readouterr().out
    assert "1.1.0" in out
    assert "uv tool upgrade physiclaw" in out
    assert up._read_staged() == "1.1.0"


# --- Phase B: maybe_stage_update (background probe + warm + mark) ---


def test_stage_skips_when_config_off(auto_env, monkeypatch, mocker) -> None:
    monkeypatch.setattr(up.CONFIG.update, "check", False)
    fetch = mocker.patch.object(up, "_fetch_pypi_version")

    up.maybe_stage_update()

    fetch.assert_not_called()


def test_stage_skips_when_env_disabled(auto_env, monkeypatch, mocker) -> None:
    monkeypatch.setenv("PHYSICLAW_DISABLE_UPDATE_CHECK", "1")
    fetch = mocker.patch.object(up, "_fetch_pypi_version")

    up.maybe_stage_update()

    fetch.assert_not_called()


def test_stage_skips_in_ci(auto_env, monkeypatch, mocker) -> None:
    monkeypatch.setenv("CI", "1")
    fetch = mocker.patch.object(up, "_fetch_pypi_version")

    up.maybe_stage_update()

    fetch.assert_not_called()


def test_stage_skips_dev_and_pip_installs(auto_env, mocker) -> None:
    mocker.patch.object(up, "_uv", return_value="/usr/bin/uv")
    mocker.patch.object(up, "_tool_version", return_value=None)
    fetch = mocker.patch.object(up, "_fetch_pypi_version")

    up.maybe_stage_update()

    fetch.assert_not_called()


def test_stage_up_to_date_clears_marker(
    auto_env, physiclaw_home: Path, monkeypatch, mocker,
) -> None:
    monkeypatch.setattr(up, "_pkg_version", "1.0.0")
    up._write_staged("0.9.0")  # a stale marker
    mocker.patch.object(up, "_uv", return_value="/usr/bin/uv")
    mocker.patch.object(up, "_tool_version", return_value="1.0.0")
    mocker.patch.object(up, "_fetch_pypi_version", return_value="1.0.0")
    run = mocker.patch.object(up, "_run")

    up.maybe_stage_update()

    run.assert_not_called()                       # no warm
    assert up._read_staged() is None              # stale marker cleared
    assert _cached_version(physiclaw_home) == "1.0.0"


def test_stage_already_staged_skips_warm(auto_env, monkeypatch, mocker) -> None:
    monkeypatch.setattr(up, "_pkg_version", "1.0.0")
    up._write_staged("1.1.0")
    mocker.patch.object(up, "_uv", return_value="/usr/bin/uv")
    mocker.patch.object(up, "_tool_version", return_value="1.0.0")
    mocker.patch.object(up, "_fetch_pypi_version", return_value="1.1.0")
    run = mocker.patch.object(up, "_run")

    up.maybe_stage_update()

    run.assert_not_called()  # already staged this version — don't re-download


def test_stage_warms_cache_and_writes_marker(auto_env, monkeypatch, mocker) -> None:
    monkeypatch.setattr(up, "_pkg_version", "1.0.0")
    mocker.patch.object(up, "_uv", return_value="/usr/bin/uv")
    mocker.patch.object(up, "_tool_version", return_value="1.0.0")
    mocker.patch.object(up, "_fetch_pypi_version", return_value="1.1.0")
    run = mocker.patch.object(up, "_run", return_value=_proc(0))

    up.maybe_stage_update()

    cmd = run.call_args.args[0]
    assert "run" in cmd and "physiclaw==1.1.0" in cmd  # the warm command
    assert up._read_staged() == "1.1.0"


def test_stage_warm_failure_leaves_no_marker(auto_env, monkeypatch, mocker) -> None:
    monkeypatch.setattr(up, "_pkg_version", "1.0.0")
    mocker.patch.object(up, "_uv", return_value="/usr/bin/uv")
    mocker.patch.object(up, "_tool_version", return_value="1.0.0")
    mocker.patch.object(up, "_fetch_pypi_version", return_value="1.1.0")
    mocker.patch.object(up, "_run", return_value=_proc(1))

    up.maybe_stage_update()

    assert up._read_staged() is None  # warm failed → don't mark
