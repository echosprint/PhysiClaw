"""Tests for `physiclaw.cli.update` — the self-update command + the
`physiclaw server` auto-update hook.

Everything that touches uv / PyPI / the process table is patched at the
module seams (`_uv`, `_tool_version`, `_run_install`, `_fetch_pypi_version`)
so no test shells out or hits the network. The autouse `physiclaw_home`
fixture isolates the version-check cache per test.
"""
from __future__ import annotations

import importlib
import json
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import typer

up = importlib.import_module("physiclaw.cli.update")


def _proc(returncode: int = 0, stderr: str = "") -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(args=[], returncode=returncode, stdout="", stderr=stderr)


def _cached_version(home: Path) -> str | None:
    p = home / "run" / "version-check.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())["latest_version"]


@pytest.fixture
def updatable(monkeypatch: pytest.MonkeyPatch, mocker) -> MagicMock:
    """Baseline: uv present, tool-managed at 1.0.0, PyPI offers 1.1.0.

    `_tool_version` answers 1.0.0 before the install and 1.1.0 after.
    Yields the `_run_install` mock (succeeds by default).
    """
    monkeypatch.setattr(up, "_pkg_version", "1.0.0")
    mocker.patch.object(up, "_uv", return_value="/usr/bin/uv")
    mocker.patch.object(up, "_tool_version", side_effect=["1.0.0", "1.1.0"])
    mocker.patch.object(up, "_fetch_pypi_version", return_value="1.1.0")
    return mocker.patch.object(up, "_run_install", return_value=_proc(0))


# ---------- `physiclaw update` pre-flight ----------


def test_update_errors_when_uv_missing(mocker, capsys) -> None:
    mocker.patch.object(up, "_uv", return_value=None)

    with pytest.raises(typer.Exit) as e:
        up.update(check=False, version=None)

    assert e.value.exit_code == 1
    assert "uv" in capsys.readouterr().out


def test_update_errors_when_not_tool_managed(mocker, capsys) -> None:
    mocker.patch.object(up, "_uv", return_value="/usr/bin/uv")
    mocker.patch.object(up, "_fetch_pypi_version", return_value="9.9.9")
    mocker.patch.object(up, "_tool_version", return_value=None)
    run = mocker.patch.object(up, "_run_install")

    with pytest.raises(typer.Exit) as e:
        up.update(check=False, version=None)

    assert e.value.exit_code == 1
    assert "uv tool" in capsys.readouterr().out
    run.assert_not_called()


def test_update_errors_when_pypi_unreachable(
    monkeypatch: pytest.MonkeyPatch, mocker, capsys,
) -> None:
    monkeypatch.setattr(up, "_pkg_version", "1.0.0")
    mocker.patch.object(up, "_uv", return_value="/usr/bin/uv")
    mocker.patch.object(up, "_tool_version", return_value="1.0.0")
    mocker.patch.object(up, "_fetch_pypi_version", return_value=None)
    run = mocker.patch.object(up, "_run_install")

    with pytest.raises(typer.Exit) as e:
        up.update(check=False, version=None)

    assert e.value.exit_code == 1
    run.assert_not_called()


# ---------- `physiclaw update` install paths ----------


def test_update_noop_when_up_to_date(
    physiclaw_home: Path, monkeypatch: pytest.MonkeyPatch, mocker, capsys,
) -> None:
    monkeypatch.setattr(up, "_pkg_version", "1.0.0")
    mocker.patch.object(up, "_uv", return_value="/usr/bin/uv")
    mocker.patch.object(up, "_tool_version", return_value="1.0.0")
    mocker.patch.object(up, "_fetch_pypi_version", return_value="1.0.0")
    run = mocker.patch.object(up, "_run_install")

    up.update(check=False, version=None)

    assert "up to date" in capsys.readouterr().out
    run.assert_not_called()
    assert _cached_version(physiclaw_home) == "1.0.0"


def test_update_installs_latest_unpinned(
    physiclaw_home: Path, updatable: MagicMock, capsys,
) -> None:
    up.update(check=False, version=None)

    # Unpinned spec — a later manual `uv tool upgrade` must keep working.
    updatable.assert_called_once_with("/usr/bin/uv", "physiclaw", capture=False)
    out = capsys.readouterr().out
    assert "1.0.0 → 1.1.0" in out
    assert _cached_version(physiclaw_home) == "1.1.0"


def test_update_trusts_installed_state_over_uv_exit_code(
    updatable: MagicMock, capsys,
) -> None:
    # uv reports failure, but `uv tool list` shows the new version — the
    # Windows Defender shim-lock case. Must succeed with a warning.
    updatable.return_value = _proc(1)

    up.update(check=False, version=None)

    out = capsys.readouterr().out
    assert "uv reported a failure" in out
    assert "1.0.0 → 1.1.0" in out


def test_update_fails_when_version_does_not_advance(
    monkeypatch: pytest.MonkeyPatch, mocker, capsys,
) -> None:
    monkeypatch.setattr(up, "_pkg_version", "1.0.0")
    mocker.patch.object(up, "_uv", return_value="/usr/bin/uv")
    mocker.patch.object(up, "_tool_version", side_effect=["1.0.0", "1.0.0"])
    mocker.patch.object(up, "_fetch_pypi_version", return_value="1.1.0")
    mocker.patch.object(up, "_run_install", return_value=_proc(1))

    with pytest.raises(typer.Exit) as e:
        up.update(check=False, version=None)

    assert e.value.exit_code == 1
    assert "--verbose" in capsys.readouterr().out  # manual-retry hint


def test_update_explicit_version_pins(
    monkeypatch: pytest.MonkeyPatch, mocker, capsys,
) -> None:
    monkeypatch.setattr(up, "_pkg_version", "1.0.0")
    mocker.patch.object(up, "_uv", return_value="/usr/bin/uv")
    mocker.patch.object(up, "_tool_version", side_effect=["1.0.0", "0.9.0"])
    mocker.patch.object(up, "_fetch_pypi_version", return_value="1.1.0")
    run = mocker.patch.object(up, "_run_install", return_value=_proc(0))

    up.update(check=False, version="0.9.0")  # downgrade is allowed

    run.assert_called_once_with("/usr/bin/uv", "physiclaw==0.9.0", capture=False)
    out = capsys.readouterr().out
    assert "1.0.0 → 0.9.0" in out
    assert "pinned" in out.lower()


def test_update_explicit_version_noop_when_already_there(
    monkeypatch: pytest.MonkeyPatch, mocker, capsys,
) -> None:
    monkeypatch.setattr(up, "_pkg_version", "1.0.0")
    mocker.patch.object(up, "_uv", return_value="/usr/bin/uv")
    mocker.patch.object(up, "_tool_version", return_value="1.0.0")
    mocker.patch.object(up, "_fetch_pypi_version", return_value="1.1.0")
    run = mocker.patch.object(up, "_run_install")

    up.update(check=False, version="1.0.0")

    assert "already at 1.0.0" in capsys.readouterr().out
    run.assert_not_called()


def test_update_ignores_env_disable(
    monkeypatch: pytest.MonkeyPatch, updatable: MagicMock,
) -> None:
    # The env var silences the banner and auto-update — never an
    # explicitly typed `physiclaw update`.
    monkeypatch.setenv("PHYSICLAW_DISABLE_UPDATE_CHECK", "1")

    up.update(check=False, version=None)

    updatable.assert_called_once()


def test_update_refuses_when_server_running(
    updatable: MagicMock, mocker, capsys,
) -> None:
    # Refuse, don't install: swapping the venv under a live server would
    # mix old and new code in that process via its later lazy imports.
    mocker.patch.object(up.runtime_state, "read_live", return_value={"pid": 4242})

    with pytest.raises(typer.Exit) as e:
        up.update(check=False, version=None)

    assert e.value.exit_code == 1
    updatable.assert_not_called()
    out = capsys.readouterr().out
    assert "4242" in out
    assert "Stop it" in out


# ---------- `physiclaw update --check` ----------


def test_check_exits_1_when_update_available(
    physiclaw_home: Path, monkeypatch: pytest.MonkeyPatch, mocker, capsys,
) -> None:
    monkeypatch.setattr(up, "_pkg_version", "1.0.0")
    mocker.patch.object(up, "_uv", return_value="/usr/bin/uv")
    mocker.patch.object(up, "_fetch_pypi_version", return_value="2.0.0")

    with pytest.raises(typer.Exit) as e:
        up.update(check=True, version=None)

    assert e.value.exit_code == 1
    assert "1.0.0 → 2.0.0" in capsys.readouterr().out
    assert _cached_version(physiclaw_home) == "2.0.0"


def test_check_exits_0_when_up_to_date(
    monkeypatch: pytest.MonkeyPatch, mocker, capsys,
) -> None:
    monkeypatch.setattr(up, "_pkg_version", "1.0.0")
    mocker.patch.object(up, "_uv", return_value="/usr/bin/uv")
    mocker.patch.object(up, "_fetch_pypi_version", return_value="1.0.0")

    up.update(check=True, version=None)  # no Exit raised

    assert "up to date" in capsys.readouterr().out


def test_check_exits_2_when_pypi_unreachable(mocker) -> None:
    mocker.patch.object(up, "_uv", return_value="/usr/bin/uv")
    mocker.patch.object(up, "_fetch_pypi_version", return_value=None)

    with pytest.raises(typer.Exit) as e:
        up.update(check=True, version=None)

    assert e.value.exit_code == 2


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


# ---------- maybe_auto_update ----------


@pytest.fixture
def auto_env(monkeypatch: pytest.MonkeyPatch, mocker):
    """Clean slate for auto-update: kill switches off, os.environ restored
    after the test (the success path mutates it for the re-exec marker)."""
    monkeypatch.delenv("PHYSICLAW_DISABLE_UPDATE_CHECK", raising=False)
    monkeypatch.delenv(up._AUTO_UPDATE_MARKER, raising=False)
    monkeypatch.delenv("CI", raising=False)  # the suite itself may run in CI
    mocker.patch.dict(os.environ)
    monkeypatch.setattr(up.CONFIG.update, "auto", True)


def test_auto_update_skips_when_config_off(
    auto_env, monkeypatch: pytest.MonkeyPatch, mocker,
) -> None:
    monkeypatch.setattr(up.CONFIG.update, "auto", False)
    fetch = mocker.patch.object(up, "_fetch_pypi_version")

    up.maybe_auto_update()

    fetch.assert_not_called()


def test_auto_update_skips_when_env_disabled(
    auto_env, monkeypatch: pytest.MonkeyPatch, mocker,
) -> None:
    monkeypatch.setenv("PHYSICLAW_DISABLE_UPDATE_CHECK", "1")
    fetch = mocker.patch.object(up, "_fetch_pypi_version")

    up.maybe_auto_update()

    fetch.assert_not_called()


def test_auto_update_skips_after_reexec_marker(
    auto_env, monkeypatch: pytest.MonkeyPatch, mocker,
) -> None:
    monkeypatch.setenv(up._AUTO_UPDATE_MARKER, "1")
    fetch = mocker.patch.object(up, "_fetch_pypi_version")

    up.maybe_auto_update()

    fetch.assert_not_called()


def test_auto_update_skips_in_ci(
    auto_env, monkeypatch: pytest.MonkeyPatch, mocker,
) -> None:
    monkeypatch.setenv("CI", "true")
    fetch = mocker.patch.object(up, "_fetch_pypi_version")

    up.maybe_auto_update()

    fetch.assert_not_called()


def test_auto_update_skips_dev_and_pip_installs(auto_env, mocker) -> None:
    mocker.patch.object(up, "_uv", return_value="/usr/bin/uv")
    mocker.patch.object(up.shutil, "which", return_value="/x/physiclaw")
    mocker.patch.object(up, "_tool_version", return_value=None)
    fetch = mocker.patch.object(up, "_fetch_pypi_version")

    up.maybe_auto_update()

    fetch.assert_not_called()


def test_auto_update_skips_when_another_server_is_live(auto_env, mocker) -> None:
    # Installing would swap the venv underneath the running server.
    mocker.patch.object(up.runtime_state, "read_live", return_value={"pid": 1})
    fetch = mocker.patch.object(up, "_fetch_pypi_version")

    up.maybe_auto_update()

    fetch.assert_not_called()


def test_auto_update_skips_when_no_shim_to_hand_off(auto_env, mocker) -> None:
    # No shim on PATH → a post-install hand-off would be impossible, and
    # continuing with a swapped venv would mix versions — so don't install.
    mocker.patch.object(up, "_uv", return_value="/usr/bin/uv")
    mocker.patch.object(up.shutil, "which", return_value=None)
    fetch = mocker.patch.object(up, "_fetch_pypi_version")
    run = mocker.patch.object(up, "_run_install")

    up.maybe_auto_update()

    fetch.assert_not_called()
    run.assert_not_called()


def test_auto_update_noop_when_up_to_date(
    auto_env, physiclaw_home: Path, monkeypatch: pytest.MonkeyPatch, mocker,
) -> None:
    monkeypatch.setattr(up, "_pkg_version", "1.0.0")
    mocker.patch.object(up, "_uv", return_value="/usr/bin/uv")
    mocker.patch.object(up.shutil, "which", return_value="/x/physiclaw")
    mocker.patch.object(up, "_tool_version", return_value="1.0.0")
    mocker.patch.object(up, "_fetch_pypi_version", return_value="1.0.0")
    run = mocker.patch.object(up, "_run_install")

    up.maybe_auto_update()

    run.assert_not_called()
    # Cache still refreshed so doctor/status don't re-hit PyPI.
    assert _cached_version(physiclaw_home) == "1.0.0"


def test_auto_update_success_reexecs_on_posix(
    auto_env, monkeypatch: pytest.MonkeyPatch, mocker, capsys,
) -> None:
    monkeypatch.setattr(up, "_pkg_version", "1.0.0")
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(sys, "argv", ["physiclaw", "server", "--port", "9000"])
    mocker.patch.object(up, "_uv", return_value="/usr/bin/uv")
    mocker.patch.object(up, "_tool_version", side_effect=["1.0.0", "1.1.0"])
    mocker.patch.object(up, "_fetch_pypi_version", return_value="1.1.0")
    mocker.patch.object(up, "_run_install", return_value=_proc(0))
    mocker.patch.object(up.shutil, "which", return_value="/home/u/.local/bin/physiclaw")
    # Real execv never returns; emulate that so the fallback path stays dark.
    execv = mocker.patch.object(up.os, "execv", side_effect=SystemExit)

    with pytest.raises(SystemExit):
        up.maybe_auto_update()

    execv.assert_called_once_with(
        "/home/u/.local/bin/physiclaw",
        ["/home/u/.local/bin/physiclaw", "server", "--port", "9000"],
    )
    assert os.environ[up._AUTO_UPDATE_MARKER] == "1"
    assert "restarting" in capsys.readouterr().out


def test_auto_update_success_hands_off_to_child_on_windows(
    auto_env, monkeypatch: pytest.MonkeyPatch, mocker,
) -> None:
    # Windows can't re-exec (running exes are locked): the old process
    # must become a thin waiter on a child running the new version, so
    # no lazily imported module can mix versions in-process.
    monkeypatch.setattr(up, "_pkg_version", "1.0.0")
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(sys, "argv", ["physiclaw", "server"])
    mocker.patch.object(up, "_uv", return_value="C:/bin/uv.exe")
    mocker.patch.object(up.shutil, "which", return_value="C:/bin/physiclaw.exe")
    mocker.patch.object(up, "_tool_version", side_effect=["1.0.0", "1.1.0"])
    mocker.patch.object(up, "_fetch_pypi_version", return_value="1.1.0")
    mocker.patch.object(up, "_run_install", return_value=_proc(0))
    execv = mocker.patch.object(up.os, "execv")
    child = MagicMock()
    child.wait.return_value = 0
    popen = mocker.patch.object(up.subprocess, "Popen", return_value=child)

    with pytest.raises(typer.Exit) as e:
        up.maybe_auto_update()

    assert e.value.exit_code == 0
    execv.assert_not_called()
    popen.assert_called_once_with(["C:/bin/physiclaw.exe", "server"])
    assert os.environ[up._AUTO_UPDATE_MARKER] == "1"


def test_auto_update_falls_back_to_child_when_execv_fails(
    auto_env, monkeypatch: pytest.MonkeyPatch, mocker,
) -> None:
    monkeypatch.setattr(up, "_pkg_version", "1.0.0")
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(sys, "argv", ["physiclaw", "server"])
    mocker.patch.object(up, "_uv", return_value="/usr/bin/uv")
    mocker.patch.object(up.shutil, "which", return_value="/x/physiclaw")
    mocker.patch.object(up, "_tool_version", side_effect=["1.0.0", "1.1.0"])
    mocker.patch.object(up, "_fetch_pypi_version", return_value="1.1.0")
    mocker.patch.object(up, "_run_install", return_value=_proc(0))
    mocker.patch.object(up.os, "execv", side_effect=OSError("noexec"))
    child = MagicMock()
    child.wait.return_value = 7
    mocker.patch.object(up.subprocess, "Popen", return_value=child)

    with pytest.raises(typer.Exit) as e:
        up.maybe_auto_update()

    assert e.value.exit_code == 7  # child's exit code propagates


def test_auto_update_failure_warns_and_continues(
    auto_env, monkeypatch: pytest.MonkeyPatch, mocker, capsys,
) -> None:
    monkeypatch.setattr(up, "_pkg_version", "1.0.0")
    mocker.patch.object(up, "_uv", return_value="/usr/bin/uv")
    mocker.patch.object(up.shutil, "which", return_value="/x/physiclaw")
    mocker.patch.object(up, "_tool_version", side_effect=["1.0.0", "1.0.0"])
    mocker.patch.object(up, "_fetch_pypi_version", return_value="1.1.0")
    mocker.patch.object(
        up, "_run_install", return_value=_proc(1, stderr="boom: no wheels\n"),
    )
    execv = mocker.patch.object(up.os, "execv")

    up.maybe_auto_update()  # must NOT raise — the server must come up

    execv.assert_not_called()
    out = capsys.readouterr().out
    assert "auto-update failed" in out
    assert "boom: no wheels" in out


def test_auto_update_silent_when_pypi_unreachable(
    auto_env, monkeypatch: pytest.MonkeyPatch, mocker, capsys,
) -> None:
    monkeypatch.setattr(up, "_pkg_version", "1.0.0")
    mocker.patch.object(up, "_uv", return_value="/usr/bin/uv")
    mocker.patch.object(up, "_tool_version", return_value="1.0.0")
    mocker.patch.object(up, "_fetch_pypi_version", return_value=None)
    run = mocker.patch.object(up, "_run_install")

    up.maybe_auto_update()

    run.assert_not_called()
    assert capsys.readouterr().out == ""
