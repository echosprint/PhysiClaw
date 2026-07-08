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
    # Post-install health check: shim present, `--version` runs clean.
    mocker.patch.object(up.shutil, "which", return_value="/x/physiclaw")
    mocker.patch.object(up, "_run", return_value=_proc(0))
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


def test_update_fails_when_installed_env_wont_run(
    updatable: MagicMock, mocker, capsys,
) -> None:
    # uv reports the new version, but a post-install `physiclaw --version`
    # exits non-zero → the env is half-written. Fail with repair guidance
    # instead of leaving a broken install to crash on next launch.
    mocker.patch.object(up, "_run", return_value=_proc(1, stderr="ImportError"))

    with pytest.raises(typer.Exit) as e:
        up.update(check=False, version=None)

    assert e.value.exit_code == 1
    assert "won't run" in capsys.readouterr().out


def test_update_succeeds_when_health_check_cannot_spawn(
    updatable: MagicMock, mocker, capsys,
) -> None:
    # An inconclusive health check (subprocess couldn't run → None) must NOT
    # fail the update — only a definitive non-zero exit does.
    mocker.patch.object(up, "_run", return_value=None)

    up.update(check=False, version=None)

    assert "1.0.0 → 1.1.0" in capsys.readouterr().out


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
    mocker.patch.object(up.shutil, "which", return_value="/x/physiclaw")
    mocker.patch.object(up, "_run", return_value=_proc(0))
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


# ---------- notify_staged_update / maybe_stage_update ----------


@pytest.fixture
def auto_env(monkeypatch: pytest.MonkeyPatch, mocker):
    """Clean slate for the startup update hooks: kill switches off, os.environ
    restored after the test, CONFIG.update.auto on."""
    monkeypatch.delenv("PHYSICLAW_DISABLE_UPDATE_CHECK", raising=False)
    monkeypatch.delenv(up._AUTO_UPDATE_MARKER, raising=False)
    monkeypatch.delenv("CI", raising=False)  # the suite itself may run in CI
    mocker.patch.dict(os.environ)
    monkeypatch.setattr(up.CONFIG.update, "auto", True)


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
    monkeypatch.setattr(up.CONFIG.update, "auto", False)
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


def test_notify_skips_when_marker_set(auto_env, monkeypatch, mocker) -> None:
    monkeypatch.setenv(up._AUTO_UPDATE_MARKER, "1")
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
    # Print a one-line notice pointing at `physiclaw update`; install nothing,
    # raise nothing, and keep the marker so the notice repeats until applied.
    mocker.patch.object(up, "_tool_version", return_value="1.0.0")

    up.notify_staged_update()  # must NOT raise

    out = capsys.readouterr().out
    assert "1.1.0" in out
    assert "physiclaw update" in out
    assert up._read_staged() == "1.1.0"


# --- Phase B: maybe_stage_update (background probe + warm + mark) ---


def test_stage_skips_when_config_off(auto_env, monkeypatch, mocker) -> None:
    monkeypatch.setattr(up.CONFIG.update, "auto", False)
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
