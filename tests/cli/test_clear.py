"""Tests for `physiclaw.cli.clear` — deleting saved debug images.

All paths derive from `paths.HOME`, which the autouse `physiclaw_home`
fixture repoints to a per-test tmp dir — no real `~/.physiclaw` is touched.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest
import typer
from typer.testing import CliRunner

from physiclaw.common import paths

clear_mod = importlib.import_module("physiclaw.cli.clear")

app = typer.Typer()
app.command()(clear_mod.clear)
runner = CliRunner()


def _seed(d: Path, n: int = 1) -> None:
    d.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        (d / f"img_{i}.jpg").write_bytes(b"\xff" * 100)


def _seed_all() -> list[Path]:
    dirs = [
        paths.snapshots_dir(),
        paths.screenshots_dir(),
        paths.tool_calls_dir(),
        paths.raw_camera_dir(),
    ]
    for d in dirs:
        _seed(d)
    return dirs


# ---------- empty states ----------


def test_clear_with_no_dirs_reports_nothing_to_clear() -> None:
    result = runner.invoke(app, [])

    assert result.exit_code == 0
    assert "No saved images to clear" in result.output


def test_clear_with_empty_dirs_reports_nothing_to_clear() -> None:
    paths.snapshots_dir().mkdir(parents=True)
    paths.raw_camera_dir().mkdir(parents=True)

    result = runner.invoke(app, [])

    assert result.exit_code == 0
    assert "No saved images to clear" in result.output


# ---------- what gets removed ----------


@pytest.mark.parametrize("flag", ["--yes", "-y"])
def test_clear_yes_removes_all_image_dirs(flag: str) -> None:
    dirs = _seed_all()

    result = runner.invoke(app, [flag])

    assert result.exit_code == 0
    assert "Cleared 4 file(s)." in result.output
    assert all(not d.exists() for d in dirs)


def test_clear_removes_only_populated_dirs() -> None:
    _seed(paths.snapshots_dir(), n=2)
    paths.screenshots_dir().mkdir(parents=True)  # present but empty

    result = runner.invoke(app, ["--yes"])

    assert result.exit_code == 0
    assert not paths.snapshots_dir().exists()
    assert paths.screenshots_dir().exists()


def test_clear_leaves_unrelated_home_dirs() -> None:
    _seed_all()
    calibration = paths.calibration_bundle()
    calibration.parent.mkdir(parents=True)
    calibration.write_text("{}")

    result = runner.invoke(app, ["--yes"])

    assert result.exit_code == 0
    assert calibration.exists()


def test_clear_lists_found_dirs_with_counts() -> None:
    _seed(paths.snapshots_dir(), n=2)

    result = runner.invoke(app, ["--yes"])

    assert "Found 2 file(s)" in result.output
    assert "MB" in result.output
    assert str(paths.snapshots_dir()) in result.output


# ---------- confirmation prompt ----------


def test_clear_confirmation_accepted_deletes() -> None:
    dirs = _seed_all()

    result = runner.invoke(app, [], input="y\n")

    assert result.exit_code == 0
    assert "Cleared 4 file(s)." in result.output
    assert all(not d.exists() for d in dirs)


def test_clear_confirmation_declined_cancels() -> None:
    dirs = _seed_all()

    result = runner.invoke(app, [], input="n\n")

    assert result.exit_code == 1
    assert "Cancelled." in result.output
    assert all(d.exists() for d in dirs)


def test_clear_confirmation_default_is_decline() -> None:
    dirs = _seed_all()

    result = runner.invoke(app, [], input="\n")

    assert result.exit_code == 1
    assert all(d.exists() for d in dirs)
