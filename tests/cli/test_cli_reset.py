"""Tests for `physiclaw.cli.reset` — clearing calibration + the learned layout.

The autouse `physiclaw_home` fixture (tests/conftest.py) re-points
`paths.HOME` at a fresh tmp dir per test, so `calibration_dir()` and
`screen_layout_dir()` live under it and deletion is isolated.
"""

from __future__ import annotations

import typer
from typer.testing import CliRunner

from physiclaw.common import paths
from physiclaw.cli.reset import reset

runner = CliRunner()


def _make_app() -> typer.Typer:
    app = typer.Typer()
    app.command()(reset)
    return app


def _seed_layout() -> None:
    d = paths.screen_layout_dir()
    d.mkdir(parents=True, exist_ok=True)
    (d / "layout.json").write_text("{}")
    (d / "layout.md").write_text("card")


def _seed_calibration() -> None:
    d = paths.calibration_dir()
    (d / "cache").mkdir(parents=True, exist_ok=True)
    paths.calibration_bundle().write_text("{}")
    (d / "cache" / "snap.jpg").write_text("x")


def test_reset_noop_when_nothing_present() -> None:
    result = runner.invoke(_make_app(), [])

    assert result.exit_code == 0
    assert "Nothing to reset" in result.stdout
    assert not paths.screen_layout_dir().exists()
    assert not paths.calibration_dir().exists()


def test_reset_clears_layout_and_calibration_with_yes() -> None:
    _seed_layout()
    _seed_calibration()

    result = runner.invoke(_make_app(), ["--yes"])

    assert result.exit_code == 0
    assert "cleared" in result.stdout.lower()
    assert not paths.screen_layout_dir().exists()
    assert not paths.calibration_dir().exists()


def test_reset_clears_calibration_even_without_layout() -> None:
    _seed_calibration()

    result = runner.invoke(_make_app(), ["--yes"])

    assert result.exit_code == 0
    assert not paths.calibration_dir().exists()


def test_reset_cancelled_keeps_everything() -> None:
    _seed_layout()
    _seed_calibration()

    result = runner.invoke(_make_app(), [], input="n\n")

    assert result.exit_code == 1
    assert "Cancelled" in result.stdout
    assert paths.screen_layout_json().exists()  # still there
    assert paths.calibration_bundle().exists()  # still there
