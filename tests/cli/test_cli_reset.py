"""Tests for `physiclaw.cli.reset` — clearing the learned screen layout.

The autouse `physiclaw_home` fixture (tests/conftest.py) re-points
`paths.HOME` at a fresh tmp dir per test, so `screen_layout_dir()` lives
under it and deletion is isolated.
"""
from __future__ import annotations

import typer
from typer.testing import CliRunner

from physiclaw import paths
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


def test_reset_noop_when_nothing_learned() -> None:
    result = runner.invoke(_make_app(), [])

    assert result.exit_code == 0
    assert "No learned screen layout" in result.stdout
    assert not paths.screen_layout_dir().exists()


def test_reset_clears_layout_with_yes() -> None:
    _seed_layout()
    result = runner.invoke(_make_app(), ["--yes"])

    assert result.exit_code == 0
    assert "cleared" in result.stdout.lower()
    assert not paths.screen_layout_dir().exists()


def test_reset_cancelled_keeps_layout() -> None:
    _seed_layout()
    result = runner.invoke(_make_app(), [], input="n\n")

    assert result.exit_code == 1
    assert "Cancelled" in result.stdout
    assert paths.screen_layout_json().exists()  # still there
