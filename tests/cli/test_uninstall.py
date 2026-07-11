"""Tests for `physiclaw.cli.uninstall` — removal of user data.

The `physiclaw_home` autouse fixture (in `tests/conftest.py`) makes
``paths.HOME`` a fresh tmp dir per test. We populate it with placeholder
files to verify deletion vs. preservation.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import typer
from typer.testing import CliRunner

from physiclaw import config as _config
from physiclaw.cli.uninstall import uninstall


def _make_app() -> typer.Typer:
    app = typer.Typer()
    app.command()(uninstall)
    return app


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture
def populated_home(physiclaw_home: Path) -> Path:
    """Drop a few placeholder files under HOME so deletion has something to remove."""
    (physiclaw_home / "config.toml").write_text("# fake config\n")
    (physiclaw_home / "calibration").mkdir()
    (physiclaw_home / "calibration" / "bundle.json").write_text("{}")
    (physiclaw_home / "models").mkdir()
    (physiclaw_home / "models" / "marker").write_text("x")
    return physiclaw_home


# ---------- --data ----------


def test_uninstall_data_yes_removes_home(
    populated_home: Path, runner: CliRunner
) -> None:
    result = runner.invoke(_make_app(), ["--data", "--yes"])

    assert result.exit_code == 0, result.stdout
    assert not populated_home.exists()
    assert "Removed" in result.stdout
    assert "uv tool uninstall physiclaw" in result.stdout


def test_uninstall_all_is_alias_for_data(
    populated_home: Path, runner: CliRunner
) -> None:
    result = runner.invoke(_make_app(), ["--all", "--yes"])

    assert result.exit_code == 0
    assert not populated_home.exists()


def test_uninstall_data_prompts_when_no_yes(
    populated_home: Path,
    runner: CliRunner,
) -> None:
    result = runner.invoke(_make_app(), ["--data"], input="y\n")

    assert result.exit_code == 0
    assert not populated_home.exists()


def test_uninstall_data_aborts_on_no(populated_home: Path, runner: CliRunner) -> None:
    result = runner.invoke(_make_app(), ["--data"], input="n\n")

    assert result.exit_code == 1
    assert populated_home.exists()
    assert "Cancelled" in result.stdout


# ---------- --config ----------


def test_uninstall_config_removes_only_config(
    populated_home: Path,
    runner: CliRunner,
) -> None:
    result = runner.invoke(_make_app(), ["--config", "--yes"])

    assert result.exit_code == 0
    assert not (populated_home / "config.toml").exists()
    # Other data is preserved.
    assert (populated_home / "calibration" / "bundle.json").exists()
    assert (populated_home / "models" / "marker").exists()


def test_uninstall_config_when_missing_skips_gracefully(
    physiclaw_home: Path,
    runner: CliRunner,
) -> None:
    # No config.toml created.
    result = runner.invoke(_make_app(), ["--config", "--yes"])

    assert result.exit_code == 0
    assert "does not exist" in result.stdout


# ---------- --dry-run ----------


def test_uninstall_dry_run_does_not_delete(
    populated_home: Path,
    runner: CliRunner,
) -> None:
    result = runner.invoke(_make_app(), ["--data", "--dry-run"])

    assert result.exit_code == 0
    assert populated_home.exists()
    assert (populated_home / "config.toml").exists()
    assert "[dry-run]" in result.stdout
    assert "would remove" in result.stdout


def test_uninstall_dry_run_with_config_does_not_delete(
    populated_home: Path,
    runner: CliRunner,
) -> None:
    cfg = populated_home / "config.toml"
    result = runner.invoke(_make_app(), ["--config", "--dry-run"])

    assert result.exit_code == 0
    assert cfg.exists()
    assert "[dry-run]" in result.stdout


# ---------- interactive (no flags) ----------


def test_uninstall_interactive_yes_removes_home(
    populated_home: Path,
    runner: CliRunner,
) -> None:
    # First prompt: "Remove everything?" → y; Second: "Remove all PhysiClaw data?" → y
    result = runner.invoke(_make_app(), [], input="y\ny\n")

    assert result.exit_code == 0
    assert not populated_home.exists()


def test_uninstall_interactive_first_no_keeps_data(
    populated_home: Path,
    runner: CliRunner,
) -> None:
    result = runner.invoke(_make_app(), [], input="n\n")

    assert result.exit_code == 0
    assert populated_home.exists()
    # No "Cancelled" — user explicitly chose nothing to remove.
    assert "uv tool uninstall physiclaw" in result.stdout


def test_uninstall_interactive_no_home_skips_gracefully(
    physiclaw_home: Path,
    runner: CliRunner,
) -> None:
    # Remove the auto-created home dir to simulate a fresh box.
    physiclaw_home.rmdir()
    assert not physiclaw_home.exists()

    result = runner.invoke(_make_app(), [])

    assert result.exit_code == 0
    assert "No PhysiClaw data found" in result.stdout
    assert "uv tool uninstall physiclaw" in result.stdout


# ---------- always-printed final reminder ----------


def test_final_reminder_prints_even_when_nothing_done(
    populated_home: Path,
    runner: CliRunner,
) -> None:
    # User declines the interactive prompt — we still want the manual reminder.
    result = runner.invoke(_make_app(), [], input="n\n")

    assert "uv tool uninstall physiclaw" in result.stdout


def test_final_reminder_prints_after_dry_run(
    populated_home: Path,
    runner: CliRunner,
) -> None:
    result = runner.invoke(_make_app(), ["--data", "--dry-run"])

    assert "uv tool uninstall physiclaw" in result.stdout


# ---------- --data + --config (--data wins) ----------


def test_data_wins_over_config_when_both_passed(
    populated_home: Path,
    runner: CliRunner,
) -> None:
    result = runner.invoke(_make_app(), ["--data", "--config", "--yes"])

    assert result.exit_code == 0
    assert not populated_home.exists()  # full tree gone, not just config


# ---------- config_path resolution ----------


def test_uninstall_config_uses_config_module_path(
    populated_home: Path,
    runner: CliRunner,
) -> None:
    expected = _config.config_path()
    assert expected.exists()  # populated_home seeded it

    result = runner.invoke(_make_app(), ["--config", "--yes"])

    assert result.exit_code == 0
    assert not expected.exists()
