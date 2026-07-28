"""Tests for `physiclaw.cli.__init__` — top-level Typer app wiring."""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

import physiclaw
from physiclaw.cli import app

runner = CliRunner()


# ---------- root commands wired ----------


@pytest.mark.parametrize(
    "subcommand",
    [
        "doctor",
        "server",
        "now",
        "auto",
        "mcp",
        "status",
        "setup",
        "config",
        "models",
        "skills",
    ],
)
def test_top_level_help_lists_subcommand(subcommand: str) -> None:
    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    assert subcommand in result.stdout


# ---------- --version ----------


def test_version_flag_prints_pkg_version() -> None:
    result = runner.invoke(app, ["--version"])

    assert result.exit_code == 0
    assert physiclaw.__version__ in result.stdout


def test_help_header_includes_pkg_version() -> None:
    """``physiclaw --help`` should show the version in the description line —
    matches the uv / cargo / npm convention so users can sanity-check what
    they have installed."""
    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    assert physiclaw.__version__ in result.stdout


# ---------- bare invocation defaults to `server` ----------


def test_no_subcommand_runs_server(mocker) -> None:
    """Bare ``physiclaw`` (no subcommand) defaults to ``physiclaw server``."""
    mock_server = mocker.patch("physiclaw.cli.server")

    result = runner.invoke(app, [])

    assert result.exit_code == 0
    mock_server.assert_called_once_with()
