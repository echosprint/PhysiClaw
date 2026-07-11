"""Tests for `physiclaw.cli.setup.__init__` — Typer subapp wiring."""

from __future__ import annotations

import importlib

from typer.testing import CliRunner

setup_mod = importlib.import_module("physiclaw.cli.setup")
setup_app = setup_mod.setup_app

runner = CliRunner()


def test_setup_help_lists_all_subcommands() -> None:
    result = runner.invoke(setup_app, ["--help"])

    assert result.exit_code == 0
    for sub in ("hardware", "local-vision-model", "phone"):
        assert sub in result.stdout
