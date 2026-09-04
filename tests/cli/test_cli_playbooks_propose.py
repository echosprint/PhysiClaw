"""Tests for `physiclaw playbooks propose` — the escalation triage
surface: hot sites rendered into concrete mining commands."""

from __future__ import annotations

import typer
from typer.testing import CliRunner

from physiclaw.cli.playbooks import playbooks_app
from physiclaw.conductor.walk import walklog

app = typer.Typer()
app.add_typer(playbooks_app, name="playbooks")
runner = CliRunner()


def test_propose_with_no_escalations_says_so() -> None:
    result = runner.invoke(app, ["playbooks", "propose"])

    assert result.exit_code == 0
    assert "No escalations recorded" in result.output


def test_propose_renders_the_site_and_the_mining_commands() -> None:
    walklog.record(
        app="taobao",
        playbook="buy",
        outcome="handover",
        idx=2,
        nodes=9,
        node="choose",
        reason="decide exceeded max_visits (3)",
    )

    result = runner.invoke(app, ["playbooks", "propose"])

    assert result.exit_code == 0
    assert "taobao/buy escalates at choose ×1" in result.output
    assert "decide exceeded max_visits" in result.output
    assert "physiclaw playbooks pages match taobao" in result.output
    assert "physiclaw playbooks pages calibrate taobao" in result.output
