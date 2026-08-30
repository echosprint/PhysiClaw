"""Tests for `physiclaw playbooks stats` — the runs.jsonl aggregate view
(the escalation-rate KPI surface), over the per-test `~/.physiclaw`."""

from __future__ import annotations

import typer
from typer.testing import CliRunner

from physiclaw.cli.playbooks import playbooks_app
from physiclaw.conductor import walklog

app = typer.Typer()
app.add_typer(playbooks_app, name="playbooks")
runner = CliRunner()


def _record(**overrides) -> None:
    base = dict(
        app="demo",
        playbook="flow",
        outcome="handover",
        idx=1,
        nodes=2,
        node="search",
        reason="leg did not land",
        micros=1,
    )
    walklog.record(**{**base, **overrides})


def test_stats_with_no_runs_says_so() -> None:
    result = runner.invoke(app, ["playbooks", "stats"])

    assert result.exit_code == 0
    assert "No walks recorded yet" in result.output


def test_stats_aggregates_and_names_hot_nodes() -> None:
    _record()
    _record(outcome="completed", node=None, reason="", micros=0)

    result = runner.invoke(app, ["playbooks", "stats"])

    assert result.exit_code == 0
    assert "demo/flow: runs=2 completed=1" in result.output
    assert "escalation=50%" in result.output
    assert "✗ search ×1  leg did not land" in result.output


def test_stats_lists_recent_rows() -> None:
    _record(outcome="completed", node=None, reason="")

    result = runner.invoke(app, ["playbooks", "stats"])

    assert "recent:" in result.output
    assert "demo/flow completed at (end)" in result.output


def test_stats_last_zero_hides_recent_rows() -> None:
    _record()

    result = runner.invoke(app, ["playbooks", "stats", "--last", "0"])

    assert "recent:" not in result.output
