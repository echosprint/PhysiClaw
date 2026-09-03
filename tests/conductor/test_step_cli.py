"""Tests for `physiclaw playbooks step` — the playbook debugger: one node
per invocation, the position persisted under debug/, the pack re-read
each time, the user channel virtual, no model fallback.

Beside the conductor tests (like `test_rehearse.py`) because it drives a
real `Program` over the shared pack fixtures in `conductor_fakes`."""

from __future__ import annotations

import json

import pytest
from conductor_fakes import make_screen, write_pack
from typer.testing import CliRunner

from physiclaw.cli import app
from physiclaw.common.text import read_text
from physiclaw.debug import stepping

runner = CliRunner()

HOME = make_screen(("Files", 0.5, 0.1)).text

FLOW = """\
description: two moves
inputs:
  keyword:
    description: what to search
route:
  - page: home
  - do: open
    macro: open-app
    with: {message: "{inputs.keyword}"}
  - page: home
  - do: search
    macro: add-cart
    with: {message: "go"}
  - page: results
"""


class _FakeMcp:
    def __init__(self, *a, **kw):
        self.calls: list[tuple[str, dict]] = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def call_tool(self, name, args=None):
        self.calls.append((name, args or {}))
        return [{"type": "image", "data": "x"}, {"type": "text", "text": HOME}]


@pytest.fixture()
def stepped_pack(mocker):
    write_pack(playbooks={"flow": FLOW}, macros=("open-app", "add-cart"))
    mocker.patch("physiclaw.agent.engine.mcp_tool.McpClient", _FakeMcp)
    mocker.patch(
        "physiclaw.macros.runner.run_and_record",
        new=mocker.AsyncMock(
            return_value=mocker.Mock(
                ok=True,
                blocks=[
                    {"type": "text", "text": "ran"},
                    {"type": "text", "text": HOME},
                ],
            )
        ),
    )


def _step(*args: str):
    return runner.invoke(app, ["playbooks", "step", "demo/flow", *args])


def test_step_runs_one_node_and_stores_the_position(stepped_pack) -> None:
    result = _step("-i", "keyword=milk")

    assert result.exit_code == 0, result.output
    assert "node open (1/2)" in result.output
    assert "screen reads match: demo.home" in result.output
    assert "paused — next node is search (2/2)" in result.output
    state = json.loads(read_text(stepping.state_path()))
    assert state["idx"] == 1 and state["values"] == {"keyword": "milk"}


def test_step_continues_from_the_stored_position_and_keeps_it_on_handover(
    stepped_pack,
) -> None:
    _step("-i", "keyword=milk")

    result = _step()

    assert result.exit_code == 0, result.output
    assert "node search (2/2)" in result.output
    # The macro lands on HOME, not results — no recover declared → handover;
    # the position stays on the node so the author can fix and step again.
    assert "position kept at search (2/2)" in result.output
    assert json.loads(read_text(stepping.state_path()))["idx"] == 1


def test_step_at_jumps_the_cursor_and_status_reports_it(stepped_pack) -> None:
    _step("-i", "keyword=milk")

    jumped = _step("--at", "open")
    status = _step("--status")

    assert jumped.exit_code == 0 and "node open (1/2)" in jumped.output
    assert "at node search (2/2)" in status.output


def test_step_refuses_new_inputs_mid_walk_and_reset_clears(stepped_pack) -> None:
    _step("-i", "keyword=milk")

    refused = _step("-i", "keyword=eggs")
    cleared = _step("--reset")

    assert refused.exit_code == 1 and "reset" in refused.output
    assert "position cleared" in cleared.output
    assert not stepping.state_path().exists()


def test_step_unknown_node_exits_one(stepped_pack) -> None:
    result = _step("-i", "keyword=milk", "--at", "nope")

    assert result.exit_code == 1
    assert "no node 'nope'" in result.output


def test_step_seeds_the_virtual_thread_with_the_user_words(stepped_pack) -> None:
    from physiclaw.debug import thread as vthread

    _step("-i", "keyword=milk", "--reply", "好的")

    thread = vthread.load()
    assert thread.bubbles[0].text == "(stepping demo/flow)"
    assert thread.staged == ["好的"]
