"""Tests for `physiclaw.conductor.replay` (behind `playbooks replay`) —
the real walk over recorded screens, writing nothing."""

from __future__ import annotations

from conductor_fakes import (
    DONE,
    ELSEWHERE,
    FLOW,
    HOME,
    RESULTS,
    build_program,
    make_screen,
    write_channel,
    write_pack,
)
from typer.testing import CliRunner

from physiclaw.cli.playbooks import playbooks_app
from physiclaw.conductor import replay, suspension, walklog

AGENT_FLOW = """\
description: parse then walk
inputs:
  keyword:
    description: what to search
route:
  - agent: parse
    prompt: "Turn this into a search term: {inputs.keyword}"
    returns:
      term: the search term
  - page: home
  - do: open
    macro: open-app
    with: {message: "{parse.term}"}
  - page: results
"""

TELLING = """\
description: tell then stop
inputs:
  keyword:
    description: what
route:
  - page: home
  - tell: note
    message: "starting {inputs.keyword}"
  - page: home
  - do: open
    macro: open-app
    with: {message: "{inputs.keyword}"}
  - page: results
"""


def _dry(name: str = "flow", **values):
    return build_program(name=name, dry=True, **values)


def test_replay_walks_the_route_to_completion_and_writes_nothing() -> None:
    write_pack(playbooks={"flow": FLOW})

    result = replay.replay(_dry(keyword="milk"), [HOME, RESULTS, DONE])

    assert result.outcome == "completed"
    assert [t.tool for t in result.turns] == ["peek", "run_macro", "run_macro", "peek"]
    assert [t.node for t in result.turns] == ["open", "open", "search", None]
    assert result.turns[1].verdict == "match demo.home"
    assert not walklog.runs_file().exists()


def test_replay_reports_where_the_walk_hands_over() -> None:
    write_pack(playbooks={"flow": FLOW})

    result = replay.replay(_dry(keyword="milk"), [HOME, ELSEWHERE])

    assert result.outcome == "handover"
    assert "did not land on 'results'" in result.detail
    assert result.turns[-1].node == "open"


def test_replay_stops_when_screens_run_out() -> None:
    write_pack(playbooks={"flow": FLOW})

    result = replay.replay(_dry(keyword="milk"), [HOME])

    assert result.outcome == "stopped" and "screens exhausted" in result.detail


def test_replay_answers_a_text_agent_from_outputs_or_stops() -> None:
    write_pack(playbooks={"flow": AGENT_FLOW})

    stopped = replay.replay(_dry(keyword="milk"), [HOME, RESULTS])
    assert stopped.outcome == "stopped" and "agent 'parse'" in stopped.detail

    answered = replay.replay(
        _dry(keyword="milk"), [HOME, RESULTS], {"parse.term": "五常大米"}
    )
    assert answered.outcome == "completed"
    assert answered.turns[1].tool == "run_macro"


def test_replay_suspension_leaves_no_file() -> None:
    write_channel()
    write_pack(playbooks={"flow": TELLING})
    thread = make_screen(("MyChat", 0.5, 0.05), ("starting milk", 0.75, 0.3)).text

    result = replay.replay(_dry(keyword="milk"), [HOME, thread])

    assert result.outcome == "suspended"
    assert result.turns[-1].tool == "end_session"
    assert not suspension.suspended_path().exists()


def test_replay_cli_reads_listing_files(tmp_path) -> None:
    write_pack(playbooks={"flow": FLOW})
    files = []
    for i, text in enumerate((HOME, RESULTS, DONE)):
        f = tmp_path / f"s{i}.txt"
        f.write_text(text, encoding="utf-8")
        files.append(str(f))
    args = ["replay", "demo/flow", "-i", "keyword=milk"]
    for f in files:
        args += ["--listing", f]

    out = CliRunner().invoke(playbooks_app, args)

    assert out.exit_code == 0, out.output
    assert "completed:" in out.output and "run_macro" in out.output
