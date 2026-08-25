"""Tests for `physiclaw.agent.conductor.program` — arming (validation +
the arm file), fail-open loading, and the walk: synthesized turns,
fingerprint checks, resume-by-locate, and hand-over on everything this
phase refuses to guess about."""

from __future__ import annotations

import json

import pytest
from conductor_fakes import PACK_MACRO, make_screen, write_pack

from physiclaw.agent.conductor import program
from physiclaw.agent.conductor.playbook import PlaybookError
from physiclaw.agent.engine.dto import (
    AssistantMessage,
    Message,
    SystemMessage,
    ToolResultMessage,
    UserMessage,
)
from physiclaw.common import paths

FLOW = """\
name: flow
description: two legs
inputs:
  keyword:
    description: what to search
nodes:
  - id: open
    type: LEG
    macro: open-app
    with: {message: "{keyword}"}
    verify: home
  - id: search
    type: LEG
    macro: add-cart
    with: {message: "go"}
    enter: home
    verify: results
"""

# Same legs, then a DECIDE — the P5 walk must hand over at it.
BRANCH = (
    FLOW.replace("name: flow", "name: branch")
    + """\
  - id: choose
    type: DECIDE
    call: choose_item
    with: {criteria: "cheapest"}
    on: {pick: escalate, scroll: escalate, none_fit: escalate, escalate: escalate}
"""
)

HOME = make_screen(("Files", 0.5, 0.1)).text
RESULTS = make_screen(("综合", 0.5, 0.1)).text
ELSEWHERE = make_screen(("Nothing known", 0.5, 0.5)).text


def _feed(
    history: list[Message],
    turn: AssistantMessage,
    text: str = "",
    *,
    error: bool = False,
) -> None:
    """Append the synthesized turn plus its action's tool result — the
    loop's contract (one result per call, in the very next messages)."""
    history.append(turn)
    history.append(
        ToolResultMessage(
            tool_call_id=turn.tool_calls[1].id, content=text, is_error=error
        )
    )


def _armed_program() -> program.Program:
    p = program.load_armed()
    assert p is not None
    return p


def _history() -> list[Message]:
    return [SystemMessage(content="sys"), UserMessage(content="wake")]


# ---------- arming ----------


def test_arm_writes_file_and_load_armed_builds_program() -> None:
    write_pack(playbooks={"flow": FLOW})

    spec = program.arm("demo", "flow", {"keyword": "milk"})

    assert len(spec.nodes) == 2
    assert program.armed_ref() == ("demo", "flow")
    p = _armed_program()
    assert p.app == "demo"
    assert p.values == {"keyword": "milk"}
    assert set(p.pack_macros) == {"demo/open-app", "demo/add-cart"}


def test_disarm_removes_the_file() -> None:
    write_pack(playbooks={"flow": FLOW})
    program.arm("demo", "flow", {"keyword": "milk"})

    assert program.disarm() is True
    assert program.disarm() is False
    assert program.armed_ref() is None
    assert program.load_armed() is None


@pytest.mark.parametrize(
    "app, name, inputs, fragment",
    [
        ("demo", "ghost", {}, "no playbook"),
        ("demo", "flow", {}, "missing required input"),
        ("demo", "flow", {"keyword": "x", "typo": "y"}, "unknown input"),
    ],
)
def test_arm_rejections(app, name, inputs, fragment) -> None:
    write_pack(playbooks={"flow": FLOW})

    with pytest.raises(PlaybookError, match=fragment):
        program.arm(app, name, inputs)


def test_arm_refuses_a_disabled_playbook() -> None:
    write_pack(playbooks={"flow": FLOW + "enabled: false\n"})

    with pytest.raises(PlaybookError, match="disabled"):
        program.arm("demo", "flow", {"keyword": "milk"})


def test_arm_refuses_disabled_leg_macros() -> None:
    write_pack(playbooks={"flow": FLOW})
    macro = paths.playbooks_dir() / "demo" / "macros" / "open-app" / "MACRO.yml"
    macro.write_text(
        PACK_MACRO.format(name="open-app") + "enabled: false\n", encoding="utf-8"
    )

    with pytest.raises(PlaybookError, match="disabled pack macro"):
        program.arm("demo", "flow", {"keyword": "milk"})


def test_load_armed_fail_open() -> None:
    # No file at all.
    assert program.load_armed() is None
    # Corrupt file.
    paths.playbooks_dir().mkdir(parents=True, exist_ok=True)
    (paths.playbooks_dir() / program.ARMED_FILENAME).write_text(
        "{not json", encoding="utf-8"
    )
    assert program.load_armed() is None
    # A file naming a playbook that no longer exists.
    write_pack(playbooks={"flow": FLOW})
    (paths.playbooks_dir() / program.ARMED_FILENAME).write_text(
        json.dumps({"schema": 1, "app": "demo", "playbook": "gone", "inputs": {}}),
        encoding="utf-8",
    )
    assert program.load_armed() is None


# ---------- the walk ----------


def test_walk_runs_both_legs_then_completes() -> None:
    write_pack(playbooks={"flow": FLOW})
    program.arm("demo", "flow", {"keyword": "milk"})
    p = _armed_program()
    h = _history()

    peek = p.advance(h)
    assert peek is not None and peek.synthesized
    assert peek.tool_names() == ["note", "peek"]

    _feed(h, peek, ELSEWHERE)  # unknown page → start from the top
    leg1 = p.advance(h)
    assert leg1 is not None and leg1.tool_names() == ["note", "run_macro"]
    assert leg1.tool_calls[1].arguments == {
        "name": "demo/open-app",
        "inputs": {"message": "milk"},  # {keyword} resolved from the arm
    }

    _feed(h, leg1, HOME)  # verify: home holds → next leg (enter: home holds too)
    leg2 = p.advance(h)
    assert leg2 is not None
    assert leg2.tool_calls[1].arguments["name"] == "demo/add-cart"

    _feed(h, leg2, RESULTS)  # verify: results holds → playbook complete
    assert p.advance(h) is None


def test_locate_resumes_past_completed_legs() -> None:
    # A killed session's next wake: the screen already shows leg 1's
    # verify page, so the walk resumes at leg 2 — no repeated gestures.
    write_pack(playbooks={"flow": FLOW})
    program.arm("demo", "flow", {"keyword": "milk"})
    p = _armed_program()
    h = _history()

    peek = p.advance(h)
    assert peek is not None
    _feed(h, peek, HOME)
    nxt = p.advance(h)

    assert nxt is not None
    assert nxt.tool_calls[1].arguments["name"] == "demo/add-cart"


def test_verify_mismatch_hands_over() -> None:
    write_pack(playbooks={"flow": FLOW})
    program.arm("demo", "flow", {"keyword": "milk"})
    p = _armed_program()
    h = _history()

    _feed(h, p.advance(h), ELSEWHERE)
    leg1 = p.advance(h)
    assert leg1 is not None
    _feed(h, leg1, RESULTS)  # landed on the WRONG known page

    assert p.advance(h) is None


def test_error_result_hands_over() -> None:
    write_pack(playbooks={"flow": FLOW})
    program.arm("demo", "flow", {"keyword": "milk"})
    p = _armed_program()
    h = _history()

    _feed(h, p.advance(h), ELSEWHERE)
    leg1 = p.advance(h)
    assert leg1 is not None
    _feed(h, leg1, "BLOCKED — not executed", error=True)

    assert p.advance(h) is None


def test_decide_node_hands_over() -> None:
    write_pack(playbooks={"branch": BRANCH})
    program.arm("demo", "branch", {"keyword": "milk"})
    p = _armed_program()
    h = _history()

    _feed(h, p.advance(h), ELSEWHERE)
    _feed(h, p.advance(h), HOME)  # leg open verified
    _feed(h, p.advance(h), RESULTS)  # leg search verified

    assert p.advance(h) is None  # the DECIDE is a later phase's job


def test_program_advance_never_raises() -> None:
    write_pack(playbooks={"flow": FLOW})
    program.arm("demo", "flow", {"keyword": "milk"})
    p = _armed_program()

    # A malformed history (no pending result will ever match) must
    # degrade to a hand-over, not an exception.
    p.advance(_history())
    assert p.advance(_history()) is None
