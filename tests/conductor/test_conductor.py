"""Tests for the conductor's `Conductor` — the turn arbiter, default
playbook only.

With no playbook, `advance()` must return None — "not mine" — so the
loop's provider call proceeds untouched. A session with no driver left
is indistinguishable from one that never had a conductor; the plugin
seam (`contract.plugin`) relies on exactly that.
"""

from __future__ import annotations

import pytest

from physiclaw.conductor.conductor import Conductor
from physiclaw.contract.dto import (
    SystemMessage,
    UserMessage,
)


@pytest.mark.asyncio
async def test_advance_without_playbook_passes_the_turn() -> None:
    history = [SystemMessage(content="sys"), UserMessage(content="hi")]

    result = await Conductor().advance(history)

    assert result is None  # "the LLM speaks" — nothing intercepted


# ---------- with an armed program ----------


def _one_node_program():
    """A real Program over a hand-built spec. Its first advance synthesizes
    the locate peek; a history without that peek's result makes the next
    advance hand over — enough to pin the conductor's arbitration without
    a pack on disk."""
    from physiclaw.conductor.playbook import Playbook, TellNode
    from physiclaw.conductor.program import Program

    spec = Playbook(
        app="demo",
        name="x",
        description="d",
        enabled=True,
        inputs=(),
        nodes=(TellNode(id="c", message="ok done"),),
    )
    return Program(spec=spec, values={}, pack_macros={}, prints=[])


@pytest.mark.asyncio
async def test_advance_prefers_the_program_then_quiet_is_permanent() -> None:
    conductor = Conductor(program=_one_node_program())
    history = [SystemMessage(content="sys"), UserMessage(content="hi")]

    first = await conductor.advance(history)
    assert first.synthesized and first.tool_names() == ["note", "peek"]

    # The peek's result never arrives → the program hands over with one
    # final brief turn; from there every advance is a pass — program
    # dropped for good, never retried.
    briefed = await conductor.advance(history)
    assert briefed.synthesized and briefed.tool_names() == ["note", "peek"]
    assert "conductor handing over" in briefed.tool_calls[0].arguments["summary"]
    assert await conductor.advance(history) is None
    assert await conductor.advance(history) is None


class FakeMicro:
    """Duck-typed MicroCaller: records requests, answers via a factory."""

    def __init__(self, factory):
        self._factory = factory
        self.requests = []

    async def run(self, req):
        from physiclaw.conductor.micro import MicroResult
        from physiclaw.contract.dto import Usage

        self.requests.append(req)
        outcome = self._factory(req)
        return MicroResult(
            outcome=outcome,
            detail=outcome.reason if outcome else "scripted escalate",
            attempts=1,
            usage=Usage(),
            elapsed_ms=1,
        )


def _agent_program():
    """A Program opening with one pure-text agent step — the shared
    scaffolding of the broker tests."""
    from physiclaw.conductor.playbook import AgentNode, Playbook
    from physiclaw.conductor.program import Program

    nodes: tuple = (
        AgentNode(
            id="parse",
            prompt="the keyword, please",
            tools=(),
            give=(),
            returns=(("keyword", "the search term"),),
            enter="",
            verify="",
            max_calls=1,
            max_scrolls=0,
        ),
    )
    spec = Playbook(
        app="demo",
        name="x",
        description="d",
        enabled=True,
        inputs=(),
        nodes=nodes,
    )
    return Program(spec=spec, values={}, pack_macros={}, prints=[])


async def _walk_to_decision(conductor, history) -> None:
    """Drive the opening peek and feed its one-row screen result."""
    from conductor_fakes import make_screen

    from physiclaw.contract.dto import ToolResultMessage

    peek = await conductor.advance(history)
    assert peek.tool_names() == ["note", "peek"]
    history.append(peek)
    history.append(
        ToolResultMessage(
            tool_call_id=peek.tool_calls[1].id,
            content=make_screen(("牛奶", 0.5, 0.3)).text,
        )
    )


@pytest.mark.asyncio
async def test_advance_brokers_decision_requests_through_the_micro_caller() -> None:
    from physiclaw.conductor.micro import AGENT_FIELDS, MicroOutcome

    prog = _agent_program()
    micro = FakeMicro(
        lambda req: MicroOutcome(
            out="done", reason="clear", confidence=0.9, payload={"keyword": "牛奶"}
        )
    )
    conductor = Conductor(program=prog, micro=micro)
    history: list = [SystemMessage(content="s"), UserMessage(content="u")]
    await _walk_to_decision(conductor, history)

    # One advance: the agent step brokers through the micro-caller, its
    # outputs land, and the walk's next turn comes back (here the
    # completion brief — the one-step route is done).
    turn = await conductor.advance(history)

    assert turn.synthesized and turn.tool_names() == ["note", "peek"]
    assert len(micro.requests) == 1 and micro.requests[0].call == AGENT_FIELDS
    assert prog.outputs == {"parse.keyword": "牛奶"}


@pytest.mark.asyncio
async def test_unwired_micro_caller_means_agent_steps_hand_over() -> None:
    prog = _agent_program()
    conductor = Conductor(program=prog)  # no micro wired
    history: list = [SystemMessage(content="s"), UserMessage(content="u")]
    await _walk_to_decision(conductor, history)

    result = await conductor.advance(history)

    # Handed over via the one final brief turn; then the LLM takes over.
    assert result.synthesized and result.tool_names() == ["note", "peek"]
    assert "conductor handing over" in result.tool_calls[0].arguments["summary"]
    assert await conductor.advance(history) is None


@pytest.mark.asyncio
async def test_advance_activates_a_playbook_off_the_thread_screen() -> None:
    """The boot walks, finds the thread, fires parse_task, and the armed
    program's first turn comes back — the baton, through the one
    arbiter, with no second driver."""
    from conductor_fakes import FLOW, thread_screen, write_channel, write_pack

    from physiclaw.conductor import setup
    from physiclaw.conductor.micro import PARSE_TASK, MicroOutcome
    from physiclaw.contract.dto import ToolResultMessage

    write_channel(CHANNEL_OPEN)
    write_pack(playbooks={"flow": FLOW})
    boot, _ = setup.session_setup()
    assert boot is not None
    micro = FakeMicro(
        lambda req: MicroOutcome(
            out="demo/flow", reason="task", confidence=0.9, payload={"keyword": "milk"}
        )
    )
    conductor = Conductor(program=boot, micro=micro)
    history: list = [SystemMessage(content="s"), UserMessage(content="u")]

    peek = await conductor.advance(history)  # the boot's opening peek
    assert peek.synthesized and peek.tool_names() == ["note", "peek"]
    history.append(peek)
    history.append(
        ToolResultMessage(
            tool_call_id=peek.tool_calls[1].id,
            content=thread_screen(("买牛奶", 0.25, 0.4)),
        )
    )

    turn = await conductor.advance(history)

    # parse_task fired on the thread screen, the boot concluded with the
    # baton, and the activated program's first synthesized turn (its
    # opening peek) came back — all in one advance, zero provider calls.
    assert len(micro.requests) == 1 and micro.requests[0].call == PARSE_TASK
    assert turn.synthesized and turn.tool_names() == ["note", "peek"]
    # Two walks, two call-id scopes: the program's ids can never find
    # the boot's stale results.
    assert peek.tool_calls[1].id.startswith("conductor-channel-boot-")
    assert turn.tool_calls[1].id.startswith("conductor-demo-flow-")
    assert conductor._program is boot.baton and boot.baton.values == {"keyword": "milk"}


@pytest.mark.asyncio
async def test_abandon_covers_an_untaken_baton() -> None:
    # Built at the boot's resolve, the session died before the next
    # advance: the activated walk still records its abandoned row —
    # and the boot, dry, records nothing.
    from conductor_fakes import (
        FLOW,
        feed,
        history,
        thread_screen,
        write_channel,
        write_pack,
    )

    from physiclaw.conductor import setup, walklog
    from physiclaw.conductor.micro import MicroOutcome

    write_channel(CHANNEL_OPEN)
    write_pack(playbooks={"flow": FLOW})
    boot, _ = setup.session_setup()
    assert boot is not None
    h = history()
    feed(h, boot.advance(h), thread_screen(("买牛奶", 0.25, 0.4)))
    boot.advance(h)
    boot.resolve(
        MicroOutcome(
            out="demo/flow", reason="t", confidence=0.9, payload={"keyword": "m"}
        )
    )
    baton = boot.baton
    assert baton is not None
    baton.advance(h)  # the activated walk started: its peek is out

    Conductor(program=boot).abandon()

    rows = walklog.load()
    assert [(r["playbook"], r["outcome"]) for r in rows] == [("flow", "abandoned")]


CHANNEL_OPEN = """\
name: open
description: open the thread
steps:
  - name: go
    tool: tap
    with: {label: t, bbox: [0.1, 0.1, 0.2, 0.2]}
"""


# ---------- a driver bug: caught once, recorded, quiet ----------


@pytest.mark.asyncio
async def test_a_program_bug_becomes_a_crash_record_and_silence(mocker) -> None:
    from physiclaw.conductor.program import Phase
    from physiclaw.conductor.walklog import Outcome

    program = _one_node_program()
    mocker.patch.object(program, "_advance", side_effect=RuntimeError("bug"))
    conductor = Conductor(program=program)
    history = [SystemMessage(content="sys"), UserMessage(content="hi")]

    assert await conductor.advance(history) is None
    assert program.phase is Phase.DONE and program.outcome is Outcome.CRASHED
    assert await conductor.advance(history) is None  # dropped; the model speaks


@pytest.mark.asyncio
async def test_a_failing_crash_record_still_goes_quiet(mocker) -> None:
    from physiclaw.conductor.program import Phase

    program = _one_node_program()
    mocker.patch.object(program, "_advance", side_effect=RuntimeError("bug"))
    mocker.patch.object(program, "_record_run", side_effect=OSError("disk"))
    conductor = Conductor(program=program)

    assert await conductor.advance([SystemMessage(content="sys")]) is None
    assert program.phase is Phase.DONE
