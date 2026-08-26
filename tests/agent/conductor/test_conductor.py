"""Tests for `physiclaw.agent.conductor` — the turn arbiter, default
playbook only.

With no playbook, `advance()` must be indistinguishable from the loop
calling the provider itself. Identity assertions (`is`) are the
strongest form of that pin — the same objects in and out means the same
bytes on the wire.
"""

from __future__ import annotations

import pytest

from physiclaw.agent.conductor import Conductor
from physiclaw.agent.engine.dto import (
    AssistantMessage,
    FinishReason,
    SystemMessage,
    Usage,
    UserMessage,
)


class RecordingProvider:
    """File-local fake recording exact call arguments, by identity."""

    def __init__(self):
        self.chat_args: tuple | None = None
        self.response = AssistantMessage(
            content="ok",
            tool_calls=[],
            finish_reason=FinishReason.TOOL_CALLS,
            usage=Usage(),
        )

    async def chat(self, history, tools):
        self.chat_args = (history, tools)
        return self.response


@pytest.mark.asyncio
async def test_advance_without_playbook_calls_the_provider_by_identity() -> None:
    inner = RecordingProvider()
    history = [SystemMessage(content="sys"), UserMessage(content="hi")]
    tools = [{"name": "peek"}]

    result = await Conductor(inner).advance(history, tools)

    assert inner.chat_args[0] is history
    assert inner.chat_args[1] is tools
    assert result is inner.response


# ---------- with an armed program ----------


def _one_node_program():
    """A real Program over a hand-built spec. Its first advance synthesizes
    the locate peek; a history without that peek's result makes the next
    advance hand over — enough to pin the conductor's arbitration without
    a pack on disk."""
    from physiclaw.agent.conductor.playbook import ConfirmNode, Playbook
    from physiclaw.agent.conductor.program import Program

    spec = Playbook(
        app="demo",
        name="x",
        description="d",
        enabled=True,
        inputs=(),
        mandate=None,
        nodes=(ConfirmNode(id="c", compose="m", args={}, message="ok done"),),
    )
    return Program(app="demo", spec=spec, values={}, pack_macros={}, prints=[])


@pytest.mark.asyncio
async def test_advance_prefers_the_program_then_quiet_is_permanent() -> None:
    inner = RecordingProvider()
    conductor = Conductor(inner, program=_one_node_program())
    history = [SystemMessage(content="sys"), UserMessage(content="hi")]

    first = await conductor.advance(history, [])
    assert first.synthesized and first.tool_names() == ["note", "peek"]
    assert inner.chat_args is None  # no provider call while the program drives

    # The peek's result never arrives → the program hands over; from here
    # every advance is a plain provider call, program dropped for good.
    second = await conductor.advance(history, [])
    assert second is inner.response

    third = await conductor.advance(history, [])
    assert third is inner.response


class FakeMicro:
    """Duck-typed MicroCaller: records requests, answers via a factory."""

    def __init__(self, factory):
        self._factory = factory
        self.requests = []

    async def run(self, req):
        from physiclaw.agent.conductor.micro import MicroResult
        from physiclaw.agent.engine.dto import Usage

        self.requests.append(req)
        outcome = self._factory(req)
        return MicroResult(
            outcome=outcome,
            detail=outcome.reason if outcome else "scripted escalate",
            attempts=1,
            usage=Usage(),
            elapsed_ms=1,
        )


def _decide_program(on: dict[str, str]):
    """A one-decision Program (plus a `done` sink when routed to) — the
    shared scaffolding of the broker tests; only the routing differs."""
    from physiclaw.agent.conductor.calls import CALLS
    from physiclaw.agent.conductor.playbook import ConfirmNode, DecideNode, Playbook
    from physiclaw.agent.conductor.program import Program

    nodes: tuple = (
        DecideNode(
            id="choose",
            call="choose_item",
            args={"criteria": "cheapest"},
            context=(),
            outs=CALLS["choose_item"].outs,
            on=on,
            max_visits=3,
        ),
    )
    if "done" in on.values():
        nodes += (ConfirmNode(id="done", compose="m", args={}, message="ok done"),)
    spec = Playbook(
        app="demo",
        name="x",
        description="d",
        enabled=True,
        inputs=(),
        mandate=None,
        nodes=nodes,
    )
    return Program(app="demo", spec=spec, values={}, pack_macros={}, prints=[])


async def _walk_to_decision(conductor, history) -> None:
    """Drive the opening peek and feed its one-row screen result."""
    from conductor_fakes import make_screen

    from physiclaw.agent.engine.dto import ToolResultMessage

    peek = await conductor.advance(history, [])
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
    from physiclaw.agent.conductor.micro import MicroOutcome

    prog = _decide_program(
        {
            "pick": "done",
            "scroll": "choose",
            "none_fit": "escalate",
            "escalate": "escalate",
        }
    )
    micro = FakeMicro(
        lambda req: MicroOutcome(
            out="pick", reason="cheapest", confidence=0.9, picked=req.candidates[0]
        )
    )
    inner = RecordingProvider()
    conductor = Conductor(inner, program=prog, micro=micro)
    history: list = [SystemMessage(content="s"), UserMessage(content="u")]
    await _walk_to_decision(conductor, history)

    # One advance: the decide brokers through the micro-caller, and the
    # pick's tap primitive comes back as the turn.
    tap = await conductor.advance(history, [])

    assert tap.synthesized and tap.tool_names() == ["note", "tap"]
    assert len(micro.requests) == 1
    assert [c.key for c in micro.requests[0].candidates] == ["牛奶"]
    assert inner.chat_args is None  # still zero provider calls


@pytest.mark.asyncio
async def test_unwired_micro_caller_means_decides_hand_over() -> None:
    prog = _decide_program(
        {
            "pick": "escalate",
            "scroll": "escalate",
            "none_fit": "escalate",
            "escalate": "escalate",
        }
    )
    inner = RecordingProvider()
    conductor = Conductor(inner, program=prog)  # no micro wired
    history: list = [SystemMessage(content="s"), UserMessage(content="u")]
    await _walk_to_decision(conductor, history)

    result = await conductor.advance(history, [])

    assert result is inner.response  # handed over, provider took the turn


@pytest.mark.asyncio
async def test_advance_activates_a_playbook_off_the_thread_screen() -> None:
    from conductor_fakes import make_screen

    from physiclaw.agent.conductor.channel import Channel
    from physiclaw.agent.conductor.micro import PARSE_TASK, MicroOutcome
    from physiclaw.agent.conductor.pages import AnchorDecl, PageDecl, PagePrint
    from physiclaw.agent.conductor.playbook import Pack, Playbook
    from physiclaw.agent.conductor.setup import Activation
    from physiclaw.agent.engine.dto import ToolResultMessage

    channel = Channel(
        prints=[
            PagePrint(
                app="channel",
                decl=PageDecl(name="thread", anchors=(AnchorDecl(text="MyChat"),)),
            )
        ],
        macros={},
    )
    spec = Playbook(
        app="demo",
        name="flow",
        description="d",
        enabled=True,
        inputs=(),
        mandate=None,
        nodes=(),
    )
    pack = Pack(app="demo", pages={}, macros={}, macro_errors={})
    activation = Activation(entries={"demo/flow": (spec, pack)}, channel=channel)
    micro = FakeMicro(
        lambda req: MicroOutcome(
            out="demo/flow", reason="task", confidence=0.9, payload={}
        )
    )
    inner = RecordingProvider()
    conductor = Conductor(inner, micro=micro, activation=activation)
    history: list = [SystemMessage(content="s"), UserMessage(content="u")]
    history.append(
        ToolResultMessage(
            tool_call_id="t1",
            content=make_screen(("MyChat", 0.5, 0.05), ("买牛奶", 0.25, 0.4)).text,
        )
    )

    turn = await conductor.advance(history, [])

    # parse_task fired on the thread screen, armed the program, and the
    # program's first synthesized turn (the locate peek) came back — all
    # in one advance, zero provider calls.
    assert len(micro.requests) == 1 and micro.requests[0].call == PARSE_TASK
    assert turn.synthesized and turn.tool_names() == ["note", "peek"]
    assert inner.chat_args is None
