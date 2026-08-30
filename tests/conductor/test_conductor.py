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
    from physiclaw.conductor.playbook import ConfirmNode, Playbook
    from physiclaw.conductor.program import Program

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


def _decide_program(routes: dict[str, str]):
    """A one-decision Program (plus a `done` sink when routed to) — the
    shared scaffolding of the broker tests; only the routing differs."""
    from physiclaw.conductor.calls import CALLS
    from physiclaw.conductor.playbook import ConfirmNode, DecideNode, Playbook
    from physiclaw.conductor.program import Program

    nodes: tuple = (
        DecideNode(
            id="choose",
            call="choose_item",
            args={"criteria": "cheapest"},
            context=(),
            outcomes=CALLS["choose_item"].outcomes,
            routes=routes,
            max_visits=3,
        ),
    )
    if "done" in routes.values():
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
    from physiclaw.conductor.micro import MicroOutcome

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
    conductor = Conductor(program=prog, micro=micro)
    history: list = [SystemMessage(content="s"), UserMessage(content="u")]
    await _walk_to_decision(conductor, history)

    # One advance: the decide brokers through the micro-caller, and the
    # pick's tap primitive comes back as the turn.
    tap = await conductor.advance(history)

    assert tap.synthesized and tap.tool_names() == ["note", "tap"]
    assert len(micro.requests) == 1
    assert [c.key for c in micro.requests[0].candidates] == ["牛奶"]


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
    """Stand-by mode: this channel has NO `open` macro, so the overture
    cannot drive — it watches the transcript and reads the intent the
    moment the MODEL lands on the thread, which is exactly what
    `Activation` did before the overture existed."""
    from conductor_fakes import make_screen

    from physiclaw.conductor.channel import Channel
    from physiclaw.conductor.micro import PARSE_TASK, MicroOutcome
    from physiclaw.conductor.overture import Overture
    from physiclaw.conductor.pages import AnchorDecl, PageDecl, PagePrint
    from physiclaw.conductor.playbook import Pack, Playbook
    from physiclaw.conductor.setup import Activation
    from physiclaw.contract.dto import ToolResultMessage

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
    overture = Overture(
        channel=channel, activation=activation, prints=list(channel.prints)
    )
    micro = FakeMicro(
        lambda req: MicroOutcome(
            out="demo/flow", reason="task", confidence=0.9, payload={}
        )
    )
    conductor = Conductor(micro=micro, overture=overture)
    history: list = [SystemMessage(content="s"), UserMessage(content="u")]
    history.append(
        ToolResultMessage(
            tool_call_id="t1",
            content=make_screen(("MyChat", 0.5, 0.05), ("买牛奶", 0.25, 0.4)).text,
        )
    )

    turn = await conductor.advance(history)

    # parse_task fired on the thread screen, armed the program, and the
    # program's first synthesized turn (the locate peek) came back — all
    # in one advance, zero provider calls.
    assert len(micro.requests) == 1 and micro.requests[0].call == PARSE_TASK
    assert turn.synthesized and turn.tool_names() == ["note", "peek"]
