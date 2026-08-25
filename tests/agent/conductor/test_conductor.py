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
        nodes=(ConfirmNode(id="c", compose="m", args={}),),
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
