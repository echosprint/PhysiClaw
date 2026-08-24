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
