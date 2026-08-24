"""Conductor — orchestration only: decide who produces each turn.

The turn loop never calls the LLM provider directly; it asks
``Conductor.advance()``. The conductor's whole job is that decision: check for an applicable playbook and follow it,
else fall back to calling the provider with the context the runtime
curated. It holds no session state and does no session management —
context assembly, compaction policy, and wire logging stay with the
engine (`EngineRun` wiring).

With no playbooks (the current phase) only the fallback exists, so
``advance()`` is a verbatim provider call — behavior is identical to the
loop calling the provider itself; ``tests/agent/conductor`` pins the
delegation by object identity.
"""

from physiclaw.agent.engine.dto import AssistantMessage, Message
from physiclaw.agent.provider import Provider


class Conductor:
    """Turn arbiter. Today: no playbooks, so every ``advance()`` falls back
    to the wrapped provider."""

    def __init__(self, provider: Provider):
        self._provider = provider

    async def advance(
        self,
        history: list[Message],
        tools: list[dict],
    ) -> AssistantMessage:
        """Produce the next assistant turn. No playbook → ask the LLM."""
        return await self._provider.chat(history, tools)
