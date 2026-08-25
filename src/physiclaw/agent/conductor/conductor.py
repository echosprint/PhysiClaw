"""Conductor — orchestration only: decide who produces each turn.

The turn loop never calls the LLM provider directly; it asks
``Conductor.advance()``. The conductor's whole job is that decision:
while an armed playbook program is live, the program synthesizes the
turn (``program.py`` — legs as ``run_macro``, verified against page
fingerprints); the moment it completes or hands over, and whenever
nothing is armed, ``advance()`` falls back to calling the provider with
the context the runtime curated. It holds no session state and does no
session management — context assembly, compaction policy, and wire
logging stay with the engine (`EngineRun` wiring).

Going quiet is permanent for the session: a program that handed over is
dropped, not retried — the transcript of its synthesized turns and their
results IS the handoff the model resumes from. Without a program,
``advance()`` is a verbatim provider call — behavior identical to the
loop calling the provider itself; ``tests/agent/conductor`` pins the
delegation by object identity.
"""

from physiclaw.agent.conductor.program import Program
from physiclaw.agent.engine.dto import AssistantMessage, Message
from physiclaw.agent.provider import Provider


class Conductor:
    """Turn arbiter: the armed program produces turns while it can; the
    provider produces every other turn."""

    def __init__(self, provider: Provider, program: Program | None = None):
        self._provider = provider
        self._program = program

    async def advance(
        self,
        history: list[Message],
        tools: list[dict],
    ) -> AssistantMessage:
        """Produce the next assistant turn. Program first; no playbook (or
        a program that handed over) → ask the LLM."""
        if self._program is not None:
            turn = self._program.advance(history)
            if turn is not None:
                return turn
            self._program = None  # quiet is permanent — transcript is the handoff
        return await self._provider.chat(history, tools)
