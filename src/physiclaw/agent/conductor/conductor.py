"""Conductor — orchestration only: decide who produces each turn.

The turn loop never calls the LLM provider directly; it asks
``Conductor.advance()``. The conductor's whole job is that decision:
while an armed playbook program is live, the program synthesizes the
turn (``program.py`` — legs as ``run_macro``, verified against page
fingerprints) and the conductor brokers its decision requests through
the engine-wired micro-caller (``micro.py``); the moment the program
completes or hands over, and whenever nothing is armed, ``advance()``
falls back to calling the provider with the context the runtime
curated. It holds no session state and does no session management —
context assembly, compaction policy, and wire logging stay with the
engine (`EngineRun` wiring; the micro-caller carries its own sinks for
the same reason).

Going quiet is permanent for the session: a program that handed over is
dropped, not retried — the transcript of its synthesized turns and their
results IS the handoff the model resumes from. Without a program,
``advance()`` is a verbatim provider call — behavior identical to the
loop calling the provider itself; ``tests/agent/conductor`` pins the
delegation by object identity.
"""

from physiclaw.agent.conductor.micro import DecisionRequest, MicroCaller
from physiclaw.agent.conductor.program import Activation, Program
from physiclaw.agent.engine.dto import AssistantMessage, Message
from physiclaw.agent.provider import Provider


class Conductor:
    """Turn arbiter: the armed program produces turns while it can; the
    provider produces every other turn."""

    def __init__(
        self,
        provider: Provider,
        program: Program | None = None,
        micro: MicroCaller | None = None,
        activation: Activation | None = None,
    ):
        self._provider = provider
        self._program = program
        self._micro = micro
        self._activation = activation

    async def advance(
        self,
        history: list[Message],
        tools: list[dict],
    ) -> AssistantMessage:
        """Produce the next assistant turn. Activation first (the one
        parse_task ask, when the screen is the user's thread); then the
        program — brokering any decision requests it raises; no playbook
        (or a program that handed over) → ask the LLM."""
        if (
            self._program is None
            and self._activation is not None
            and self._micro is not None
        ):
            req = self._activation.request(history)
            if req is not None:
                res = await self._micro.run(req)
                self._program = self._activation.build(res.outcome)
            if self._activation.attempted:
                # One-shot consumed — release the parsed pack entries.
                self._activation = None
        if self._program is not None:
            step = self._program.advance(history)
            while isinstance(step, DecisionRequest):
                # An unwired micro-caller resolves as no outcome — the
                # program hands over, same as any failed decision.
                outcome = None
                if self._micro is not None:
                    outcome = (await self._micro.run(step)).outcome
                step = self._program.resolve(outcome)
            if step is not None:
                return step
            # Quiet is permanent — transcript is the handoff (the program
            # already retired itself at the moment it went quiet).
            self._program = None
        return await self._provider.chat(history, tools)
