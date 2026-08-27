"""Conductor — orchestration only: decide who produces each turn.

The turn loop never calls the LLM provider directly; it asks
``Conductor.advance()``. The conductor's whole job is that decision:
while a program or the overture is live, IT synthesizes the turn
(``overture.py`` — boot to the user's thread and read the intent there;
``program.py`` — legs as ``run_macro``, verified against page
fingerprints) and the conductor brokers their decision requests through
the engine-wired micro-caller (``micro.py``); the moment they complete
or hand over, ``advance()`` falls back to calling the provider with the
context the runtime curated. It holds no session state and does no
session management — context assembly, compaction policy, and wire
logging stay with the engine (`EngineRun` wiring; the micro-caller
carries its own sinks for the same reason).

Going quiet is permanent for the session: a driver that handed over is
dropped, not retried — the transcript of its synthesized turns and their
results IS the handoff the model resumes from. The one exception is an
overture that produced no turn but is not ``done``: it has no hand to
navigate with and is watching for the model to reach the thread, so it
is kept. With no driver left, ``advance()`` is a verbatim provider call
— behavior identical to the loop calling the provider itself;
``tests/agent/conductor`` pins the delegation by object identity.
"""

import logging
from typing import Protocol

from physiclaw.agent.conductor.micro import DecisionRequest, MicroCaller, MicroOutcome
from physiclaw.agent.conductor.overture import Overture
from physiclaw.agent.conductor.program import Program
from physiclaw.agent.engine.dto import AssistantMessage, Message
from physiclaw.agent.provider import Provider

log = logging.getLogger(__name__)


class Driver(Protocol):
    """What the conductor can hand a turn to — the contract `Overture`
    and `Program` both satisfy structurally.

    Each call answers one of three ways: a synthesized turn to dispatch,
    a `DecisionRequest` for the conductor to broker back through
    `resolve`, or None — "no turn from me". Implementations NEVER raise:
    they catch internally and degrade to None, which is why `_drive` has
    no fail-open wrapper of its own."""

    def advance(
        self, history: list[Message]
    ) -> "AssistantMessage | DecisionRequest | None": ...

    def resolve(
        self, outcome: MicroOutcome | None
    ) -> "AssistantMessage | DecisionRequest | None": ...


class Conductor:
    """Turn arbiter: the overture boots and the program walks, each
    producing turns while it can; the provider produces every other."""

    def __init__(
        self,
        provider: Provider,
        program: Program | None = None,
        micro: MicroCaller | None = None,
        overture: Overture | None = None,
    ):
        self._provider = provider
        self._program = program
        self._micro = micro
        self._overture = overture

    async def advance(
        self,
        history: list[Message],
        tools: list[dict],
    ) -> AssistantMessage:
        """Produce the next assistant turn. The overture first — it boots
        to the user's thread and reads the intent there, and may hand a
        program back; then the program, brokering any decision requests it
        raises; no playbook (or one that handed over) → ask the LLM."""
        if self._program is None and self._overture is not None:
            turn = await self._drive(self._overture, history)
            if turn is not None:
                return turn
            if self._overture.done:
                # Spent: take the baton if there is one, then release the
                # parsed pack entries. Not done = standing by for a later
                # turn (no `open` macro to drive with).
                self._program = self._overture.program
                self._overture = None
        if self._program is not None:
            turn = await self._drive(self._program, history)
            if turn is not None:
                return turn
            # Quiet is permanent — transcript is the handoff (the program
            # already retired itself at the moment it went quiet).
            self._program = None
        return await self._provider.chat(history, tools)

    async def _drive(
        self, driver: Driver, history: list[Message]
    ) -> AssistantMessage | None:
        """Run one driver for one turn, brokering every decision request
        it raises. None = it produced no turn (it went quiet, or is
        standing by) and the caller decides what that means.

        Both drivers speak this contract — synthesize, or ask, or go
        quiet — so the brokering rule lives here once. In particular the
        rule that an unwired micro-caller resolves as NO outcome: the
        program then hands over like any failed decision, and the
        overture finishes empty like `not_a_task`. Neither driver raises
        (both catch internally and degrade to quiet), so there is no
        fail-open wrapper here to disagree with theirs."""
        step = driver.advance(history)
        while isinstance(step, DecisionRequest):
            outcome = None
            if self._micro is not None:
                outcome = (await self._micro.run(step)).outcome
            step = driver.resolve(outcome)
        return step
