"""Conductor — orchestration only: decide who produces each turn.

The turn loop never calls the LLM provider while a driver is live; it
asks ``Conductor.advance()``. The conductor's whole job is that
decision: while a program or the overture is live, IT synthesizes the
turn (``overture.py`` — boot to the user's thread and read the intent
there; ``program.py`` — legs as ``run_macro``, verified against page
fingerprints) and the conductor brokers their decision requests through
the wired micro-caller (``micro.py``); when they complete or hand over
— one final synthesized brief turn (``brief.py``), then None — "not
mine" — the loop calls the provider with the context the runtime
curated. It holds no session
state and does no session management — context assembly, compaction
policy, and wire logging stay with the engine (the micro-caller carries
its own sinks for the same reason).

Going quiet is permanent for the session: a driver that handed over is
dropped, not retried — the transcript of its synthesized turns and their
results IS the handoff the model resumes from. The one exception is an
overture that produced no turn but is not ``done``: it has no hand to
navigate with and is watching for the model to reach the thread, so it
is kept. With no driver left, every ``advance()`` is None — behavior
identical to a session that never had a conductor.
"""

import logging
from typing import Protocol

from physiclaw.conductor.micro import DecisionRequest, MicroCaller, MicroOutcome
from physiclaw.conductor.overture import Overture
from physiclaw.conductor.program import Program
from physiclaw.contract.dto import AssistantMessage, Message

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
    producing turns while it can; None means the LLM produces this one."""

    def __init__(
        self,
        program: Program | None = None,
        micro: MicroCaller | None = None,
        overture: Overture | None = None,
    ):
        self._program = program
        self._micro = micro
        self._overture = overture

    def abandon(self) -> None:
        """Session teardown (the plugin's aclose): a walk still in
        flight records its abandoned row and breadcrumb —
        `Program.abandon` is latched, so a walk that closed properly is
        a no-op. The overture's un-taken baton counts too (built at
        resolve, session died before the next advance). Fail-open:
        teardown must never raise."""
        for program in (
            self._program,
            self._overture.program if self._overture is not None else None,
        ):
            if program is None:
                continue
            try:
                program.abandon()
            except Exception:
                log.exception("conductor: abandon record failed — ignored")

    async def advance(self, history: list[Message]) -> AssistantMessage | None:
        """Produce the next assistant turn, or None ("the LLM speaks").
        The overture first — it boots to the user's thread and reads the
        intent there, and may hand a program back; then the program,
        brokering any decision requests it raises; no playbook (or one
        that handed over) → None."""
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
        return None

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
