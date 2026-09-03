"""Conductor — orchestration only: decide who produces each turn.

The turn loop never calls the LLM provider while a driver is live; it
asks ``Conductor.advance()``. The conductor's whole job is that
decision: while a program or the overture is live, IT synthesizes the
turn (``overture.py`` — boot to the user's thread and read the intent
there; ``program.py`` — moves as ``run_macro``, verified against page
fingerprints) and the conductor brokers their model requests through
the wired micro-caller (``micro.py``); when they complete or hand over
— one final synthesized brief turn (``brief.py``), then None — "not
mine" — the loop calls the provider with the context the runtime
curated. It holds no session
state and does no session management — context assembly, compaction
policy, and wire logging stay with the engine (the micro-caller carries
its own sinks for the same reason).

Going quiet is permanent for the session: a driver that handed over is
dropped, not retried — the transcript of its synthesized turns and their
results IS the handoff the model resumes from. With no driver left,
every ``advance()`` is None — behavior identical to a session that never
had a conductor.
"""

import logging
from typing import Protocol

from physiclaw.conductor.micro import DecisionRequest, MicroCaller, MicroOutcome
from physiclaw.conductor.overture import Overture
from physiclaw.conductor.program import Program
from physiclaw.conductor.step import Paused
from physiclaw.contract.dto import AssistantMessage, Message

log = logging.getLogger(__name__)


class Driver(Protocol):
    """What the conductor can hand a turn to — the contract `Overture`
    and `Program` both satisfy structurally.

    Each call answers one of three ways: a synthesized turn to dispatch,
    a `DecisionRequest` for the conductor to broker back through
    `resolve`, or None — "no turn from me". A driver may raise on a
    program bug; `_drive` is the one place that turns that into `crash`
    (recorded, quiet from here), because a session must survive the
    conductor while every tool and test sees the traceback."""

    def advance(
        self, history: list[Message]
    ) -> "AssistantMessage | DecisionRequest | Paused | None": ...

    def resolve(
        self, outcome: MicroOutcome | None
    ) -> "AssistantMessage | DecisionRequest | Paused | None": ...

    def crash(self) -> None: ...


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
            # Spent: take the baton if there is one, then release the
            # parsed pack entries.
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
        it raises. None = it produced no turn (it went quiet) and the
        caller decides what that means.

        Both drivers speak this contract — synthesize, or ask, or go
        quiet — so the brokering rule lives here once. In particular the
        rule that an unwired micro-caller resolves as NO outcome: the
        program then hands over like any failed decision, and the
        overture finishes empty like `not_a_task`. Neither driver raises
        (both catch internally and degrade to quiet), so there is no
        fail-open wrapper here to disagree with theirs."""
        try:
            step = driver.advance(history)
            while isinstance(step, DecisionRequest):
                outcome = None
                if self._micro is not None:
                    outcome = (await self._micro.run(step)).outcome
                step = driver.resolve(outcome)
            if isinstance(step, Paused):
                # Only a stepping tool sets a walk to pause; under the
                # engine it is a program bug like any other.
                raise RuntimeError("a walk paused under the engine")
        except Exception:
            log.exception("conductor: driver crashed — handing over to the model")
            try:
                driver.crash()
            except Exception:
                log.exception("conductor: crash record failed — ignored")
            return None
        return step
