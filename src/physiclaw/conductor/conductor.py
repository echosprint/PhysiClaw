"""Conductor — orchestration only: decide who produces each turn.

The turn loop never calls the LLM provider while a walk is live; it
asks ``Conductor.advance()``. The conductor's whole job is that
decision: while a program is live, IT synthesizes the turn
(``program.py`` — moves as ``run_macro``, verified against page
fingerprints; the boot to the user's thread is the channel pack's own
playbook, walked the same way) and the conductor brokers the walk's
model requests through the wired micro-caller (``micro.py``); when a
walk goes quiet it takes the baton the walk may hand on (the boot's
`activate` step built the program the thread asked for) and drives
that; with none — completed, handed over after one final synthesized
brief turn (``brief.py``), or the boot found no task — it answers
None, "not mine", and the loop calls the provider with the context the
runtime curated. It holds no session state and does no session
management — context assembly, compaction policy, and wire logging
stay with the engine (the micro-caller carries its own sinks for the
same reason).

Going quiet is permanent for the session: a walk that handed over is
dropped, not retried — the transcript of its synthesized turns and
their results IS the handoff the model resumes from. With no walk
left, every ``advance()`` is None — behavior identical to a session
that never had a conductor.
"""

import logging

from physiclaw.conductor.micro import DecisionRequest, MicroCaller
from physiclaw.conductor.program import Program
from physiclaw.conductor.step import Paused
from physiclaw.contract.dto import AssistantMessage, Message

log = logging.getLogger(__name__)


class Conductor:
    """Turn arbiter: the live program produces turns while it can, then
    its baton does; None means the LLM produces this one."""

    def __init__(
        self,
        program: Program | None = None,
        micro: MicroCaller | None = None,
    ):
        self._program = program
        self._micro = micro

    def abandon(self) -> None:
        """Session teardown (the plugin's aclose): a walk still in
        flight records its abandoned row and breadcrumb —
        `Program.abandon` is latched, so a walk that closed properly is
        a no-op. An un-taken baton counts too (built at the boot's
        resolve, session died before the next advance). Fail-open:
        teardown must never raise."""
        walk = self._program
        while walk is not None:
            try:
                walk.abandon()
            except Exception:
                log.exception("conductor: abandon record failed — ignored")
            walk = walk.baton

    async def advance(self, history: list[Message]) -> AssistantMessage | None:
        """Produce the next assistant turn, or None ("the LLM speaks"):
        the live program's turn, brokering any decision requests it
        raises; when it goes quiet, its baton's — the boot hands the
        activated program on this way; no walk, or one that handed over
        → None."""
        while self._program is not None:
            turn = await self._drive(self._program, history)
            if turn is not None:
                return turn
            # Quiet is permanent — the transcript is the handoff (the
            # walk already retired itself at the moment it went quiet).
            # The baton, if it set one, is the next walk; else nobody.
            self._program = self._program.baton
        return None

    async def _drive(
        self, program: Program, history: list[Message]
    ) -> AssistantMessage | None:
        """Run one walk for one turn, brokering every decision request
        it raises. None = it produced no turn (it went quiet) and the
        caller decides what that means.

        The brokering rule lives here once — in particular the rule
        that an unwired micro-caller resolves as NO outcome: the walk
        then hands over like any failed decision (the boot finishes
        empty like `not_a_task`). A program bug is caught here, once:
        the walk is crashed (recorded, quiet from here) and the model
        takes the session, because a session must survive the conductor
        while every tool and test sees the traceback."""
        try:
            step = program.advance(history)
            while isinstance(step, DecisionRequest):
                outcome = None
                if self._micro is not None:
                    outcome = (await self._micro.run(step)).outcome
                step = program.resolve(outcome)
            if isinstance(step, Paused):
                # Only a stepping tool sets a walk to pause; under the
                # engine it is a program bug like any other.
                raise RuntimeError("a walk paused under the engine")
        except Exception:
            log.exception("conductor: walk crashed — handing over to the model")
            try:
                program.crash()
            except Exception:
                log.exception("conductor: crash record failed — ignored")
            return None
        return step
