"""The conductor as a turn plugin — the package's one composition point.

`contract.plugin.TurnPlugin` is the seam: the engine loads this module
by its config-listed dotted path (`[agent] plugins`) and never imports
the conductor's name. Everything the engine used to wire by hand lands
here instead — `setup.session_setup()` at wake, the micro-caller with
its cheap-tier client, and the `Conductor` arbiter per turn.

The micro-caller wiring: None only when nothing drives (no program —
the boot, a resumed walk; an agent step can fire on any live walk,
and the owned client is built lazily on the FIRST call, so wiring
costs nothing at wake); `[conductor] micro_model`
selects the cheap decision tier, ours to close (`aclose`); fail-open to
the session provider on any problem, at parse time or build time.
"""

import logging
from functools import partial

from physiclaw.common.config import CONFIG, parse_model_ref
from physiclaw.conductor.drive import setup as conductor_setup
from physiclaw.conductor.drive.conductor import Conductor
from physiclaw.conductor.walk.micro import MicroCaller
from physiclaw.conductor.walk.program import Program
from physiclaw.contract.dto import AssistantMessage, Message
from physiclaw.contract.plugin import SessionSetup, SetupContext
from physiclaw.provider import make_provider

log = logging.getLogger(__name__)


def build() -> "ConductorPlugin":
    """The factory `[agent] plugins` names."""
    return ConductorPlugin()


class ConductorPlugin:
    """One session's conductor behind the plugin seam.

    The promise to the runtime: NOTHING here raises. The engine keeps
    its own belt at the seam (setup, advance, and close are each
    wrapped there), and this class keeps the conductor's: every door
    catches, logs, and degrades — setup to a plain model session,
    advance to "the LLM speaks" with the conductor dropped for the
    rest of the session, close to a logged failure. Two belts on
    purpose: the guarantee must not depend on which side is edited."""

    def __init__(self) -> None:
        self._conductor: Conductor | None = None
        self._micro: MicroCaller | None = None

    async def session_setup(self, ctx: SetupContext) -> SessionSetup | None:
        try:
            program, hidden = conductor_setup.session_setup(events=ctx.events)
            self._micro = _wire_micro(program, ctx)
            self._conductor = Conductor(program=program, micro=self._micro)
            return SessionSetup(gated_macros=hidden)
        except Exception:
            log.exception("conductor setup crashed — plain model session")
            self._conductor = None
            return SessionSetup(gated_macros={})

    async def advance(self, history: list[Message]) -> AssistantMessage | None:
        if self._conductor is None:
            return None
        try:
            return await self._conductor.advance(history)
        except Exception:
            # The conductor is dropped for the rest of the session: its
            # state is unknown, and the transcript so far is the model's
            # hand-off. Every later advance is a pass.
            log.exception("conductor advance crashed — the model takes the session")
            self._conductor = None
            return None

    async def aclose(self) -> None:
        # Teardown runs on EVERY session end (the engine's finally): a
        # walk cut short mid-flight (Ctrl-C, wall-clock budget) records
        # its abandoned row + breadcrumb here — the conductor's
        # counterpart of `loop.log_external_stop`, which only covers
        # model sessions (it keys on a drafted plan). Each close is its
        # own belt, so one failing never skips the other.
        if self._conductor is not None:
            try:
                self._conductor.abandon()
            except Exception:
                log.exception("conductor abandon record failed — ignored")
        if self._micro is not None:
            try:
                await self._micro.aclose()
            except Exception:
                log.warning("conductor micro client close failed", exc_info=True)


def _wire_micro(program: "Program | None", ctx: SetupContext) -> MicroCaller | None:
    """The micro-caller for the conductor's model calls — None only
    when NOTHING drives (no program). Any live walk can need one (the
    boot's parse_task, an ask's reply judgment, an agent step) — and
    the owned cheap-tier client is built lazily on the FIRST call, so
    wiring the caller costs nothing at wake. The session provider (off
    the setup context) is the fail-open floor."""
    if program is None:
        return None
    factory = None
    ref = CONFIG.conductor.micro_model
    if ref:
        try:
            pid, mid = parse_model_ref(ref)
            factory = partial(make_provider, pid, mid, usage_sink=ctx.events)
        except Exception as e:
            log.warning(
                "conductor micro_model %r unusable (%s) — using the session model",
                ref,
                e,
            )
    return MicroCaller(
        ctx.session_provider,
        confidence_floor=CONFIG.conductor.micro_confidence,
        tr=ctx.events,
        rlog=ctx.wire,
        owned_factory=factory,
    )
