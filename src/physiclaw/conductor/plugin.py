"""The conductor as a turn plugin — the package's one composition point.

`contract.plugin.TurnPlugin` is the seam: the engine loads this module
by its config-listed dotted path (`[agent] plugins`) and never imports
the conductor's name. Everything the engine used to wire by hand lands
here instead — `setup.session_setup()` at wake, the micro-caller with
its cheap-tier client, and the `Conductor` arbiter per turn.

The micro-caller wiring: None only when nothing drives (no program and
no overture — an ask's reply judgment can fire on any live walk, and
the owned client is built lazily on the FIRST call, so wiring costs
nothing at wake); `[conductor] micro_model`
selects the cheap decision tier, ours to close (`aclose`); fail-open to
the session provider on any problem, at parse time or build time.
"""

import logging
from functools import partial

from physiclaw.common.config import CONFIG, parse_model_ref
from physiclaw.conductor import setup as conductor_setup
from physiclaw.conductor.conductor import Conductor
from physiclaw.conductor.micro import MicroCaller
from physiclaw.contract.dto import AssistantMessage, Message
from physiclaw.contract.plugin import SessionSetup, SetupContext
from physiclaw.provider import make_provider

log = logging.getLogger(__name__)


def build() -> "ConductorPlugin":
    """The factory `[agent] plugins` names."""
    return ConductorPlugin()


class ConductorPlugin:
    """One session's conductor behind the plugin seam. Fail-open like
    everything conductor-side: a crash in setup degrades to a plain
    model session; `advance` never raises (the Conductor and both
    drivers catch internally)."""

    def __init__(self) -> None:
        self._conductor: Conductor | None = None
        self._micro: MicroCaller | None = None

    async def session_setup(self, ctx: SetupContext) -> SessionSetup | None:
        # No fail-open wrapper here: `setup.session_setup` is fail-open
        # internally, and the engine wraps every plugin's setup call at
        # the seam — a third belt would only disagree with theirs (the
        # same rule `Conductor._drive` documents for its drivers).
        program, overture, hidden = conductor_setup.session_setup()
        self._micro = _wire_micro(program, overture, ctx)
        self._conductor = Conductor(
            program=program, micro=self._micro, overture=overture
        )
        return SessionSetup(gated_macros=hidden)

    async def advance(self, history: list[Message]) -> AssistantMessage | None:
        if self._conductor is None:
            return None
        return await self._conductor.advance(history)

    async def aclose(self) -> None:
        # Teardown runs on EVERY session end (the engine's finally): a
        # walk cut short mid-flight (Ctrl-C, wall-clock budget) records
        # its abandoned row + breadcrumb here — the conductor's
        # counterpart of `loop.log_external_stop`, which only covers
        # model sessions (it keys on a drafted plan).
        if self._conductor is not None:
            self._conductor.abandon()
        if self._micro is not None:
            await self._micro.aclose()


def _wire_micro(program, overture, ctx: SetupContext) -> MicroCaller | None:
    """The micro-caller for the conductor's model calls — None only
    when NOTHING drives (no program and no overture). Any live walk can
    need one (an ask's reply judgment, an agent step) — and the owned
    cheap-tier client is built lazily on the FIRST call, so wiring the
    caller costs nothing at wake. The session provider (off the setup
    context) is the fail-open floor."""
    if overture is None and program is None:
        return None
    factory = None
    ref = CONFIG.conductor.micro_model
    if ref:
        try:
            pid, mid = parse_model_ref(ref)
            factory = partial(make_provider, pid, mid)
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
