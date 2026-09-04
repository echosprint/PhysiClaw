"""Building a walk — the one `Program` constructor call.

`build_program` is how every door assembles a walk — the wake's
suspension and boot, the CLI rehearsal, the stepping tool, the offline
replay, and the tests — so a program is whole at construction (channel,
OS prints, any restored projection) and never patched up afterwards.
`load_spec` is the one place a playbook's spec and live-readiness are
read off disk for a named ref, `resolve_inputs` the one input resolver.
"""

import logging

from physiclaw.conductor.spec import pack
from physiclaw.conductor.spec.channel import Channel
from physiclaw.conductor.spec.conventions import IOS_APP
from physiclaw.conductor.spec.model import Pack, Playbook, PlaybookError
from physiclaw.conductor.spec.pages import PagePrint, prints_for_app
from physiclaw.conductor.walk.program import Program
from physiclaw.conductor.walk.step import Activator
from physiclaw.macros import inputs as macro_inputs
from physiclaw.macros.model import MacroError

log = logging.getLogger(__name__)


def build_program(
    spec: Playbook,
    pack_: Pack,
    values: dict[str, str],
    channel: Channel | None,
    *,
    suspended: dict | None = None,
    position: dict | None = None,
    dry: bool = False,
    activation: Activator | None = None,
) -> "Program":
    """The one Program constructor call — the boot's activation, a
    resumed suspension, the CLI rehearsal, and the offline replay all
    come through here, so a program is whole at construction (channel
    and any suspended state included) and never patched up afterwards.
    `dry` (the replay, the boot) runs the walk without writing any
    record; `position` is a stepping tool's checkpoint — the projection
    overlaid without a wake-suspension's cursor floor (`Program`);
    `activation` is the boot's menu of enabled playbooks
    (`activation.activation_for`), which every door passes when the
    route activates."""
    return Program(
        spec=spec,
        values=values,
        pack_macros=pack.qualified_pack(spec.app, pack_)
        | pack.qualified_inline(spec.app, spec),
        prints=prints_for_app(spec.app, decls=pack_.pages) + os_prints(),
        channel=channel,
        suspended=suspended,
        position=position,
        landmarks=pack_.landmarks,
        dry=dry,
        activation=activation,
    )


def os_prints() -> list[PagePrint]:
    """The OS states every walk matches beside its own pages — the ios
    pack's declared lock screen (the matcher reads the cover by shape
    regardless; a declared hint is the sharper belt on a device that
    prints one). Fail-open: missing or malformed just means the shape
    read stands alone."""
    try:
        return prints_for_app(IOS_APP)
    except Exception as e:
        log.warning("ios pages unavailable (%s) — the lock screen reads by shape", e)
        return []


def load_spec(
    app: str, name: str, *, require_live: bool = True
) -> tuple[Playbook, Pack]:
    """The parsed playbook and its pack. `require_live` additionally holds
    it to what a real wake needs — enabled, with every referenced macro enabled —
    which a resuming suspension must satisfy but a rehearsal deliberately
    need not (`physiclaw macros run` rehearses disabled macros for the same
    reason: you rehearse BEFORE you enable)."""
    loaded = pack.load_pack(app)
    entry = next((e for e in pack.scan_playbooks(app, loaded) if e.name == name), None)
    if entry is None:
        raise PlaybookError(f"no playbook {app}/{name} on disk")
    if entry.spec is None:
        raise PlaybookError(f"{app}/{name} is invalid: {entry.error}")
    if require_live:
        pack.require_live(entry.spec, loaded)
    return entry.spec, loaded


def resolve_inputs(spec: Playbook, provided: dict[str, str]) -> dict[str, str]:
    """Provided values against the declared inputs — the macro layer's
    resolution contract verbatim (unknown keys, missing required, defaults,
    strings only), translated to this spec's error class at the one seam."""
    try:
        return macro_inputs.resolve_inputs(spec, provided)
    except MacroError as e:
        raise PlaybookError(str(e)) from e
