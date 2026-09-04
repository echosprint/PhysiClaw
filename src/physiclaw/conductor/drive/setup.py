"""Wake-time setup — the doors a wake enters a walk by.

Two doors, both fail-open (a missing, stale, or invalid file degrades
to a normal session, never takes one down):

  - ``load_suspended`` — a walk that asked the user something resumes at
    its stored node (one-shot; ANY wake resumes, the WAIT job is just the
    alarm clock).
  - the boot — nothing suspended, but enabled playbooks exist and the
    channel pack's boot playbook is live: the conductor walks
    `channel/boot` (the thread page with its declared hands, then the
    `activate` step), which fires ONE parse_task micro-call over the
    playbook menu (`activation.py`) and hands the program it built on
    as its baton.

A playbook on disk IS the grant; there is nothing to pre-declare.

``session_setup`` is the plugin's single wake-time setup call
(`plugin.py` runs it behind the seam): it resolves those doors in
priority order and assembles the hidden qualified-macro registry —
every pack plus the channel on the boot path (the boot may activate
any playbook, so all conductor hands must be dispatchable); a live
program narrows it to its own pack + channel (`walk_registry`), and no
channel means a channel-less registry. Construction itself is
`build.py`'s.
"""

import json
import logging

from physiclaw.common.text import read_text
from physiclaw.conductor.drive.activation import activation_from, discover
from physiclaw.conductor.drive.build import build_program, load_spec
from physiclaw.conductor.spec import scaffold
from physiclaw.conductor.spec.channel import Channel, load_channel
from physiclaw.conductor.spec.model import PlaybookError
from physiclaw.conductor.walk import suspension
from physiclaw.conductor.walk.program import Program
from physiclaw.macros.model import Macro

log = logging.getLogger(__name__)


def load_suspended(channel: Channel | None = None) -> Program | None:
    """A suspended walk restored at its node, or None. One-shot: the file is
    consumed on load (a crash mid-resume loses the suspension, and the next
    wake runs as a plain session — fail-open, never a loop). The WAIT
    job that may also fire is just the alarm clock; ANY wake resumes.
    `channel` avoids a second channel load when the caller (session_setup)
    already holds one."""
    p = suspension.suspended_path()
    if not p.exists():
        return None
    try:
        data = json.loads(read_text(p))
        if data.get("schema") != suspension.SUSPENDED_SCHEMA:
            raise PlaybookError(f"unknown suspended schema {data.get('schema')!r}")
        app = str(data["app"])
        name = str(data["playbook"])
        spec, pack = load_spec(app, name)
        program = build_program(
            spec,
            pack,
            {str(k): str(v) for k, v in (data.get("values") or {}).items()},
            channel if channel is not None else load_channel(),
            suspended=data,
        )
    except Exception as e:
        log.warning("suspended playbook could not load (%s) — dropped: %s", p, e)
        return None
    finally:
        suspension.clear_suspended()  # one-shot: consumed on ANY load outcome
    return program


def walk_registry(program: Program, channel: Channel | None) -> dict[str, Macro]:
    """The qualified dispatch registry one live walk needs — its own
    pack's macros plus the channel's. The ONE spelling of that rule:
    `session_setup` arms a real wake with it and the CLI rehearsal
    dispatches through it, so the two can never drift."""
    registry: dict[str, Macro] = {}
    if channel is not None:
        registry.update(channel.macros)
    registry.update(program.pack_macros)
    return registry


def session_setup() -> tuple[Program | None, dict[str, Macro]]:
    """The plugin's one wake-time setup call, fail-open throughout:
    (the program to drive — a resumed suspension, else the boot — or
    None; the hidden qualified macro registry). The registry spans
    every pack plus the channel ON THE BOOT PATH — the boot may
    activate any playbook, so all conductor hands must be
    dispatchable; a live program narrows it to its own pack + channel,
    and no channel means the channel-less registry only.

    A playbook on disk IS the grant: with any enabled playbook and a
    live boot in the channel pack, the boot walks. Nothing suspended,
    nothing enabled, or no boot means a plain model session."""
    channel = load_channel()
    program = load_suspended(channel)
    if program is not None:
        # A live program names only its own pack + the channel — the
        # full cross-pack discovery below is the boot's need, and the
        # boot is off while a program drives (a suspended walk
        # navigates to the thread on its own account).
        return program, walk_registry(program, channel)
    hidden: dict[str, Macro] = {}
    if channel is not None:
        hidden.update(channel.macros)
    if channel is None or channel.boot is None:
        # No channel → nothing to boot to and nothing to ask over; no
        # live boot (the channel logged why) → no walk to boot with.
        # No consumer for the discovery either way, so skip the
        # every-pack parse.
        return None, hidden
    entries, packs = discover()
    hidden.update(packs)
    if not entries:
        return None, hidden
    # The lock screen is matched by shape; `ensure_ios_pack` materializes
    # the OS declarations on first look so a device that prints a hint
    # gets the sharper belt without the user having run `playbooks init
    # ios`. Fail-open inside.
    scaffold.ensure_ios_pack()
    boot = build_program(
        channel.boot,
        channel.pack,
        {},
        channel,
        dry=True,  # the boot leaves no record: no runs row, no daily-log line
        activation=activation_from(entries, channel),
    )
    hidden.update(boot.pack_macros)  # a hand embedded in the boot route
    return boot, hidden
