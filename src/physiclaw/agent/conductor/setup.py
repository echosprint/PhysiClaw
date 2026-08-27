"""Wake-time setup — how a `Program` comes to exist.

Three doors into a walk, all fail-open (a missing, stale, or invalid
file degrades to a normal session, never takes one down):

  - ``load_suspended`` — a suspended walk resumes at its stored node
    (one-shot; ANY wake resumes, the WAIT job is just the alarm clock).
  - ``load_armed`` — the standing order from `physiclaw playbooks arm`.
  - ``Overture`` — nothing armed, but playbooks exist: the conductor
    boots to the user's thread and fires ONE parse_task micro-call over
    the playbook menu (``overture.py``); a positive answer builds the
    program on the spot. ``Activation`` is that last half — menu, call,
    and build — with the overture owning the navigation that reaches a
    thread screen to ask over.

``session_setup`` is the engine's single wake-time conductor call: it
resolves those doors in priority order and assembles the hidden
qualified-macro registry — every pack plus the channel on the overture
path (the boot may activate any playbook, so all conductor hands must be
dispatchable); a live program narrows it to its own pack + channel, and
no channel means a channel-less registry.
``_build_program`` is the one Program constructor call — a program is
whole at construction: channel, origin, and any suspended state included.
"""

import json
import logging
from dataclasses import dataclass

from physiclaw.agent.conductor import arming, scaffold
from physiclaw.agent.conductor.channel import Channel, load_channel
from physiclaw.agent.conductor.ledger import check_ledger_value
from physiclaw.agent.conductor.micro import (
    LIST_INPUT_MARK,
    NOT_A_TASK,
    PARSE_TASK,
    DecisionRequest,
    MicroOutcome,
    build_request,
)
from physiclaw.agent.conductor.overture import Overture
from physiclaw.agent.conductor.pages import (
    IOS_APP,
    RESERVED_APPS,
    prints_for_app,
)
from physiclaw.agent.conductor.playbook import (
    Pack,
    Playbook,
    PlaybookError,
    disabled_leg_macros,
    list_apps,
    load_pack,
    qualified_pack,
    scan_playbooks,
)
from physiclaw.agent.conductor.program import Program
from physiclaw.agent.macros.model import Macro
from physiclaw.common.listing import Screen
from physiclaw.common.text import read_text

log = logging.getLogger(__name__)


def load_armed(channel: "Channel | None" = None) -> "Program | None":
    """The armed playbook as a ready `Program`, or None. Fail-open on
    everything: no arm file, a pack edited out from under it, bad inputs —
    the session runs as if nothing were armed, with one warning saying
    why."""
    try:
        data = arming.read_armed()
        if data is None:
            return None
        if data.get("schema") != arming.ARMED_SCHEMA:
            raise PlaybookError(f"unknown armed.json schema {data.get('schema')!r}")
        app = str(data["app"])
        name = str(data["playbook"])
        raw_inputs = data.get("inputs") or {}
        spec, pack = arming.armed_spec(app, name)
        values = arming.resolve_inputs(spec, raw_inputs)
        # `arm` validated the value it wrote, but the file is on disk —
        # re-hold a hand-edited ledger to the same caps.
        check_ledger_value(spec, values)
    except Exception as e:
        log.warning(
            "armed playbook could not load (%s) — session runs without it: %s",
            arming.armed_path(),
            e,
        )
        return None
    log.info("conductor: armed %s/%s (%d nodes)", app, name, len(spec.nodes))
    return _build_program(app, spec, pack, values, channel, origin="armed")


def load_suspended(channel: "Channel | None" = None) -> "Program | None":
    """A suspended walk restored at its node, or None. One-shot: the file is
    consumed on load (a crash mid-resume loses the suspension, and the next
    wake runs as a plain session — fail-open, never a loop). The WAIT
    job that may also fire is just the alarm clock; ANY wake resumes.
    `channel` avoids a second channel load when the caller (session_setup)
    already holds one."""
    p = arming.suspended_path()
    if not p.exists():
        return None
    try:
        data = json.loads(read_text(p))
        if data.get("schema") != arming.SUSPENDED_SCHEMA:
            raise PlaybookError(f"unknown suspended schema {data.get('schema')!r}")
        app = str(data["app"])
        name = str(data["playbook"])
        spec, pack = arming.armed_spec(app, name)
        program = _build_program(
            app,
            spec,
            pack,
            {str(k): str(v) for k, v in (data.get("values") or {}).items()},
            channel if channel is not None else load_channel(),
            # The suspension carries its lineage: an activation-built walk's
            # terminal outcome must never consume an arm file it never
            # owned, even when a same-named arm exists by coincidence.
            # (Absent in older suspends — those were armed-lineage only.)
            origin="activation" if data.get("origin") == "activation" else "suspended",
            suspended=data,
        )
    except Exception as e:
        log.warning("suspended playbook could not load (%s) — dropped: %s", p, e)
        return None
    finally:
        arming.clear_suspended()  # one-shot: consumed on ANY load outcome
    return program


def _build_program(
    app: str,
    spec: Playbook,
    pack: Pack,
    values: dict[str, str],
    channel: "Channel | None",
    *,
    origin: str,
    suspended: dict | None = None,
) -> "Program":
    """The one Program constructor call — armed, suspended, and activation
    builds all come through here: a program is whole at construction
    (channel, origin, and any suspended state included), never patched up
    afterwards. `origin` decides whether a terminal outcome consumes the
    arm file (see `Program.retire`)."""
    return Program(
        app=app,
        spec=spec,
        values=values,
        pack_macros=qualified_pack(app, pack),
        prints=prints_for_app(app),
        channel=channel,
        origin=origin,
        suspended=suspended,
    )


@dataclass
class Activation:
    """The parse_task half of the boot: turn a THREAD screen into one
    scoped ask, and its answer into a Program. Reaching a thread screen
    is the overture's job (`overture.py`); this owns only the menu, the
    call, and the build. `entries` is the single source — the answer
    space is its keys, the menu a render of it, each value the parsed
    spec+pack so a positive answer activates without re-reading disk."""

    entries: dict[str, tuple[Playbook, Pack]]
    channel: Channel

    def request(self, screen: Screen) -> DecisionRequest:
        """The parse_task request for a thread screen. The CALLER
        establishes that the screen IS the thread — the overture knows,
        because it either drove there or watched the model arrive — so
        this is now purely "turn the menu and this screen into a call".
        `entries` is non-empty by construction — `session_setup` stands
        down before building an Activation with nothing to offer."""
        # Playbook refs only — the `not_a_task` escape is the call's own
        # (its _SPECS row appends it; no caller can forget the exit).
        return build_request(
            PARSE_TASK,
            "activation",
            tuple(self.entries),
            {"menu": self._menu()},
            screen,
        )

    def _menu(self) -> str:
        lines = ["Available playbooks:"]
        for ref, (spec, _pack) in self.entries.items():
            inputs = ", ".join(
                # The mark keys the parse_task prompt's JSON-array rule.
                f"{i.name} {LIST_INPUT_MARK} ({i.description})"
                if i.kind == "list"
                else f"{i.name} ({i.description})"
                for i in spec.inputs
            )
            lines.append(
                f"- {ref}: {spec.description}"
                + (f" [inputs: {inputs}]" if inputs else "")
            )
        return "\n".join(lines)

    def build(self, outcome: MicroOutcome | None) -> "Program | None":
        """A ready Program from a parse_task outcome, or None (not a
        task, low confidence, or inputs that don't resolve — all stay in
        default mode, fail-open)."""
        if outcome is None or outcome.out == NOT_A_TASK:
            return None
        app = outcome.out.partition("/")[0]
        spec, pack = self.entries[outcome.out]
        try:
            values = arming.resolve_inputs(spec, outcome.payload or {})
            check_ledger_value(spec, values)
        except PlaybookError as e:
            log.warning("activation %s: inputs did not resolve (%s)", outcome.out, e)
            return None
        log.info("conductor: activated %s (%s)", outcome.out, outcome.reason)
        return _build_program(
            app, spec, pack, values, self.channel, origin="activation"
        )


def session_setup() -> "tuple[Program | None, Overture | None, dict[str, Macro]]":
    """The engine's one wake-time conductor call, fail-open throughout:
    (suspended-or-armed program, the overture, the hidden qualified macro
    registry). The registry spans every pack plus the channel ON THE
    OVERTURE PATH — the boot may activate any playbook, so all conductor
    hands must be dispatchable; a live program narrows it to its own pack
    + channel, and no channel means the channel-less registry only.

    A playbook on disk IS the grant: with any enabled playbook and a
    channel, the overture drives. Nothing armed and nothing enabled means
    a plain model session, exactly as before."""
    hidden: dict[str, Macro] = {}
    channel = load_channel()
    if channel is not None:
        hidden.update(channel.macros)
    program = load_suspended(channel) or load_armed(channel)
    if program is not None:
        # A live program names only its own pack + the channel — the
        # full cross-pack discovery below is the overture's need, and
        # the overture is off while a program drives (a suspended walk
        # navigates to the thread on its own account).
        hidden.update(program.pack_macros)
        return program, None, hidden
    if channel is None:
        # No channel → nothing to boot to and nothing to ask over; no
        # consumer for the discovery, so skip the every-pack parse.
        return None, None, hidden
    entries: dict[str, tuple[Playbook, Pack]] = {}
    for app in list_apps():
        if app in RESERVED_APPS:
            # Infrastructure namespaces, not task packs: `channel` is the
            # conductor's own hands, and a user override of a built-in
            # (`ios`) is page declarations only — neither holds playbooks.
            continue
        try:
            pack = load_pack(app)
        except Exception as e:
            log.warning("pack %s unusable at wake (%s) — skipped", app, e)
            continue
        hidden.update(qualified_pack(app, pack))
        for entry in scan_playbooks(app, pack):
            spec = entry.spec
            if spec is None or not spec.enabled or disabled_leg_macros(spec, pack):
                continue
            entries[f"{app}/{entry.name}"] = (spec, pack)
    if not entries:
        return None, None, hidden
    # The overture matches against the thread and the OS states — the two
    # things it must tell apart. `ensure_ios_pack` materializes the OS
    # declarations on first look so the boot does not depend on the user
    # having run `playbooks init ios`; both loads are fail-open, and
    # missing prints just mean more screens read as unknown.
    prints = list(channel.prints)
    if channel.open is not None:
        # Drive mode only: stand-by never routes on an OS state, so a
        # channel with no `open` macro must not pay the disk read at wake
        # — nor score a candidate page it can never act on, every turn.
        try:
            scaffold.ensure_ios_pack()
            prints += prints_for_app(IOS_APP)
        except Exception as e:
            log.warning(
                "ios pages unavailable (%s) — a locked phone reads as unknown", e
            )
    overture = Overture(
        channel=channel,
        activation=Activation(entries=entries, channel=channel),
        prints=prints,
    )
    return None, overture, hidden
