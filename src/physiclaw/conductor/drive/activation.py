"""Activation — the boot's menu, its one parse_task call, and the
program a positive answer builds.

`discover` reads every pack on disk once at wake: the entries the boot
may offer (enabled and live), the whole dispatch table, and the roster
— every playbook with its state — the wake log prints so a plain model
session never leaves "why no playbook?" to guesswork. `Activation`
turns a THREAD screen into the scoped parse_task request and its
outcome into a `Program` — the baton the boot's `select` step
(`step_activate.py`) hands the conductor. Reaching the thread is the
boot route's job; this owns only the menu, the call, and the build.
"""

import logging
from dataclasses import dataclass

from physiclaw.common.listing import Screen
from physiclaw.conductor.drive.build import build_program, resolve_inputs
from physiclaw.conductor.spec import context
from physiclaw.conductor.spec.channel import Channel
from physiclaw.conductor.spec.conventions import RESERVED_APPS
from physiclaw.conductor.spec.model import Pack, Playbook, PlaybookEntry, PlaybookError
from physiclaw.conductor.spec.pack import (
    list_apps,
    live_gap,
    load_pack,
    qualified_inline,
    qualified_pack,
    scan_playbooks,
)
from physiclaw.conductor.walk.micro import (
    NOT_A_TASK,
    PARSE_TASK,
    DecisionRequest,
    MicroOutcome,
    build_request,
)
from physiclaw.conductor.walk.program import Program
from physiclaw.contract.plugin import EventSink
from physiclaw.macros.model import Macro, MacroInput

log = logging.getLogger(__name__)


def _menu_input(i: MacroInput) -> str:
    """One declared input on the parse_task menu: name, description —
    and the authored
    `example:`, which is the extraction hint ("五常大米 5kg" shows the
    shape a value should take better than any rule prose)."""
    example = f"; e.g. {i.example}" if i.example else ""
    return f"{i.name} ({i.description}{example})"


@dataclass
class Activation:
    """The parse_task half of the boot: turn a THREAD screen into one
    scoped ask, and its answer into a Program. Reaching a thread screen
    is the boot route's job (`channel/boot.yml`, walked like any
    playbook); this owns only the menu, the call, and the build, and
    rides the boot program for its `select` step (`step_activate.py`).
    `entries` is the single source — the answer space is its keys, the
    menu a render of it, each value the parsed spec+pack so a positive
    answer activates without re-reading disk."""

    entries: dict[str, tuple[Playbook, Pack]]
    channel: Channel | None
    # parse_task's context: the recent daily-log entries, assembled by
    # `activation_from`.
    context: str = ""
    # The session's event stream the activated program records into.
    events: EventSink | None = None

    def request(self, screen: Screen, node_id: str) -> DecisionRequest:
        """The parse_task request for a thread screen. The CALLER
        establishes that the screen IS the thread — the boot's activate
        step knows, its enter check just read it — so this is purely
        "turn the menu and this screen into a call". `node_id` names
        the step for the logs.
        `entries` is non-empty by construction — `activation_for` stands
        down before building an Activation with nothing to offer."""
        # Playbook refs only — the `not_a_task` escape is the call's own
        # (its _SPECS row appends it; no caller can forget the exit).
        return build_request(
            PARSE_TASK,
            node_id,
            tuple(self.entries),
            {"menu": self._menu()},
            screen,
            self.context,
        )

    def _menu(self) -> str:
        """One line per playbook: the ref (the answer key), what it does,
        its inputs. The description is what the model chooses by, so a
        playbook's description names its app the way users say it
        (淘宝, 京东) — `playbooks check` warns when two enabled
        playbooks across packs read the same."""
        lines = ["Available playbooks:"]
        for ref, (spec, _pack) in self.entries.items():
            inputs = ", ".join(_menu_input(i) for i in spec.inputs)
            lines.append(
                f"- {ref}: {spec.description}"
                + (f" [inputs: {inputs}]" if inputs else "")
            )
        return "\n".join(lines)

    def build(self, outcome: MicroOutcome | None) -> Program | None:
        """A ready Program from a parse_task outcome, or None (not a
        task, low confidence, or inputs that don't resolve — all stay in
        default mode, fail-open)."""
        if outcome is None or outcome.out == NOT_A_TASK:
            return None
        spec, pack = self.entries[outcome.out]
        try:
            values = resolve_inputs(spec, outcome.payload or {})
        except PlaybookError as e:
            log.warning("activation %s: inputs did not resolve (%s)", outcome.out, e)
            return None
        log.info("conductor: activated %s (%s)", outcome.out, outcome.reason)
        return build_program(spec, pack, values, self.channel, events=self.events)


@dataclass(frozen=True)
class Discovery:
    """Every pack on disk, read once at wake.

    `entries`: ref → (spec, pack) for the live playbooks only — what the
    boot may offer. `macros`: the whole dispatch table, disabled
    playbooks included (gating is the entries filter, never the table).
    `roster`: one line per playbook (and per unusable pack) with its
    state — `app/name (live)`, `(disabled)`, `(disabled macro: x)`,
    `(invalid: …)` — the wake log's answer to "why no playbook?"."""

    entries: dict[str, tuple[Playbook, Pack]]
    macros: dict[str, Macro]
    roster: list[str]


def playbook_gap(entry: PlaybookEntry, pack: Pack) -> str | None:
    """Why the boot cannot offer this playbook — None when it can: the
    file did not parse, or `pack.live_gap` names the readiness gap."""
    if entry.spec is None:
        return f"invalid: {entry.error or 'unreadable'}"
    return live_gap(entry.spec, pack)


def discover() -> Discovery:
    """Every pack on disk, once — see `Discovery`. Fail-open per pack: a
    pack that will not load is a roster line, never a failed wake."""
    entries: dict[str, tuple[Playbook, Pack]] = {}
    macros: dict[str, Macro] = {}
    roster: list[str] = []
    for app in list_apps():
        if app in RESERVED_APPS:
            # Infrastructure namespaces, not task packs: `channel` is the
            # conductor's own hands, and a user override of a built-in
            # (`ios`) is page declarations only — neither holds app
            # playbooks.
            continue
        try:
            pack = load_pack(app)
        except Exception as e:
            log.warning("pack %s unusable at wake (%s) — skipped", app, e)
            roster.append(f"{app} (pack unusable: {e})")
            continue
        # One scan per pack: the dispatch table (every playbook's inline
        # bodies, disabled ones included) and the roster off the same
        # entries.
        macros.update(qualified_pack(app, pack))
        for entry in scan_playbooks(app, pack):
            ref = f"{app}/{entry.name}"
            gap = playbook_gap(entry, pack)
            roster.append(f"{ref} ({gap or 'live'})")
            if entry.spec is None:
                continue
            macros.update(qualified_inline(app, entry.spec))
            if gap is None:
                entries[ref] = (entry.spec, pack)
    return Discovery(entries=entries, macros=macros, roster=roster)


def activation_for(channel: Channel | None) -> Activation | None:
    """The activation a boot walked OUTSIDE a wake (a stepping tool, a
    rehearsal) runs with — the same discovery, or None when no enabled
    playbook is on disk (the step then hands over, saying so)."""
    found = discover()
    return activation_from(found.entries, channel) if found.entries else None


def activation_from(
    entries: "dict[str, tuple[Playbook, Pack]]",
    channel: Channel | None,
    events: EventSink | None = None,
) -> Activation:
    return Activation(
        entries=entries,
        channel=channel,
        context=_activation_context(entries),
        events=events,
    )


def _activation_context(entries: dict[str, tuple[Playbook, Pack]]) -> str:
    """parse_task's context, assembled once at wake, from the agent's
    OWN memory convention (never a conductor-private store): the recent
    daily-log window — where completed purchases and suspensions are
    recorded, so "never re-run a finished task" reads off the same
    record the model would. The one loader agent steps declare through."""
    return context.load((context.DAYLOG,))
