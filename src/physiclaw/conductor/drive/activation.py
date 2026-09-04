"""Activation — the boot's menu, its one parse_task call, and the
program a positive answer builds.

`discover` reads every pack on disk once at wake: the entries the boot
may offer (enabled and live) and the whole dispatch table. `Activation`
turns a THREAD screen into the scoped parse_task request and its
outcome into a `Program` — the baton the boot's `activate` step
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
from physiclaw.conductor.spec.model import Pack, Playbook, PlaybookError
from physiclaw.conductor.spec.pack import (
    disabled_macros,
    list_apps,
    load_pack,
    qualified_all,
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
    rides the boot program for its `activate` step (`step_activate.py`).
    `entries` is the single source — the answer space is its keys, the
    menu a render of it, each value the parsed spec+pack so a positive
    answer activates without re-reading disk."""

    entries: dict[str, tuple[Playbook, Pack]]
    channel: Channel | None
    # parse_task's context: the recent daily-log entries, assembled by
    # `activation_from`.
    context: str = ""

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
        return build_program(spec, pack, values, self.channel)


def discover() -> "tuple[dict[str, tuple[Playbook, Pack]], dict[str, Macro]]":
    """Every pack on disk, once: the activation entries (ref → spec and
    pack, enabled and live only — what the boot may offer) and the
    whole dispatch table (disabled playbooks included — gating is the
    entries filter, never the table). Fail-open per pack."""
    entries: dict[str, tuple[Playbook, Pack]] = {}
    hidden: dict[str, Macro] = {}
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
            continue
        hidden.update(qualified_all(app, pack))
        for entry in scan_playbooks(app, pack):
            spec = entry.spec
            if spec is None or not spec.enabled or disabled_macros(spec, pack):
                continue
            entries[f"{app}/{entry.name}"] = (spec, pack)
    return entries, hidden


def activation_for(channel: Channel | None) -> Activation | None:
    """The activation a boot walked OUTSIDE a wake (a stepping tool, a
    rehearsal) runs with — the same discovery, or None when no enabled
    playbook is on disk (the step then hands over, saying so)."""
    entries, _ = discover()
    return activation_from(entries, channel) if entries else None


def activation_from(
    entries: "dict[str, tuple[Playbook, Pack]]", channel: Channel | None
) -> Activation:
    return Activation(
        entries=entries, channel=channel, context=_activation_context(entries)
    )


def _activation_context(entries: dict[str, tuple[Playbook, Pack]]) -> str:
    """parse_task's context, assembled once at wake, from the agent's
    OWN memory convention (never a conductor-private store): the recent
    daily-log window — where completed purchases and suspensions are
    recorded, so "never re-run a finished task" reads off the same
    record the model would. The one loader agent steps declare through."""
    return context.load((context.DAYLOG,))
