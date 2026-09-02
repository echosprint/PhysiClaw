"""Wake-time setup — how a `Program` comes to exist.

Two doors into a walk, both fail-open (a missing, stale, or invalid file
degrades to a normal session, never takes one down):

  - ``load_suspended`` — a walk that asked the user something resumes at
    its stored node (one-shot; ANY wake resumes, the WAIT job is just the
    alarm clock).
  - ``Overture`` — nothing suspended, but playbooks exist: the conductor
    boots to the user's thread and fires ONE parse_task micro-call over
    the playbook menu (``overture.py``); a positive answer builds the
    program on the spot. ``Activation`` is that last half — menu, call,
    and build — with the overture owning the navigation that reaches a
    thread screen to ask over.

A playbook on disk IS the grant; there is nothing to pre-declare.

``session_setup`` is the plugin's single wake-time setup call
(`plugin.py` runs it behind the seam): it
resolves those doors in priority order and assembles the hidden
qualified-macro registry — every pack plus the channel on the overture
path (the boot may activate any playbook, so all conductor hands must be
dispatchable); a live program narrows it to its own pack + channel, and
no channel means a channel-less registry.
``build_program`` is the one Program constructor call — a program is
whole at construction, suspended state included — and the one place a
playbook's spec, inputs and live-readiness are resolved. The CLI's
rehearsal (`physiclaw playbooks run`) and the tests build through it too,
so nothing gets to assemble a walk by a private route.
"""

import json
import logging
from dataclasses import dataclass

from physiclaw.common import daylog
from physiclaw.common.config import CONFIG
from physiclaw.common.listing import Screen
from physiclaw.common.text import read_text
from physiclaw.conductor import reply, scaffold, suspension
from physiclaw.conductor.channel import Channel, load_channel
from physiclaw.conductor.micro import (
    NOT_A_TASK,
    PARSE_TASK,
    DecisionRequest,
    MicroOutcome,
    build_request,
)
from physiclaw.conductor.overture import Overture
from physiclaw.conductor.pages import (
    IOS_APP,
    RESERVED_APPS,
    prints_for_app,
)
from physiclaw.conductor.playbook import (
    AskNode,
    Pack,
    Playbook,
    PlaybookError,
    disabled_macros,
    list_apps,
    load_pack,
    qualified_inline,
    qualified_pack,
    scan_playbooks,
)
from physiclaw.conductor.program import Program
from physiclaw.macros import inputs as macro_inputs
from physiclaw.macros.model import Macro, MacroError

log = logging.getLogger(__name__)


def load_suspended(channel: "Channel | None" = None) -> "Program | None":
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


def build_program(
    spec: Playbook,
    pack: Pack,
    values: dict[str, str],
    channel: "Channel | None",
    *,
    suspended: dict | None = None,
) -> "Program":
    """The one Program constructor call — the overture's activation, a
    resumed suspension, and the CLI rehearsal all come through here, so a
    program is whole at construction (channel and any suspended state
    included) and never patched up afterwards."""
    return Program(
        spec=spec,
        values=values,
        pack_macros=qualified_pack(spec.app, pack) | qualified_inline(spec.app, spec),
        prints=prints_for_app(spec.app, decls=pack.pages),
        channel=channel,
        suspended=suspended,
        landmarks=pack.landmarks,
    )


def load_spec(
    app: str, name: str, *, require_live: bool = True
) -> tuple[Playbook, Pack]:
    """The parsed playbook and its pack. `require_live` additionally holds
    it to what a real wake needs — enabled, with every referenced macro enabled —
    which a resuming suspension must satisfy but a rehearsal deliberately
    need not (`physiclaw macros run` rehearses disabled macros for the same
    reason: you rehearse BEFORE you enable)."""
    pack = load_pack(app)
    entry = next((e for e in scan_playbooks(app, pack) if e.name == name), None)
    if entry is None:
        raise PlaybookError(f"no playbook {app}/{name} on disk")
    if entry.spec is None:
        raise PlaybookError(f"{app}/{name} is invalid: {entry.error}")
    if require_live:
        if not entry.spec.enabled:
            raise PlaybookError(
                f"{app}/{name} is disabled — set `enabled: true` once rehearsed"
            )
        disabled = disabled_macros(entry.spec, pack)
        if disabled:
            raise PlaybookError(
                f"{app}/{name} references disabled pack macro(s): "
                f"{', '.join(disabled)} — rehearse, then enable"
            )
    return entry.spec, pack


def resolve_inputs(spec: Playbook, provided: dict[str, str]) -> dict[str, str]:
    """Provided values against the declared inputs — the macro layer's
    resolution contract verbatim (unknown keys, missing required, defaults,
    strings only), translated to this spec's error class at the one seam."""
    try:
        return macro_inputs.resolve_inputs(spec, provided)
    except MacroError as e:
        raise PlaybookError(str(e)) from e


def readiness_warnings(spec: Playbook) -> list[str]:
    """The actionable non-blockers `playbooks check` prints: things that
    let a walk start and then quietly under-perform. Advisory by design —
    each is legal, and the author is told the cost rather than refused."""
    return _gate_word_warnings(spec)


def _gate_word_warnings(spec: Playbook) -> list[str]:
    """A gate ask that quotes no word the deterministic reply tier matches
    still works — every reply just rides the LLM tier (bounded by
    GATE_MAX_CHECKS). An ask in a language our word lists don't cover is
    legal; the author is told the cost, not refused."""
    out = []
    for node in spec.nodes:
        if not isinstance(node, AskNode):
            continue
        for key, text in (
            ("message", node.message),
            ("over_message", node.over_message),
        ):
            if text is None:
                continue
            norm = reply.normalize(text)
            if not any(w in norm for w in reply.CONFIRM_WORDS) or not any(
                w in norm for w in reply.DENY_WORDS
            ):
                out.append(
                    f"gate {node.id!r} `{key}` quotes no reply word the word "
                    "tier matches (好的/ok…, 不用/no…) — every reply will "
                    "spend an LLM check"
                )
    return out


def walk_registry(program: "Program", channel: "Channel | None") -> dict[str, Macro]:
    """The qualified dispatch registry one live walk needs — its own
    pack's macros plus the channel's. The ONE spelling of that rule:
    `session_setup` arms a real wake with it and the CLI rehearsal
    dispatches through it, so the two can never drift."""
    registry: dict[str, Macro] = {}
    if channel is not None:
        registry.update(channel.macros)
    registry.update(program.pack_macros)
    return registry


def _menu_input(i) -> str:
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
    is the overture's job (`overture.py`); this owns only the menu, the
    call, and the build. `entries` is the single source — the answer
    space is its keys, the menu a render of it, each value the parsed
    spec+pack so a positive answer activates without re-reading disk."""

    entries: dict[str, tuple[Playbook, Pack]]
    channel: Channel
    # parse_task's context: the recent daily-log entries — assembled
    # once at wake by `session_setup` (`_activation_context`).
    context: str = ""

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
            self.context,
        )

    def _menu(self) -> str:
        lines = ["Available playbooks:"]
        for ref, (spec, _pack) in self.entries.items():
            inputs = ", ".join(_menu_input(i) for i in spec.inputs)
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
        spec, pack = self.entries[outcome.out]
        try:
            values = resolve_inputs(spec, outcome.payload or {})
        except PlaybookError as e:
            log.warning("activation %s: inputs did not resolve (%s)", outcome.out, e)
            return None
        log.info("conductor: activated %s (%s)", outcome.out, outcome.reason)
        return build_program(spec, pack, values, self.channel)


def session_setup() -> "tuple[Program | None, Overture | None, dict[str, Macro]]":
    """The plugin's one wake-time setup call, fail-open throughout:
    (a resumed suspension, the overture, the hidden qualified macro
    registry). The registry spans every pack plus the channel ON THE
    OVERTURE PATH — the boot may activate any playbook, so all conductor
    hands must be dispatchable; a live program narrows it to its own pack
    + channel, and no channel means the channel-less registry only.

    A playbook on disk IS the grant: with any enabled playbook and a
    channel, the overture drives. Nothing suspended and nothing enabled
    means a plain model session, exactly as before."""
    channel = load_channel()
    program = load_suspended(channel)
    if program is not None:
        # A live program names only its own pack + the channel — the
        # full cross-pack discovery below is the overture's need, and
        # the overture is off while a program drives (a suspended walk
        # navigates to the thread on its own account).
        return program, None, walk_registry(program, channel)
    hidden: dict[str, Macro] = {}
    if channel is not None:
        hidden.update(channel.macros)
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
            if spec is None:
                continue
            # Inline macros ride the registry alongside directory macros
            # — disabled playbooks included, mirroring `qualified_pack`
            # (which carries disabled macros too; gating is the entries
            # filter below, not the dispatch table).
            hidden.update(qualified_inline(app, spec))
            if not spec.enabled or disabled_macros(spec, pack):
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
        activation=Activation(
            entries=entries,
            channel=channel,
            context=_activation_context(entries),
        ),
        prints=prints,
    )
    return None, overture, hidden


def _activation_context(entries: dict[str, tuple[Playbook, Pack]]) -> str:
    """parse_task's context, assembled once at wake, from the agent's
    OWN memory convention (never a conductor-private store): the recent
    daily-log entries — the same window the engine preloads into the
    model's wake context (`[memory] bootstrap_log_entries`), which is
    where completed purchases and suspensions are recorded, so "never
    re-run a finished task" reads off the same record the model would."""
    parts: list[str] = []
    recent = daylog.load_recent_entries(CONFIG.memory.bootstrap_log_entries)
    if recent:
        parts.append(
            "Recent daily-log entries (the assistant's own activity "
            f"record, newest first):\n{recent}"
        )
    return "\n".join(parts)
