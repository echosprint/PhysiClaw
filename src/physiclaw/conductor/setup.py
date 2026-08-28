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

A playbook on disk IS the grant; there is nothing to pre-declare. (There
used to be a third door, ``armed.json``, naming the playbook to run next
— the overture retired it.)

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

from physiclaw.common.listing import Screen
from physiclaw.common.text import read_text
from physiclaw.conductor import memory, reply, scaffold, suspension
from physiclaw.conductor.channel import Channel, load_channel
from physiclaw.conductor.ledger import check_ledger_value
from physiclaw.conductor.micro import (
    LIST_INPUT_MARK,
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
    ConfirmNode,
    DecideNode,
    HumanGateNode,
    LegNode,
    Pack,
    Playbook,
    PlaybookError,
    disabled_leg_macros,
    list_apps,
    load_pack,
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
            app,
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
    app: str,
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
        app=app,
        spec=spec,
        values=values,
        pack_macros=qualified_pack(app, pack),
        prints=prints_for_app(app),
        channel=channel,
        suspended=suspended,
    )


def load_spec(
    app: str, name: str, *, require_live: bool = True
) -> tuple[Playbook, Pack]:
    """The parsed playbook and its pack. `require_live` additionally holds
    it to what a real wake needs — enabled, with every leg macro enabled —
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
        disabled = disabled_leg_macros(entry.spec, pack)
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
    return (
        _memory_slice_warnings(spec)
        + _gate_word_warnings(spec)
        + _gate_reentry_warnings(spec)
    )


def _memory_slice_warnings(spec: Playbook) -> list[str]:
    """A declared `memory.<slug>` slice with no matching `## <slug>`
    section on THIS device runs empty (fail-closed — memory.py owns the
    contract; this is its check-time projection)."""
    slugs = sorted(
        {
            entry.partition(".")[2]
            for node in spec.nodes
            if isinstance(node, DecideNode)
            for entry in node.context
            if entry.startswith("memory.")
        }
    )
    if not slugs:
        return []
    sections = memory.read_sections()
    have = frozenset().union(*(tokens for tokens, _ in sections)) if sections else ()
    missing = [slug for slug in slugs if slug.casefold() not in have]
    if not missing:
        return []
    return [
        f"memory slice(s) {', '.join(missing)}: no `## <slug>` section in "
        "memory.md on this device — those decisions run without memory "
        "context (fail-closed)"
    ]


def _gate_word_warnings(spec: Playbook) -> list[str]:
    """A gate ask that quotes no word the deterministic reply tier matches
    still works — every reply just rides the LLM tier (bounded by
    GATE_MAX_CHECKS). An ask in a language our word lists don't cover is
    legal; the author is told the cost, not refused."""
    out = []
    for node in spec.nodes:
        if not isinstance(node, HumanGateNode):
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


def _gate_reentry_warnings(spec: Playbook) -> list[str]:
    """After any gate's confirmed reply the phone shows the IM thread. The
    payment case is a parse ERROR; for other gates a fall-through leg with
    no `enter:` (and no gate `return:`) runs its macro blind off the
    thread — the verify check catches the landing, but the gestures
    already happened on the messenger."""
    out = []
    nodes = spec.nodes
    for i, node in enumerate(nodes):
        if not isinstance(node, HumanGateNode) or node.return_macro is not None:
            continue
        nxt = nodes[i + 1] if i + 1 < len(nodes) else None
        if nxt is None or isinstance(nxt, (ConfirmNode, HumanGateNode)):
            continue
        if not (isinstance(nxt, LegNode) and nxt.enter is not None):
            out.append(
                f"gate {node.id!r}: no `return:` and {nxt.id!r} declares no "
                "`enter:` — after a confirmed reply the first action runs "
                "off the IM thread"
            )
    return out


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
            values = resolve_inputs(spec, outcome.payload or {})
            check_ledger_value(spec, values)
        except PlaybookError as e:
            log.warning("activation %s: inputs did not resolve (%s)", outcome.out, e)
            return None
        log.info("conductor: activated %s (%s)", outcome.out, outcome.reason)
        return build_program(app, spec, pack, values, self.channel)


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
    hidden: dict[str, Macro] = {}
    channel = load_channel()
    if channel is not None:
        hidden.update(channel.macros)
    program = load_suspended(channel)
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
