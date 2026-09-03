"""Wake-time setup — how a `Program` comes to exist.

Two doors into a walk, both fail-open (a missing, stale, or invalid file
degrades to a normal session, never takes one down):

  - ``load_suspended`` — a walk that asked the user something resumes at
    its stored node (one-shot; ANY wake resumes, the WAIT job is just the
    alarm clock).
  - the boot — nothing suspended, but enabled playbooks exist and the
    channel pack's boot playbook is live: the conductor walks
    `channel/boot` (the thread page with its declared hands, then the
    `activate` step — `step_activate.py`), which fires ONE parse_task
    micro-call over the playbook menu and hands the program it built
    on as its baton. ``Activation`` is that last half — menu, call, and
    build — carried by the boot program for its activate step to run.

A playbook on disk IS the grant; there is nothing to pre-declare.

``session_setup`` is the plugin's single wake-time setup call
(`plugin.py` runs it behind the seam): it resolves those doors in
priority order and assembles the hidden qualified-macro registry —
every pack plus the channel on the boot path (the boot may activate
any playbook, so all conductor hands must be dispatchable); a live
program narrows it to its own pack + channel, and no channel means a
channel-less registry.
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
from physiclaw.conductor import context, reply, scaffold, suspension
from physiclaw.conductor.channel import Channel, load_channel
from physiclaw.conductor.micro import (
    NOT_A_TASK,
    PARSE_TASK,
    DecisionRequest,
    MicroOutcome,
    build_request,
)
from physiclaw.conductor.pages import (
    IOS_APP,
    MIN_ROBUST_ANCHORS,
    RESERVED_APPS,
    PagePrint,
    load_learned,
    prints_for_app,
)
from physiclaw.conductor.playbook import (
    AgentNode,
    AskNode,
    DoNode,
    Pack,
    Playbook,
    PlaybookError,
    TellNode,
    disabled_macros,
    list_apps,
    load_pack,
    qualified_all,
    qualified_inline,
    qualified_pack,
    scan_playbooks,
)
from physiclaw.conductor.playbook import require_live as playbook_require_live
from physiclaw.conductor.program import Program
from physiclaw.macros import inputs as macro_inputs
from physiclaw.macros.model import Macro, MacroError, MacroInput

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
    dry: bool = False,
    activation: "Activation | None" = None,
) -> "Program":
    """The one Program constructor call — the boot's activation, a
    resumed suspension, the CLI rehearsal, and the offline replay all
    come through here, so a program is whole at construction (channel
    and any suspended state included) and never patched up afterwards.
    `dry` (the replay, the boot) runs the walk without writing any
    record. The boot route's `activate` step needs the menu of enabled
    playbooks: the wake passes the `activation` it discovered; a tool
    stepping or rehearsing the boot gets it discovered here."""
    if activation is None and spec.activates:
        activation = activation_for(channel)
    return Program(
        spec=spec,
        values=values,
        pack_macros=qualified_pack(spec.app, pack) | qualified_inline(spec.app, spec),
        prints=prints_for_app(spec.app, decls=pack.pages) + os_prints(),
        channel=channel,
        suspended=suspended,
        landmarks=pack.landmarks,
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
    pack = load_pack(app)
    entry = next((e for e in scan_playbooks(app, pack) if e.name == name), None)
    if entry is None:
        raise PlaybookError(f"no playbook {app}/{name} on disk")
    if entry.spec is None:
        raise PlaybookError(f"{app}/{name} is invalid: {entry.error}")
    if require_live:
        playbook_require_live(entry.spec, pack)
    return entry.spec, pack


def resolve_inputs(spec: Playbook, provided: dict[str, str]) -> dict[str, str]:
    """Provided values against the declared inputs — the macro layer's
    resolution contract verbatim (unknown keys, missing required, defaults,
    strings only), translated to this spec's error class at the one seam."""
    try:
        return macro_inputs.resolve_inputs(spec, provided)
    except MacroError as e:
        raise PlaybookError(str(e)) from e


def readiness_warnings(spec: Playbook, pack: Pack) -> list[str]:
    """The actionable non-blockers `playbooks check` prints: things that
    let a walk start and then quietly under-perform. Advisory by design —
    each is legal, and the author is told the cost rather than refused."""
    return (
        _gate_word_warnings(spec)
        + _resume_warnings(spec)
        + _anchor_warnings(spec, pack)
    )


def _resume_warnings(spec: Playbook) -> list[str]:
    """A non-payment ask without `resume:`, or a tell, followed by a move
    that runs on the app: the phone is on the IM thread (or wherever
    the next wake finds it), so that move's enter check runs against
    the wrong screen and spends the page's recover hand, or hands over
    with none. Legal; the author is told. (A payment ask in that shape
    is refused at parse — consent never recovers.)"""
    out = []
    nodes = spec.nodes
    for i, n in enumerate(nodes[:-1]):
        nxt = nodes[i + 1]
        if not (isinstance(nxt, DoNode) and nxt.enter) and not (
            isinstance(nxt, AgentNode) and nxt.tools
        ):
            continue
        if isinstance(n, AskNode) and n.resume is None:
            out.append(
                f"ask {n.id!r} has no `resume:` but {nxt.id!r} runs on the app "
                "next — the enter check will read the IM thread; declare "
                "`resume:` or expect the page's recover hand to spend"
            )
        elif isinstance(n, TellNode):
            out.append(
                f"tell {n.id!r} suspends the walk, and {nxt.id!r} runs on the "
                "app on the next wake — its enter check reads whatever the "
                "phone shows then; expect the page's recover hand to spend"
            )
    return out


def _anchor_warnings(spec: Playbook, pack: Pack) -> list[str]:
    """A page the route checks that declares too few anchors to clear
    its declaration-only threshold with one missing
    (`pages.MIN_ROBUST_ANCHORS`) and has no learned geometry yet —
    legal, but one OCR miss reads it unknown and the walk spends its
    recover hand (or hands over) for nothing. Calibrated geometry lifts
    the warning: its threshold is the page's own."""
    learned = load_learned(spec.app)
    checked = {
        p
        for n in spec.nodes
        if isinstance(n, (DoNode, AgentNode))
        for p in (n.enter, n.verify)
        if p
    }
    out = []
    for name in sorted(checked):
        decl = pack.pages.get(name)
        if decl is None or name in learned or len(decl.anchors) >= MIN_ROBUST_ANCHORS:
            continue
        out.append(
            f"page {name!r} declares {len(decl.anchors)} anchor(s) and no "
            f"geometry is learned — one OCR miss reads it unknown; declare "
            f"{MIN_ROBUST_ANCHORS}+ anchors or calibrate the page"
        )
    return out


def _gate_word_warnings(spec: Playbook) -> list[str]:
    """An ask whose message quotes none of its own `yes:`/`no:` words
    still works — the user just has to guess what to reply, and any
    other wording hands the walk over. Legal; the author is told the
    cost, not refused."""
    out = []
    for node in spec.nodes:
        if not isinstance(node, AskNode):
            continue
        norm = reply.normalize(node.message)
        if not any(w in norm for w in node.yes) or not any(w in norm for w in node.no):
            out.append(
                f"ask {node.id!r} `message` quotes none of its yes/no words "
                f"({', '.join(node.yes)} / {', '.join(node.no)}) — a reply "
                "in other words hands the walk over"
            )
    return out


def menu_warnings(entries: dict[str, Playbook]) -> list[str]:
    """Across packs: two enabled playbooks whose descriptions read the
    same would sit on the activation menu as two identical lines under
    different refs — the model could only tell them apart by the ref,
    which users never type. Advisory: name the app in the description
    the way users say it. `entries` is ref → spec, the menu's own shape."""
    seen: dict[str, str] = {}
    out: list[str] = []
    for ref, spec in entries.items():
        if not spec.enabled:
            continue
        key = " ".join(spec.description.split()).casefold()
        if key in seen:
            out.append(
                f"{ref} and {seen[key]} carry the same description — the "
                "activation menu cannot tell them apart; say which app each "
                "is for, the way users name it"
            )
        else:
            seen[key] = ref
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
    rides the boot program for its `activate` step (`step_activate.py`). `entries` is the single source — the answer
    space is its keys, the menu a render of it, each value the parsed
    spec+pack so a positive answer activates without re-reading disk."""

    entries: dict[str, tuple[Playbook, Pack]]
    channel: "Channel | None"
    # parse_task's context: the recent daily-log entries — assembled
    # once at wake by `session_setup` (`_activation_context`).
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


def session_setup() -> "tuple[Program | None, dict[str, Macro]]":
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
    # The boot file is a template the user owns: materialized beside an
    # existing channel pack on the first wake, then never touched.
    scaffold.ensure_channel_boot()
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
        activation=_activation(entries, channel),
    )
    hidden.update(boot.pack_macros)  # a hand embedded in the boot route
    return boot, hidden


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


def activation_for(channel: "Channel | None") -> "Activation | None":
    """The activation a boot walked OUTSIDE a wake (a stepping tool, a
    rehearsal) runs with — the same discovery, or None when no enabled
    playbook is on disk (the step then hands over, saying so)."""
    entries, _ = discover()
    return _activation(entries, channel) if entries else None


def _activation(
    entries: "dict[str, tuple[Playbook, Pack]]", channel: "Channel | None"
) -> "Activation":
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
