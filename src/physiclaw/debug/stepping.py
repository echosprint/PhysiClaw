"""Stepping — one node of a playbook, or one gesture of a macro, per
call: the debugger's core. `physiclaw playbooks step` and `macros run`
are typer around it for an agent at a terminal; the studio's Playbook
and Macro tabs are HTML around it for a person in a browser. Neither
skin holds any of this logic.

Two granularities, deliberately separate. A playbook node runs with
its page checks (enter, verify, the declared recover hand), so it is
stepped whole, or its FIRST gesture only (`start_at`/`stop_after`) —
from the second gesture on, the screen is no longer the enter page and
the walk would recover instead of tapping. A macro is stepped gesture
by gesture with no page checks at all (`run_macro`, over the macro
runner's own step range), which is how a hand that misses is fixed:
run one gesture, look, tap by hand to recover, edit the step, run it
again. Pack macros (directory and inline) are addressable by their
qualified `app/name`, the same name a walk dispatches.

A step rebuilds the walk from a persisted position (`Program.state`,
the suspension projection), runs the node at the cursor on the live
phone through the rehearsal loop (`conductor.rehearsal.walk` with
`Program.step_one` set), and persists where the cursor stands when the
loop pauses. The user channel is virtual: an ask's send still runs on
the phone, but its observation is rewritten from `debug/thread.json`
(`interceptor.FakeChannel`), so the reply an ask reads is the one
staged here. No model ever takes over on a handover — the position
stays on the node so the author can edit the pack and step again.

The position file is a tool's scratch under `debug/` (never a wake's
contract: `conductor.suspension` stays the only cross-wake file).
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

from physiclaw.common import paths
from physiclaw.common.logger import write_json_atomic
from physiclaw.common.text import read_text
from physiclaw.conductor import rehearsal
from physiclaw.conductor.hooks import Emit, McpCaller, Observe, OnExchange
from physiclaw.conductor.playbook import (
    ActivateNode,
    AgentNode,
    AskNode,
    DoNode,
    Node,
    Playbook,
    PlaybookError,
    TellNode,
    qualified_all,
    qualified_macro,
)
from physiclaw.conductor.specfile import SpecError
from physiclaw.conductor.suspension import SUSPENDED_SCHEMA
from physiclaw.contract.dto import ToolCall
from physiclaw.debug import thread as vthread
from physiclaw.macros.model import Macro, MacroError, MacroInput
from physiclaw.macros.steps import Step as MacroStep

log = logging.getLogger(__name__)

STATE_FILENAME = "walk.json"

# What a step ends as — one closed vocabulary both skins read.
PAUSED = "paused"  # the node settled; the cursor moved on
SUSPENDED = "suspended"  # an ask ran out of patience: stage a reply, step again
STOPPED = "stopped"  # handed over or errored; the position is kept on the node
COMPLETED = "completed"  # past the last node; the position is cleared

# The route's entry kinds, as the panel labels them.
KIND_START = "start"
KIND_DO = "do"
KIND_AGENT = "agent"
KIND_ASK = "ask"
KIND_TELL = "tell"
KIND_ACTIVATE = "activate"


def state_path() -> Path:
    return paths.debug_dir() / STATE_FILENAME


# ---------- the route, for the eye ----------


@dataclass(frozen=True)
class NodeInfo:
    """One compiled move as the panel lists it. `macro` is the qualified
    name of the hand a `do`/`start` runs (or an ask's resume), the name
    the Macro tab steps gesture by gesture."""

    id: str
    kind: str
    enter: str = ""
    verify: str = ""
    irreversible: str | None = None
    macro: str | None = None

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "kind": self.kind,
            "enter": self.enter,
            "verify": self.verify,
            "irreversible": self.irreversible,
            "macro": self.macro,
        }


def node_info(app: str, node: Node) -> NodeInfo:
    if isinstance(node, DoNode):
        return NodeInfo(
            node.id,
            KIND_START if node.start else KIND_DO,
            node.enter,
            node.verify,
            node.irreversible,
            qualified_macro(app, node.macro),
        )
    if isinstance(node, AgentNode):
        return NodeInfo(node.id, KIND_AGENT, node.enter, node.verify, node.irreversible)
    if isinstance(node, AskNode):
        resume = qualified_macro(app, node.resume) if node.resume else None
        return NodeInfo(node.id, KIND_ASK, node.enter, macro=resume)
    if isinstance(node, ActivateNode):
        return NodeInfo(node.id, KIND_ACTIVATE, node.enter)
    assert isinstance(node, TellNode)
    return NodeInfo(node.id, KIND_TELL)


def route(spec: Playbook) -> list[NodeInfo]:
    return [node_info(spec.app, n) for n in spec.nodes]


def node_index(spec: Playbook, node: str) -> int:
    ids = [n.id for n in spec.nodes]
    if node not in ids:
        raise PlaybookError(
            f"no node {node!r} in {spec.app}/{spec.name}; nodes: {', '.join(ids)}"
        )
    return ids.index(node)


def node_label(spec: Playbook, idx: int) -> str:
    total = len(spec.nodes)
    if idx >= total:
        return f"(end, {total}/{total})"
    return f"{spec.nodes[idx].id} ({idx + 1}/{total})"


# ---------- the position ----------


@dataclass(frozen=True)
class Position:
    """Where a stepped walk stands — the stored projection, read for
    the eye (`status`) and for the next step."""

    app: str
    name: str
    idx: int
    node: str  # the label of the node at the cursor, "(i/N)" included
    values: dict[str, str]
    outputs: dict[str, str]
    awaiting: bool
    quoted: float | None
    consented: float | None
    staged: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "app": self.app,
            "playbook": self.name,
            "idx": self.idx,
            "node": self.node,
            "values": dict(self.values),
            "outputs": dict(self.outputs),
            "awaiting": self.awaiting,
            "quoted": self.quoted,
            "consented": self.consented,
            "staged": list(self.staged),
        }

    def describe(self) -> str:
        return "\n".join(
            [
                f"{self.app}/{self.name}: at node {self.node}",
                f"  inputs: {self.values}",
                f"  outputs: {self.outputs}",
                f"  gate: awaiting={self.awaiting} quoted={self.quoted} "
                f"consented={self.consented}",
                f"  staged replies: {self.staged}",
            ]
        )


def read_state() -> dict | None:
    """The one stored position, whoever's — None when missing or
    unreadable. `load_state` narrows it to a playbook."""
    p = state_path()
    if not p.exists():
        return None
    try:
        data = json.loads(read_text(p))
    except Exception:
        return None
    return data if data.get("schema") == SUSPENDED_SCHEMA else None


def _owned(state: dict | None, app: str, name: str) -> dict | None:
    if state is None or state.get("app") != app or state.get("playbook") != name:
        return None
    return state


def load_state(app: str, name: str) -> dict | None:
    """The stored position for THIS playbook, or None (missing,
    unreadable, or another playbook's — a new ref starts a new walk)."""
    return _owned(read_state(), app, name)


def save_state(state: dict) -> None:
    write_json_atomic(state_path(), state)


def reset(app: str, name: str) -> str:
    """Forget the stored position. Returns the line to print."""
    p = state_path()
    if p.exists():
        p.unlink()
        return f"{app}/{name}: position cleared — the next step starts the route over"
    return f"{app}/{name}: no stored position"


def position(spec: Playbook, state: dict, staged: list[str] | None = None) -> Position:
    """`staged` is the virtual thread's queue; read here when the caller
    has not already."""
    return Position(
        app=spec.app,
        name=spec.name,
        idx=int(state["idx"]),
        node=node_label(spec, int(state["idx"])),
        values=dict(state.get("values") or {}),
        outputs=dict(state.get("outputs") or {}),
        awaiting=bool(state.get("awaiting")),
        quoted=state.get("quoted"),
        consented=state.get("consented"),
        staged=list(vthread.load().staged if staged is None else staged),
    )


def status(app: str, name: str) -> Position | None:
    """The stored position, or None when the next step starts the route."""
    from physiclaw.conductor import setup as conductor_setup

    state = load_state(app, name)
    if state is None:
        return None
    spec, _pack = conductor_setup.load_spec(app, name, require_live=False)
    return position(spec, state)


def catalog() -> list[dict]:
    """Every pack that has playbooks, each playbook with its inputs,
    its route, and its stored position — the studio panel's whole
    model, JSON-shaped. A pack that fails to load reports its error
    instead of hiding."""
    from physiclaw.conductor.playbook import list_apps, load_pack, scan_playbooks

    stored = read_state()
    staged = list(vthread.load().staged)
    packs: list[dict] = []
    for app in list_apps():
        try:
            pack = load_pack(app)
        except Exception as e:
            packs.append({"app": app, "error": str(e), "playbooks": []})
            continue
        entries: list[dict] = []
        for entry in scan_playbooks(app, pack):
            item: dict = {
                "name": entry.name,
                "enabled": bool(entry.spec and entry.spec.enabled),
                "error": entry.error,
            }
            spec = entry.spec
            if spec is not None:
                item["description"] = spec.description
                item["inputs"] = [_input_item(i) for i in spec.inputs]
                item["nodes"] = [n.to_dict() for n in route(spec)]
                state = _owned(stored, app, entry.name)
                item["position"] = (
                    position(spec, state, staged).to_dict() if state else None
                )
            entries.append(item)
        if entries:
            packs.append({"app": app, "playbooks": entries})
    return packs


def _input_item(i: MacroInput) -> dict:
    return {
        "name": i.name,
        "description": i.description,
        "default": i.default,
        "example": i.example,
    }


# ---------- one step ----------


@dataclass(frozen=True)
class StepResult:
    """What a step ended as, where the walk stands, and the one line a
    skin prints."""

    outcome: str
    message: str
    position: Position | None  # None once the walk completed

    def to_dict(self) -> dict:
        return {
            "outcome": self.outcome,
            "message": self.message,
            "position": self.position.to_dict() if self.position else None,
        }


async def step(
    app: str,
    name: str,
    mcp: McpCaller,
    *,
    values: dict[str, str] | None = None,
    at: str | None = None,
    to: str | None = None,
    outputs: dict[str, str] | None = None,
    reply: str | None = None,
    start_at: str = "",
    stop_after: str = "",
    emit: Emit,
    emit_warn: Emit | None = None,
    verbose: bool = False,
    observe: Observe | None = None,
    raw: bool = False,
    on_exchange: OnExchange | None = None,
) -> StepResult:
    """Rebuild the walk at its stored position, run one node (or up to
    `to`, `end` for the route's end), persist where it stands, and say
    what happened. `values` are a fresh walk's inputs (refused mid-walk:
    reset to start over); `at` puts the cursor on a node first; `outputs`
    seed a pure-text agent's answer, and the cursor fast-forwards past
    the settled prefix so no call fires for it (an explicit `at` wins);
    `reply` stages the user's next reply (for the boot, the request the
    thread shows); `start_at`/`stop_after` narrow
    the first macro of this invocation to a step range. `mcp` is an
    open client with `call_tool`; `emit` receives every progress line,
    `emit_warn` the readiness advisories, `observe(call, blocks)` every
    real result before the virtual channel rewrites it; `raw` emits
    every model round-trip as sent and received, and `on_exchange`
    receives each one as a record (`rehearsal.exchanges`).
    Raises PlaybookError on a bad ref, node, or input."""
    from physiclaw.conductor import channel as channel_mod
    from physiclaw.conductor import setup as conductor_setup
    from physiclaw.debug.interceptor import FakeChannel

    emit_warn = emit_warn or emit
    spec, pack = conductor_setup.load_spec(app, name, require_live=False)
    channel = channel_mod.load_channel()
    # The boot's menu of enabled playbooks: every pack on disk, read
    # once per invocation rather than per node built.
    activation = conductor_setup.activation_for(channel) if spec.activates else None
    state = load_state(app, name)
    if state is None:
        # A new walk's position at the route top — `Program.state`, the
        # one projection, off a program that never advanced.
        state = conductor_setup.build_program(
            spec,
            pack,
            conductor_setup.resolve_inputs(spec, values or {}),
            channel,
            dry=True,
            activation=activation,
        ).state()
        for line in conductor_setup.readiness_warnings(spec, pack):
            emit_warn(line)
        # The virtual thread opens with the user's words — the input that
        # reads like a message — so an ask's reply reading has a thread.
        # The boot has no inputs and reads the thread first: `--reply`
        # IS the user's message there, staged for its opening read.
        said = state["values"].get("user_said", "")
        if not said and reply is not None and spec.activates:
            said, reply = reply, None
        vthread.seed(said or f"(stepping {app}/{name})", [])
        emit(
            "user channel is virtual while stepping: an ask reads the reply you "
            "stage (its send still runs on the phone)"
        )
    elif values:
        raise PlaybookError(
            "inputs are fixed for a walk in progress — reset to start over"
        )
    state["outputs"] = {**state.get("outputs", {}), **(outputs or {})}
    if at is not None:
        # A jump abandons whatever the cursor was holding — an ask left
        # awaiting its reply — and forgets the jumped-to node's own
        # recorded answer, so a re-run really re-runs it (the walk
        # starts past any SETTLED pure-text node).
        state["idx"] = node_index(spec, at)
        state["awaiting"] = False
        state["outputs"] = {
            k: v for k, v in state["outputs"].items() if not k.startswith(f"{at}.")
        }
    if reply is not None:
        vthread.stage([reply])
    to_idx = None
    if to is not None:
        to_idx = len(spec.nodes) if to == "end" else node_index(spec, to)
    fake = FakeChannel()
    opts = {k: v for k, v in (("start_at", start_at), ("stop_after", stop_after)) if v}
    first = True
    while True:
        program = conductor_setup.build_program(
            spec,
            pack,
            state["values"],
            channel,
            suspended=state,
            dry=True,
            activation=activation,
        )
        program.step_one = True
        registry = conductor_setup.walk_registry(program, channel)
        emit(f"node {node_label(spec, program.idx)}")
        outcome = await rehearsal.walk(
            program,
            registry,
            mcp,
            emit=emit,
            transform=lambda call, blocks: fake.intercept(call, True, blocks),
            macro_opts=opts,
            verbose=verbose,
            observe=observe,
            raw=raw,
            on_exchange=on_exchange,
            unlock=first,  # the lock-screen preamble once per invocation
        )
        first = False
        opts = {}  # a step range narrows the first node's macro only
        if program.outcome == "completed":
            reset(app, name)
            return StepResult(
                COMPLETED, f"{app}/{name}: playbook complete — position cleared", None
            )
        # The projection is whole wherever the walk stopped — an ask
        # awaiting its reply, the cursor after a pause or a handover.
        state = program.state()
        save_state(state)
        here = position(spec, state)
        if outcome == rehearsal.WALK_SUSPENDED:
            return StepResult(
                SUSPENDED,
                f"ask sent — waiting for the reply at {here.node}: stage one and "
                "step again",
                here,
            )
        if outcome != rehearsal.WALK_PAUSED:
            return StepResult(
                STOPPED,
                f"{outcome}; position kept at {here.node} — fix, then step again",
                here,
            )
        if to_idx is None or here.idx > to_idx:
            return StepResult(PAUSED, f"paused — next node is {here.node}", here)


# ---------- macros: one gesture at a time ----------


def _macro_item(name: str, source: str, spec: Macro | None, error: str | None) -> dict:
    item: dict = {
        "name": name,
        "source": source,
        "enabled": bool(spec and spec.enabled),
        "error": error,
    }
    if spec is not None:
        item["description"] = spec.description
        item["inputs"] = [_input_item(i) for i in spec.inputs]
        item["steps"] = [
            {"name": st.name, "tool": st.tool, "detail": _step_detail(st)}
            for st in spec.steps
        ]
    return item


def _step_detail(st: MacroStep) -> str:
    """The one thing to show beside a step: its target label, its text,
    or its wait."""
    args = getattr(st, "args", None) or {}
    label = args.get("label") or args.get("text")
    if isinstance(label, list):
        label = " / ".join(map(str, label))
    if label:
        return str(label)
    seconds = getattr(st, "seconds", None)
    return f"{seconds}s" if seconds is not None else ""


def _pack_macros(app: str) -> tuple[dict[str, Macro], dict[str, str]]:
    """A pack's macros under their qualified names — directory macros
    and every playbook's inline bodies, what a walk of that pack can
    dispatch — and the directory macros that failed to parse, by the
    same names. Raises MacroError when the pack itself does not load."""
    from physiclaw.conductor.playbook import load_pack

    try:
        pack = load_pack(app)
    except (SpecError, OSError) as e:
        raise MacroError(f"pack {app!r}: {e}") from e
    errors = {qualified_macro(app, n): e for n, e in pack.macro_errors.items()}
    return qualified_all(app, pack), errors


def macro_catalog() -> list[dict]:
    """Every macro a rehearsal can run by name: the user's own, then each
    pack's under `app/name` — with inputs and steps, the Macro tab's
    whole model. A pack or macro that fails to load reports its error."""
    from physiclaw.conductor.playbook import list_apps
    from physiclaw.macros import store as macro_store

    out = [_macro_item(e.dir_name, "user", e.spec, e.error) for e in macro_store.scan()]
    for app in list_apps():
        try:
            macros, errors = _pack_macros(app)
        except MacroError as e:
            out.append(_macro_item(app, app, None, str(e)))
            continue
        out.extend(
            _macro_item(name, app, spec, None) for name, spec in sorted(macros.items())
        )
        out.extend(
            _macro_item(name, app, None, err) for name, err in sorted(errors.items())
        )
    return out


def find_macro(name: str) -> Macro:
    """The macro a name addresses: a user macro's directory name, or a
    pack macro's qualified `app/name` (an inline body is
    `app/<playbook>.<move>`). Raises MacroError, naming the reason."""
    from physiclaw.conductor.playbook import macro_app
    from physiclaw.macros import store as macro_store

    app = macro_app(name)
    if app:
        macros, errors = _pack_macros(app)
        spec = macros.get(name)
        if spec is None:
            if name in errors:
                raise MacroError(f"{name}: {errors[name]}")
            raise MacroError(
                f"no macro {name!r} in pack {app!r}; it has: "
                f"{', '.join(sorted(macros)) or '(none)'}"
            )
        return spec
    for e in macro_store.scan():
        if e.dir_name == name:
            if e.spec is None:
                raise MacroError(f"{name}: {e.error}")
            return e.spec
    raise MacroError(f"no macro named {name!r}")


async def run_macro(
    spec: Macro,
    values: dict[str, str],
    mcp: McpCaller,
    *,
    start_at: str = "",
    stop_after: str = "",
    emit: Emit,
    observe: Observe | None = None,
    caller: str = "cli",
) -> dict:
    """Run a macro's step range through the macro runner — no page
    checks, the runner's own guards only — emitting its header and step
    log, and showing `observe` the result view. Returns {ok, message,
    run_id}. Raises MacroError before any gesture on bad inputs or an
    unknown step name (recorded as bad_input by the runner)."""
    from physiclaw.common import gesture_vocab, verdict
    from physiclaw.macros import runner as macro_runner

    result = await macro_runner.run_and_record(
        spec, values, mcp, caller=caller, start_at=start_at, stop_after=stop_after
    )
    text = verdict.action_text(result.blocks)
    lines = rehearsal.result_lines(text, False)
    for line in lines:
        emit(line)
    if observe is not None:
        observe(
            ToolCall(
                id="macro", name=gesture_vocab.RUN_MACRO, arguments={"name": spec.name}
            ),
            result.blocks,
        )
    return {
        "ok": result.ok,
        "message": lines[0] if lines else f"macro {spec.name}: no result",
        "run_id": result.run_id,
    }
