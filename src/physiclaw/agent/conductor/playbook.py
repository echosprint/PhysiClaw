"""PLAYBOOK.yml → a validated `Playbook` — the model, its parser and
lints, and the ref grammar's two halves (validate + `fill_refs`).

A playbook is one app task as a small node graph: LEG nodes invoke the
pack's own macros (verified against page fingerprints), DECIDE nodes
parameterize code-owned decision calls (`calls.py`), CONFIRM parks on
the user, HUMAN_GATE holds an irreversible step until the user confirms
over the user channel. Execution lives in `program.py`; this module's
contract is the macro parser's: all-or-nothing validation with errors
that name the offending field, so a walker never meets an unknown
macro, a dangling page id, or an unrouted decision.

The grammar, top-down::

    playbook  ::= name description [enabled] [inputs] [mandate] nodes
    inputs    ::= {id: {description, [default], [example]}}   # ≤ MAX_INPUTS
    mandate   ::= {max_amount, [expires_minutes]}
    nodes     ::= [node, ...]                                 # 1–MAX_NODES
    node      ::= id "LEG" macro [with] [enter] verify
                  [compensate] [irreversible]
                | id "DECIDE" call [with] [context] [outs] on [max_visits]
                | id "CONFIRM" compose [with] message
                | id "HUMAN_GATE" gate compose [with] message [over_message] [return]
    on        ::= {out: node-id | "escalate"}                 # total over outs

Control flow: non-DECIDE nodes fall through to the next node in list
order (past the last node = done); DECIDE routes every one of its outs
explicitly. A HUMAN_GATE falls through only once the user has confirmed:
it composes and sends the full-context message over the user channel,
waits for a reply, and a micro-call judges whether the reply confirms.
Unconfirmed → wait and check again, GATE_MAX_CHECKS times in all; still
unconfirmed → the session is done and parks for the next wake-up, the
regular-session contract. `escalate` is a reserved target — the
conductor goes quiet and the model takes over. The only legal cycle is
a DECIDE self-routing its call's re-ask arm (choose_item's `scroll` —
the conductor swipes between re-asks, bounded by `max_visits`);
anything wider is the model's job, not a playbook's.

Wiring is by placeholder: `{name}` reads a declared input, `{node.field}`
reads an EARLIER decide node's declared payload field. Dotted refs are
playbook-level — they are resolved to plain strings before any macro
sees them, so pack macros keep the stock single-name template grammar.

Money is a parse-time lint, not doctrine: a node tagged
`irreversible: payment` must be unreachable except through a HUMAN_GATE,
and its playbook must carry a `mandate:`.
"""

import io
import re
from dataclasses import dataclass
from typing import Any

from physiclaw.agent.conductor import _spec
from physiclaw.agent.conductor.calls import CALLS, ESCALATE
from physiclaw.agent.conductor.pages import (
    PAGES_FILENAME,
    RESERVED_APPS,
    PageDecl,
    PagesError,
    scan_app_decls,
)
from physiclaw.agent.macros import store as macro_store
from physiclaw.agent.macros.model import Macro
from physiclaw.common import paths
from physiclaw.common.text import read_text

MAX_NODES = 20
MAX_INPUTS = 8
MAX_VISITS_CAP = 10
DEFAULT_MAX_VISITS = 3
# How many reply-check rounds a HUMAN_GATE gets before the session parks
# for the next wake-up. Fixed, not authorable: the patience budget is the
# conductor's contract with the user, not a per-playbook knob.
GATE_MAX_CHECKS = 3

# `{name}` or `{node.field}` — the playbook's own ref grammar. Dotted
# refs are deliberately NOT part of the macro template layer (its
# tokenizer rejects them); same `{{`/`}}` escapes, same
# fail-at-load-on-stray-brace rule.
REF_RE = re.compile(r"\{\{|\}\}|\{([a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*)?)\}|[{}]")
# A string that is EXACTLY one input ref (the mandate's string form).
_SOLE_REF = re.compile(r"^\{([a-z][a-z0-9_]*)\}$")

IRREVERSIBLE_CLASSES = ("payment", "send_message")
# `context:` entries are prompt-slice ids a decision call may receive.
# `inputs.*` is cross-checked against the declared inputs; `memory.*` can
# only be shape-checked at parse (the slices exist at runtime).
CONTEXT_ROOTS = ("memory", "inputs")
CONTEXT_RE = re.compile(rf"^({'|'.join(CONTEXT_ROOTS)})\.[a-z][a-z0-9_]*$")

# Reserved routing targets — graph vocabulary, not call vocabulary: a
# DECIDE arm routed here makes the conductor go quiet (the model takes
# over). Rejected as node ids so a node can never shadow the sink.
RESERVED_TARGETS = frozenset({ESCALATE})

_TOP_KEYS = {"name", "description", "enabled", "inputs", "mandate", "nodes"}
_INPUT_KEYS = {"description", "default", "example"}
_MANDATE_KEYS = {"max_amount", "expires_minutes"}
_NODE_COMMON = {"id", "type"}
_NODE_KEYS = {
    "LEG": _NODE_COMMON
    | {"macro", "with", "enter", "verify", "compensate", "irreversible"},
    "DECIDE": _NODE_COMMON | {"call", "with", "context", "outs", "on", "max_visits"},
    "CONFIRM": _NODE_COMMON | {"compose", "with", "message"},
    "HUMAN_GATE": _NODE_COMMON
    | {"gate", "compose", "with", "return", "message", "over_message"},
}

PACK_MACROS_DIRNAME = "macros"

# Shared scalar layer: macro naming/prose rules bound to this spec's
# error class; `_yaml` stays a module name so tests can patch per-module.
_yaml = _spec.yaml_loader


class PlaybookError(ValueError):
    """A playbook (or its pack wiring) is invalid. Message is user-facing:
    `physiclaw playbooks check` prints it verbatim. All-or-nothing."""


_require_str, _prose, _opt_prose, check_name = _spec.bind(PlaybookError)
INPUT_NAME_RE = _spec.INPUT_NAME_RE


@dataclass(frozen=True)
class InputRef:
    """A `{input}` reference resolved at parse time — the consumer (the
    mandate check, of all places) must never re-derive the brace grammar."""

    name: str


@dataclass(frozen=True)
class PlaybookInput:
    name: str
    description: str
    default: str | None = None  # present → optional
    example: str | None = None

    @property
    def required(self) -> bool:
        return self.default is None


@dataclass(frozen=True)
class Mandate:
    """The user-authorized spend bound, enforced by the conductor in code
    at checkout-class nodes — never trusted to any model call."""

    max_amount: float | InputRef
    expires_minutes: int | None


@dataclass(frozen=True)
class LegNode:
    id: str
    macro: str
    args: dict[str, Any]
    enter: str | None  # page id that must match before the leg
    verify: str  # page id the leg must land on — the reflector, mandatory
    compensate: str | None  # pack macro that undoes this node
    irreversible: str | None  # one of IRREVERSIBLE_CLASSES


@dataclass(frozen=True)
class DecideNode:
    id: str
    call: str
    args: dict[str, Any]
    context: tuple[str, ...]
    outs: tuple[str, ...]  # resolved: fixed for choose_item, authored for decide
    on: dict[str, str]  # out → node id | reserved target
    max_visits: int


@dataclass(frozen=True)
class ConfirmNode:
    id: str
    compose: str
    args: dict[str, Any]
    # The authored ask ({input}/{node.field} refs, parse-validated,
    # runtime-filled), sent VERBATIM and REQUIRED: only the playbook
    # author knows the user's language, so the conductor composes no
    # prose — ever.
    message: str


@dataclass(frozen=True)
class HumanGateNode:
    """Ask-and-hold: compose the full-context message, send it over the
    user channel, and fall through only on a micro-call-confirmed reply.
    GATE_MAX_CHECKS unconfirmed rounds park the session (regular wake-up
    contract) — so the node after a gate runs human-approved or not at all."""

    id: str
    gate: str  # what is being authorized, e.g. "payment"
    compose: str
    args: dict[str, Any]
    # The authored ask, REQUIRED and sent verbatim like
    # ConfirmNode.message. The consent contract is enforced as lints on
    # the template, not as appended code prose: a payment gate's
    # `message` must reference {gate.total} (the ask IS the consent
    # record), and `over_message` — the ask sent instead when the sheet
    # total exceeds the mandate cap — must reference {gate.total} AND
    # {gate.cap} (the breach must be disclosed). Both slots are
    # runtime-filled from the payment sheet.
    message: str
    over_message: str | None
    # Pack macro run after a confirmed reply to get BACK into the app
    # (the gate's ask left it for the IM thread); the next node's
    # `enter:` judges the landing. None = the walk tries the next node
    # directly and hands over if its enter check fails.
    return_macro: str | None


Node = LegNode | DecideNode | ConfirmNode | HumanGateNode


@dataclass(frozen=True)
class Playbook:
    """One validated playbook. `parse_playbook` is the only producer, so a
    Playbook is correct by construction against its pack (declared pages,
    pack macros, call declarations)."""

    app: str
    name: str
    description: str
    enabled: bool
    inputs: tuple[PlaybookInput, ...]
    mandate: Mandate | None
    nodes: tuple[Node, ...]


@dataclass(frozen=True)
class PlaybookEntry:
    """One playbook file as found on disk — parsed, or the reason it was
    excluded (the macro ScanEntry shape, named so downstream consumers get
    fields instead of tuple positions)."""

    app: str
    name: str
    spec: "Playbook | None" = None
    error: str | None = None


@dataclass(frozen=True)
class Pack:
    """What a playbook validates against: the app's declared pages and its
    private macros (name → parsed Macro, or the parse error string)."""

    app: str
    pages: dict[str, PageDecl]
    macros: dict[str, Macro]
    macro_errors: dict[str, str]


# ---------- pack loading ----------


def load_pack(app: str) -> Pack:
    """The app pack's shared assets. Page declarations may raise
    PagesError (surfaced by `check`); a broken pack macro is carried as
    its error string so the playbook referencing it fails with the cause."""
    try:
        pages = scan_app_decls(app)
    except PagesError as e:
        raise PlaybookError(f"{app}/{PAGES_FILENAME}: {e}") from e
    macros: dict[str, Macro] = {}
    errors: dict[str, str] = {}
    root = paths.playbooks_dir() / app / PACK_MACROS_DIRNAME
    if root.is_dir():
        # One scanner for both macro roots: traversal guard, dot-dir
        # convention, and the broad-except lesson live in `store.scan`.
        for entry in macro_store.scan(root):
            if entry.spec is not None:
                macros[entry.dir_name] = entry.spec
            else:
                errors[entry.dir_name] = entry.error or "invalid"
    return Pack(app=app, pages=pages, macros=macros, macro_errors=errors)


def scan_playbooks(app: str, pack: Pack | None = None) -> list[PlaybookEntry]:
    """Every `<name>.yml` in the pack (pages.yml excluded), parsed against
    the pack. Callers that already hold the Pack thread it through so its
    macros are not re-parsed."""
    root = paths.playbooks_dir() / app
    if not root.is_dir():
        return []
    if pack is None:
        pack = load_pack(app)
    out: list[PlaybookEntry] = []
    for f in sorted(root.glob("*.yml")):
        if f.name == PAGES_FILENAME:
            continue
        name = f.stem
        try:
            spec = parse_playbook(read_text(f), name, pack)
            out.append(PlaybookEntry(app=app, name=name, spec=spec))
        except Exception as e:  # broad: exclude whole, never take a session down
            out.append(
                PlaybookEntry(app=app, name=name, error=str(e) or type(e).__name__)
            )
    return out


def list_apps() -> list[str]:
    """Packs on disk (any pages.yml or *.yml present), sorted."""
    root = paths.playbooks_dir()
    if not root.is_dir():
        return []
    return sorted(
        d.name
        for d in root.iterdir()
        if d.is_dir() and not d.name.startswith(("_", ".")) and any(d.glob("*.yml"))
    )


# ---------- parsing ----------


def parse_playbook(text: str, name: str, pack: Pack) -> Playbook:
    """Parse + validate one playbook file against its pack. Raises
    PlaybookError naming the offending field; never a partial spec."""
    try:
        data = _yaml.load(io.StringIO(text))
    except Exception as e:  # broad: loader errors are not confined to YAMLError
        raise PlaybookError(f"invalid YAML: {e or type(e).__name__}") from e
    if not isinstance(data, dict):
        raise PlaybookError("a playbook must be a YAML mapping (key: value pairs)")

    unknown = sorted(set(data.keys()) - _TOP_KEYS)
    if unknown:
        raise PlaybookError(f"unknown key(s): {', '.join(map(str, unknown))}")

    declared_name = _require_str(data.get("name"), "`name`")
    if declared_name != name:
        raise PlaybookError(
            f"name {declared_name!r} must equal the filename stem {name!r}"
        )
    check_name(name, "`name`")

    description = _prose(data.get("description"), "`description`")

    enabled = data.get("enabled", True)
    if not isinstance(enabled, bool):
        raise PlaybookError("`enabled` must be true or false")

    inputs = _parse_inputs(data.get("inputs", {}))
    input_names = {i.name for i in inputs}
    mandate = (
        _parse_mandate(data["mandate"], input_names) if "mandate" in data else None
    )
    nodes = _parse_nodes(data.get("nodes"), input_names, pack)
    ids = {n.id: i for i, n in enumerate(nodes)}
    _check_graph(nodes, ids)
    _check_money(nodes, ids, mandate)
    return Playbook(
        app=pack.app,
        name=name,
        description=description,
        enabled=enabled,
        inputs=inputs,
        mandate=mandate,
        nodes=tuple(nodes),
    )


def _parse_inputs(raw: Any) -> tuple[PlaybookInput, ...]:
    if not isinstance(raw, dict):
        raise PlaybookError("`inputs` must be a mapping of input names")
    if len(raw) > MAX_INPUTS:
        raise PlaybookError(f"too many inputs ({len(raw)} > {MAX_INPUTS})")
    out: list[PlaybookInput] = []
    for name, spec in raw.items():
        if not isinstance(name, str) or not INPUT_NAME_RE.match(name):
            raise PlaybookError(
                f"input name {name!r} must be lowercase, start with a "
                "letter, and contain only letters/digits/underscores"
            )
        if not isinstance(spec, dict):
            raise PlaybookError(f"input {name!r} must be a mapping")
        unknown = sorted(set(spec.keys()) - _INPUT_KEYS)
        if unknown:
            raise PlaybookError(f"input {name!r}: unknown key(s): {', '.join(unknown)}")
        out.append(
            PlaybookInput(
                name=name,
                description=_prose(
                    spec.get("description"), f"input {name!r}: `description`"
                ),
                default=_opt_prose(spec.get("default"), f"input {name!r}: `default`"),
                example=_opt_prose(spec.get("example"), f"input {name!r}: `example`"),
            )
        )
    return tuple(out)


def _parse_mandate(raw: Any, input_names: set[str]) -> Mandate:
    where = "`mandate`"
    if not isinstance(raw, dict):
        raise PlaybookError(f"{where} must be a mapping")
    unknown = sorted(set(raw.keys()) - _MANDATE_KEYS)
    if unknown:
        raise PlaybookError(f"{where}: unknown key(s): {', '.join(map(str, unknown))}")
    if "max_amount" not in raw:
        raise PlaybookError(f"{where} needs `max_amount`")
    amount = raw["max_amount"]
    max_amount: float | InputRef
    if isinstance(amount, (int, float)) and not isinstance(amount, bool):
        if amount <= 0:
            raise PlaybookError(f"{where}.max_amount must be positive (got {amount})")
        max_amount = float(amount)
    elif isinstance(amount, str) and (m := _SOLE_REF.match(amount.strip())):
        name = m.group(1)
        if name not in input_names:
            raise PlaybookError(
                f"{where}.max_amount: placeholder {{{name}}} not declared "
                "under `inputs`"
            )
        max_amount = InputRef(name=name)
    else:
        raise PlaybookError(
            f"{where}.max_amount must be a number or exactly one `{{input}}` ref"
        )
    expires = raw.get("expires_minutes")
    if expires is not None and (
        isinstance(expires, bool) or not isinstance(expires, int) or expires <= 0
    ):
        raise PlaybookError(f"{where}.expires_minutes must be a positive whole number")
    return Mandate(max_amount=max_amount, expires_minutes=expires)


# ---------- nodes ----------


def _parse_nodes(raw: Any, input_names: set[str], pack: Pack) -> list[Node]:
    if not isinstance(raw, list) or not raw:
        raise PlaybookError("`nodes` must be a non-empty list")
    if len(raw) > MAX_NODES:
        raise PlaybookError(f"too many nodes ({len(raw)} > {MAX_NODES})")
    seen: dict[str, int] = {}
    # Grows as the single pass advances, so `{node.field}` refs are
    # defined-before-use by construction (list order).
    payload_so_far: dict[str, tuple[str, ...]] = {}
    out: list[Node] = []
    for i, node in enumerate(raw, start=1):
        parsed = _parse_node(i, node, input_names, pack, seen, payload_so_far)
        if isinstance(parsed, DecideNode):
            decl = CALLS[parsed.call]
            payload_so_far[parsed.id] = decl.payload
        out.append(parsed)
    return out


def _parse_node(
    i: int,
    node: Any,
    input_names: set[str],
    pack: Pack,
    seen: dict[str, int],
    payloads: dict[str, tuple[str, ...]],
) -> Node:
    where = f"node {i}"
    if not isinstance(node, dict):
        raise PlaybookError(f"{where} must be a mapping")
    ntype = node.get("type")
    if ntype not in _NODE_KEYS:
        raise PlaybookError(
            f"{where}: `type` must be one of {', '.join(sorted(_NODE_KEYS))} "
            f"(got {ntype!r})"
        )
    unknown = sorted(set(node.keys()) - _NODE_KEYS[ntype])
    if unknown:
        raise PlaybookError(
            f"{where}: unknown key(s) for {ntype}: {', '.join(map(str, unknown))}"
        )
    nid = _require_str(node.get("id"), f"{where}: `id`")
    check_name(nid, f"{where}: `id`")
    if nid in RESERVED_TARGETS:
        raise PlaybookError(
            f"{where}: id {nid!r} is a reserved routing target — a node "
            "must not shadow the escalation sink"
        )
    if nid in seen:
        raise PlaybookError(
            f"{where}: duplicate node id {nid!r} (node {seen[nid]} already "
            "uses it) — routing addresses nodes by id, so they must be unique"
        )
    seen[nid] = i
    where = f"node {nid!r}"

    args = node.get("with", {})
    if not isinstance(args, dict):
        raise PlaybookError(f"{where}: `with` must be a mapping of arguments")
    _check_arg_refs(args, input_names, payloads, where)

    if ntype == "LEG":
        return _parse_leg(where, nid, node, args, pack)
    if ntype == "DECIDE":
        return _parse_decide(where, nid, node, args, input_names)
    compose = _require_str(node.get("compose"), f"{where}: `compose`")
    check_name(compose, f"{where}: `compose`")
    if ntype == "CONFIRM":
        message, _ = _ask_message(where, node, "message", input_names, payloads)
        return ConfirmNode(id=nid, compose=compose, args=dict(args), message=message)
    gate = _require_str(node.get("gate"), f"{where}: `gate`")
    check_name(gate, f"{where}: `gate`")
    g_payloads = payloads
    if gate == "payment":
        # The consent slots, runtime-filled from the payment sheet and
        # available only here. (A DECIDE literally named `gate` would be
        # shadowed in this message — the money slots win, both at parse
        # and at fill.)
        g_payloads = {**payloads, "gate": ("total", "cap")}
    message, msg_refs = _ask_message(where, node, "message", input_names, g_payloads)
    over = _opt_prose(node.get("over_message"), f"{where}: `over_message`")
    if gate == "payment":
        if "gate.total" not in msg_refs:
            raise PlaybookError(
                f"{where}: a payment gate's `message` must quote the sheet "
                "total — reference {gate.total} (the ask IS the consent "
                "record)"
            )
        if over is None:
            raise PlaybookError(
                f"{where}: a payment gate needs `over_message` — the ask "
                "sent instead when the total exceeds the mandate cap; it "
                "must reference {gate.total} and {gate.cap}"
            )
        over_refs = _refs(over, f"{where}: `over_message`")
        _check_refs(over_refs, input_names, g_payloads, f"{where}: `over_message`")
        missing = sorted({"gate.total", "gate.cap"} - over_refs)
        if missing:
            raise PlaybookError(
                f"{where}: `over_message` must reference "
                + " and ".join("{" + m + "}" for m in missing)
                + " — an over-cap ask must disclose the total AND the budget"
            )
    elif over is not None:
        raise PlaybookError(f"{where}: `over_message` is only for `gate: payment`")
    return_macro = _optional_pack_macro(node, "return", where, pack)
    return HumanGateNode(
        id=nid,
        gate=gate,
        compose=compose,
        args=dict(args),
        message=message,
        over_message=over,
        return_macro=return_macro,
    )


def _ask_message(
    where: str,
    node: dict,
    key: str,
    input_names: set[str],
    payloads: dict[str, tuple[str, ...]],
) -> tuple[str, set[str]]:
    """A REQUIRED authored ask: the exact text sent to the user — only
    the author knows the user's language, so the conductor composes no
    prose around it. Refs held to the same defined-before-use rules as
    `with:` values; returned with them so gate lints can inspect."""
    text = _prose(node.get(key), f"{where}: `{key}`")
    refs = _refs(text, f"{where}: `{key}`")
    _check_refs(refs, input_names, payloads, f"{where}: `{key}`")
    return text, refs


def _optional_pack_macro(node: dict, key: str, where: str, pack: Pack) -> str | None:
    """An optional node key naming one of THIS pack's macros — the
    compensate/return idiom, one spelling."""
    name = node.get(key)
    if name is None:
        return None
    name = _require_str(name, f"{where}: `{key}`")
    if name not in pack.macros:
        raise PlaybookError(
            f"{where}: {key} macro {name!r} not found in this pack's "
            f"{PACK_MACROS_DIRNAME}/"
        )
    return name


def _parse_leg(where: str, nid: str, node: dict, args: dict, pack: Pack) -> LegNode:
    macro = _require_str(node.get("macro"), f"{where}: `macro`")
    if macro in pack.macro_errors:
        raise PlaybookError(
            f"{where}: pack macro {macro!r} is invalid: {pack.macro_errors[macro]}"
        )
    if macro not in pack.macros:
        available = ", ".join(sorted(pack.macros)) or "(none)"
        raise PlaybookError(
            f"{where}: macro {macro!r} not found in this pack's "
            f"{PACK_MACROS_DIRNAME}/ — playbooks reference only their own "
            f"pack's macros. Available: {available}"
        )
    declared = {inp.name: inp for inp in pack.macros[macro].inputs}
    unknown_args = sorted(set(args.keys()) - set(declared))
    if unknown_args:
        raise PlaybookError(
            f"{where}: `with` key(s) {', '.join(unknown_args)} are not "
            f"inputs of macro {macro!r} (declares: "
            f"{', '.join(sorted(declared)) or '(none)'})"
        )
    missing_args = sorted(
        name for name, inp in declared.items() if inp.required and name not in args
    )
    if missing_args:
        raise PlaybookError(
            f"{where}: macro {macro!r} requires input(s) "
            f"{', '.join(missing_args)} — supply them under `with`"
        )
    if "verify" not in node:
        raise PlaybookError(
            f"{where}: `verify` is required — a leg without a landing-page "
            "check cannot prove it ran (the reflector is what breaks silent "
            "per-step error compounding)"
        )
    verify = _page_ref(node.get("verify"), f"{where}: `verify`", pack)
    enter = (
        _page_ref(node.get("enter"), f"{where}: `enter`", pack)
        if "enter" in node
        else None
    )
    compensate = _optional_pack_macro(node, "compensate", where, pack)
    irreversible = node.get("irreversible")
    if irreversible is not None and irreversible not in IRREVERSIBLE_CLASSES:
        raise PlaybookError(
            f"{where}: `irreversible` must be one of "
            f"{', '.join(IRREVERSIBLE_CLASSES)} (got {irreversible!r})"
        )
    return LegNode(
        id=nid,
        macro=macro,
        args=dict(args),
        enter=enter,
        verify=verify,
        compensate=compensate,
        irreversible=irreversible,
    )


def _parse_decide(
    where: str, nid: str, node: dict, args: dict, input_names: set[str]
) -> DecideNode:
    call = node.get("call")
    if not isinstance(call, str) or call not in CALLS:
        raise PlaybookError(
            f"{where}: `call` must be one of {', '.join(sorted(CALLS))} (got {call!r})"
        )
    decl = CALLS[call]
    unknown_params = sorted(set(args.keys()) - set(decl.params))
    if unknown_params:
        raise PlaybookError(
            f"{where}: `with` key(s) {', '.join(unknown_params)} not accepted "
            f"by {call} (accepts: {', '.join(decl.params) or '(none)'})"
        )
    missing = sorted(set(decl.params) - set(args.keys()))
    if missing:
        raise PlaybookError(f"{where}: {call} requires `with.{missing[0]}`")

    context_raw = node.get("context", [])
    if not isinstance(context_raw, list):
        raise PlaybookError(f"{where}: `context` must be a list")
    context = []
    for c in context_raw:
        if not isinstance(c, str) or not CONTEXT_RE.match(c):
            raise PlaybookError(
                f"{where}: context entry {c!r} must look like "
                f"`{'.<x>` or `'.join(CONTEXT_ROOTS)}.<x>`"
            )
        root, _, member = c.partition(".")
        # `inputs.*` is checkable now; `memory.*` slices only exist at
        # runtime, so that half stays shape-only by necessity, not oversight.
        if root == "inputs" and member not in input_names:
            raise PlaybookError(
                f"{where}: context entry {c!r} names an input not declared "
                "under `inputs`"
            )
        context.append(c)

    if decl.outs:
        if "outs" in node:
            raise PlaybookError(
                f"{where}: {call} declares its outs itself "
                f"({', '.join(decl.outs)}) — remove `outs`"
            )
        outs = decl.outs
    else:
        raw_outs = node.get("outs")
        if not isinstance(raw_outs, list) or len(raw_outs) < 2:
            raise PlaybookError(
                f"{where}: {call} needs `outs` — the answers this question "
                "can have (at least 2, including `escalate`)"
            )
        outs_list = []
        for o in raw_outs:
            o = _require_str(o, f"{where}: `outs` item")
            check_name(o, f"{where}: `outs` item")
            if o in outs_list:
                raise PlaybookError(f"{where}: duplicate out {o!r}")
            outs_list.append(o)
        if ESCALATE not in outs_list:
            raise PlaybookError(
                f"{where}: `outs` must include {ESCALATE!r} — every closed "
                "choice needs the concrete escape arm"
            )
        outs = tuple(outs_list)

    on_raw = node.get("on")
    if not isinstance(on_raw, dict) or not on_raw:
        raise PlaybookError(
            f"{where}: `on` must map every out to a node id (or {ESCALATE!r})"
        )
    extra = sorted(set(on_raw.keys()) - set(outs))
    if extra:
        raise PlaybookError(
            f"{where}: `on` routes unknown out(s): {', '.join(map(str, extra))}"
        )
    unrouted = sorted(set(outs) - set(on_raw.keys()))
    if unrouted:
        raise PlaybookError(
            f"{where}: `on` must route EVERY out — missing: {', '.join(unrouted)}"
        )
    on = {
        out: _require_str(target, f"{where}: `on.{out}`")
        for out, target in on_raw.items()
    }
    for out, target in on.items():
        # A self-route is sanctioned only on the call's re-ask arm — the
        # conductor refreshes the screen (swipes) between those visits.
        # Any other self-route would re-ask the identical screen with the
        # identical prompt: a lint-free playbook that can never converge.
        if target == nid and out != decl.reask_arm:
            raise PlaybookError(
                f"{where}: `on.{out}` routes back to this node — only "
                f"{call}'s re-ask arm "
                f"({decl.reask_arm or '(none for this call)'}) may "
                "self-loop; the conductor scrolls between those re-asks"
            )

    max_visits = node.get("max_visits", DEFAULT_MAX_VISITS)
    if (
        isinstance(max_visits, bool)
        or not isinstance(max_visits, int)
        or not 1 <= max_visits <= MAX_VISITS_CAP
    ):
        raise PlaybookError(
            f"{where}: `max_visits` must be 1–{MAX_VISITS_CAP} (got {max_visits!r})"
        )
    return DecideNode(
        id=nid,
        call=call,
        args=dict(args),
        context=tuple(context),
        outs=outs,
        on=on,
        max_visits=max_visits,
    )


def _page_ref(value: Any, where: str, pack: Pack) -> str:
    """A page reference: bare `<page>` (this pack) or `<app>.<page>` where
    the app is a reserved built-in namespace. Own-pack pages must be
    declared; reserved pages resolve against built-ins later, so only the
    namespace is checked here."""
    ref = _require_str(value, where)
    if "." in ref:
        app, _, page = ref.partition(".")
        if app == pack.app:
            ref = page  # normalize the self-qualified spelling
        elif app in RESERVED_APPS:
            check_name(page, f"{where}: page")
            return ref
        else:
            raise PlaybookError(
                f"{where}: {ref!r} references app {app!r} — playbooks may "
                f"reference only their own pack's pages or the reserved "
                f"namespaces ({', '.join(sorted(RESERVED_APPS))})"
            )
    if ref not in pack.pages:
        declared = ", ".join(sorted(pack.pages)) or "(none)"
        raise PlaybookError(
            f"{where}: page {ref!r} not declared in "
            f"{pack.app}/{PAGES_FILENAME}. Declared: {declared}"
        )
    return ref


# ---------- refs (`{name}` / `{node.field}`) ----------


def _refs(text: str, where: str) -> set[str]:
    """The ref names in one string. Raises on a stray brace — a typo'd
    template must fail at load, not resolve wrong later."""
    names: set[str] = set()
    for m in REF_RE.finditer(text):
        if m.group(0) in ("{{", "}}"):
            continue
        if m.group(1) is None:
            raise PlaybookError(
                f"{where}: stray {m.group(0)!r} — use {{name}} for an input, "
                "{node.field} for an earlier decision's output, or "
                "{{ / }} for a literal brace"
            )
        names.add(m.group(1))
    return names


def _check_refs(
    refs: set[str],
    input_names: set[str],
    payloads: dict[str, tuple[str, ...]],
    where: str,
) -> None:
    for ref in sorted(refs):
        if "." in ref:
            node_id, _, fld = ref.partition(".")
            if node_id not in payloads:
                raise PlaybookError(
                    f"{where}: {{{ref}}} references node {node_id!r}, which "
                    "is not an EARLIER decide node — outputs wire forward "
                    "only, in list order"
                )
            if fld not in payloads[node_id]:
                raise PlaybookError(
                    f"{where}: {{{ref}}}: node {node_id!r} has no output "
                    f"{fld!r} (has: {', '.join(payloads[node_id]) or '(none)'})"
                )
        elif ref not in input_names:
            raise PlaybookError(
                f"{where}: placeholder {{{ref}}} not declared under `inputs`"
            )


def _check_arg_refs(
    value: Any,
    input_names: set[str],
    payloads: dict[str, tuple[str, ...]],
    where: str,
) -> None:
    if isinstance(value, str):
        _check_refs(_refs(value, where), input_names, payloads, where)
    elif isinstance(value, list):
        for v in value:
            _check_arg_refs(v, input_names, payloads, where)
    elif isinstance(value, dict):
        for v in value.values():
            _check_arg_refs(v, input_names, payloads, where)


def fill_refs(value: Any, values: dict[str, str], where: str) -> Any:
    """A `with:` value with its refs resolved from `values` — the runtime
    half of the ref grammar, beside REF_RE so no consumer re-derives the
    braces. `values` is keyed by both plain input names and dotted
    `node.field` output keys, exactly the ref spellings. Recurses into
    lists/dicts exactly as `_check_arg_refs` validates them. Raises
    PlaybookError on a ref with no value (e.g. a decision output not yet
    recorded)."""
    if isinstance(value, list):
        return [fill_refs(v, values, where) for v in value]
    if isinstance(value, dict):
        return {k: fill_refs(v, values, where) for k, v in value.items()}
    if not isinstance(value, str):
        return value

    def repl(m: re.Match[str]) -> str:
        token = m.group(0)
        if token == "{{":
            return "{"
        if token == "}}":
            return "}"
        ref = m.group(1)
        if ref is None:
            raise PlaybookError(f"{where}: stray {token!r} in {value!r}")
        if ref not in values:
            raise PlaybookError(f"{where}: no value for {{{ref}}}")
        return values[ref]

    return REF_RE.sub(repl, value)


def disabled_leg_macros(spec: Playbook, pack: Pack) -> list[str]:
    """Leg-referenced pack macros still disabled — the live-readiness
    rule: `playbooks check` warns about it, `arm` refuses on it. Safe
    unguarded access: parse rejects any LegNode whose macro is missing
    or broken, so every referenced macro is in `pack.macros`."""
    return sorted(
        {
            n.macro
            for n in spec.nodes
            if isinstance(n, LegNode) and not pack.macros[n.macro].enabled
        }
    )


# ---------- graph lints ----------


def _successors(nodes: list[Node], idx: int) -> list[str]:
    """Outgoing edges of one node: DECIDE routes explicitly; everything
    else (HUMAN_GATE included — its unconfirmed path parks the session,
    which is no edge) falls through to the next node in list order."""
    node = nodes[idx]
    if isinstance(node, DecideNode):
        return [t for t in node.on.values() if t != ESCALATE]
    if idx + 1 < len(nodes):
        return [nodes[idx + 1].id]
    return []


def _check_graph(nodes: list[Node], ids: dict[str, int]) -> None:
    # Targets resolve. (A non-DECIDE can never self-route: its one
    # successor is the NEXT node by construction, and ids are unique —
    # so "only a DECIDE may self-loop" needs no check of its own.)
    for node in nodes:
        if isinstance(node, DecideNode):
            for out, target in node.on.items():
                if target in RESERVED_TARGETS:
                    continue
                if target not in ids:
                    raise PlaybookError(
                        f"node {node.id!r}: `on.{out}` routes to unknown "
                        f"node {target!r} (or a reserved target: "
                        f"{', '.join(sorted(RESERVED_TARGETS))})"
                    )
    # Reachability from the first node.
    reachable: set[str] = set()
    stack = [nodes[0].id]
    while stack:
        nid = stack.pop()
        if nid in reachable:
            continue
        reachable.add(nid)
        stack.extend(_successors(nodes, ids[nid]))
    unreachable = [n.id for n in nodes if n.id not in reachable]
    if unreachable:
        raise PlaybookError(
            f"unreachable node(s): {', '.join(unreachable)} — every node "
            "must be reachable from the first"
        )
    _check_acyclic(nodes, ids)


def _check_acyclic(nodes: list[Node], ids: dict[str, int]) -> None:
    """DFS cycle detection, self-edges excluded (the one sanctioned,
    bounded loop). Wider cycles are the model's job."""
    visiting: set[str] = set()
    done: set[str] = set()

    def visit(nid: str) -> None:
        visiting.add(nid)
        for target in _successors(nodes, ids[nid]):
            if target == nid:
                continue
            if target in visiting:
                raise PlaybookError(
                    f"cycle via {nid!r} → {target!r} — playbooks are "
                    "forward-only except a DECIDE's bounded re-ask "
                    "self-loop; wider loops belong to the model"
                )
            if target not in done:
                visit(target)
        visiting.discard(nid)
        done.add(nid)

    for n in nodes:
        if n.id not in done:
            visit(n.id)


def _check_money(
    nodes: list[Node], ids: dict[str, int], mandate: Mandate | None
) -> None:
    """A `payment`-tagged node must be unreachable except through a
    HUMAN_GATE, and its playbook must carry a mandate. Walked over the
    real edges with a gate-passed flag, so the guarantee is structural."""
    money = [
        n.id for n in nodes if isinstance(n, LegNode) and n.irreversible == "payment"
    ]
    if not money:
        return
    if mandate is None:
        raise PlaybookError(
            f"node(s) {', '.join(money)} are irreversible: payment — the "
            "playbook must declare a `mandate` (max_amount)"
        )
    # Adjacency, not just reachability: the conductor reads the payment
    # sheet AT the gate (the ask quotes its total) and fires the leg as
    # the gate's fall-through — a node in between would desynchronize
    # the consent from the sheet. The gate must also DECLARE the class
    # (`gate: payment`), so the runtime keys off the declaration.
    for i, n in enumerate(nodes):
        if isinstance(n, LegNode) and n.irreversible == "payment":
            prev = nodes[i - 1] if i > 0 else None
            if not (isinstance(prev, HumanGateNode) and prev.gate == "payment"):
                raise PlaybookError(
                    f"node {n.id!r} (irreversible: payment) must DIRECTLY "
                    "follow a HUMAN_GATE with `gate: payment` — the gate "
                    "reads the sheet its ask quotes, and consent must not "
                    "desynchronize from it"
                )
    # DFS over (node, gate_passed) states; a money node seen with
    # gate_passed False is reachable around the human.
    seen: set[tuple[str, bool]] = set()
    stack: list[tuple[str, bool]] = [(nodes[0].id, False)]
    while stack:
        nid, gated = stack.pop()
        if (nid, gated) in seen:
            continue
        seen.add((nid, gated))
        node = nodes[ids[nid]]
        if isinstance(node, LegNode) and node.irreversible == "payment" and not gated:
            raise PlaybookError(
                f"node {nid!r} (irreversible: payment) is reachable without "
                "passing a HUMAN_GATE — money always goes through the human"
            )
        passed = gated or isinstance(node, HumanGateNode)
        for target in _successors(nodes, ids[nid]):
            stack.append((target, passed))
