"""PLAYBOOK.yml → a validated `Playbook` — the model, its parser and
lints, and the ref grammar's two halves (validate + `fill_refs`).

A playbook is one app task as a small node graph: LEG nodes invoke the
pack's own macros (verified against page fingerprints), DECIDE nodes
parameterize code-owned decision calls (`calls.py`), CONFIRM suspends on
the user, HUMAN_GATE holds an irreversible step until the user confirms
over the user channel. Execution lives in `program.py`; this module's
contract is the macro parser's: all-or-nothing validation with errors
that name the offending field, so a walker never meets an unknown
macro, a dangling page id, or an unrouted decision.

The grammar, top-down::

    playbook  ::= name description [enabled] [inputs] [mandate]
                  [parse_context] nodes
    inputs    ::= {id: {description, [default], [example], [kind]}}  # ≤ MAX_INPUTS
    mandate   ::= {max_amount, [expires_minutes]}
    nodes     ::= [node, ...]                                 # 1–MAX_NODES
    node      ::= id "LEG" macro [with] [enter] verify
                  [compensate] [irreversible]
                | id "DECIDE" call [with] [context] [outcomes] routes [max_visits]
                | id "RECONCILE" page
                | id "CONFIRM" compose [with] message
                | id "HUMAN_GATE" gate compose [with] message [over_message]
                  [return] [revise]
    macro     ::= name                    # macros/<name>/ — a directory macro
                | {[inputs] steps}        # inline — MACRO.yml grammar minus
                                          # name/description/enabled
                                          # (`compensate:`/`return:` take the same
                                          #  form; both dispatch with no arguments,
                                          #  so required inputs are rejected there
                                          #  — either spelling)
    routes    ::= {outcome: node-id | "escalate"}             # total over outcomes

An inline macro is single-use by construction — an anonymous body has
no name for another site to reference: its name is synthesized
`<playbook>.<node-id>` (a leg) or `<playbook>.<node-id>.<role>`
(a compensate/return body) — dot-joined, a spelling no directory macro
or node id can take, so the pack-wide dispatch namespace stays
collision-free by construction. It is enabled iff its playbook is, and
it records stats/runs under that synthesized name like any pack macro.
The one role that must stay a directory macro is the rescue ladder's
`open`: it is pack-level — resolved by name with no node to hang a body
on. Grammar boundary: everything under an inline `macro:` IS a macro
(single-name `{x}` templates, fed by the node's `with:`), everything
outside stays dotted — the same split as the file boundary, moved to
the key.

The ledger stack (`kind: list` input + `next_item` loop + RECONCILE +
gate `revise:`) is one unit — `_check_ledger` holds its pieces
together: the ONE list input is the buying list (desired state), the
`next_item` DECIDE closes the shopping loop over it (its `next` arm is
the one sanctioned backward edge; body nodes read loop-scoped
`{item.query}`/`{item.qty}`), RECONCILE diffs the cart (observed state)
against the ledger in code, and a payment gate's `revise:` routes a
"yes, but change it" reply back into the loop instead of handing over.

Control flow: non-DECIDE nodes fall through to the next node in list
order (past the last node = done); DECIDE routes every one of its outcomes
explicitly. A HUMAN_GATE falls through only once the user has confirmed:
it composes and sends the full-context message over the user channel,
waits for a reply, and a micro-call judges whether the reply confirms.
Unconfirmed → wait and check again, GATE_MAX_CHECKS times in all; still
unconfirmed → the session is done and suspends for the next wake-up, the
regular-session contract. `escalate` is a reserved target — the
conductor goes quiet and the model takes over. The only legal cycles
are a DECIDE self-routing its call's re-ask arm (choose_item's
`scroll` — the conductor swipes between re-asks, bounded by
`max_visits`) and the ledger loop's declared backward arm
(`CallDecl.loop_arm`, terminating by item consumption); anything wider
is the model's job, not a playbook's.

Wiring is by placeholder, and every ref is dotted — the same
`<root>.<name>` rule as page references: `{inputs.name}` reads a
declared input, `{node.field}` reads an EARLIER decide node's declared
payload field, `{item.field}` the ledger loop's current item. A bare
`{name}` is a load error. Page references
(`enter:`/`verify:`/`page:`) are always `<root>.<page>` over a closed
root set: `pages.<name>` points at the pack file's own `pages:`
section, `ios.<page>`/`channel.<page>` reach the reserved built-ins —
one required spelling, so every ref names its section. Dotted refs are
playbook-level — they are resolved to plain strings before any macro
sees them, so pack macros keep the stock single-name template grammar.

Money is a parse-time lint, not doctrine: a node tagged
`irreversible: payment` must be unreachable except through a HUMAN_GATE,
and its playbook must carry a `mandate:`.
"""

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from physiclaw.common import paths
from physiclaw.common.paths import PACK_FILENAME
from physiclaw.conductor import _spec
from physiclaw.conductor.calls import CALLS, ESCALATE, LEDGER_FIELDS, NEXT_ITEM
from physiclaw.conductor.pages import (
    RESERVED_APPS,
    PageDecl,
    PagesError,
    parse_pages_data,
)
from physiclaw.macros import store as macro_store
from physiclaw.macros.model import Macro, MacroError
from physiclaw.macros.parse import parse_inline_macro

MAX_NODES = 20
MAX_INPUTS = 8
MAX_VISITS_CAP = 10
DEFAULT_MAX_VISITS = 3
# How many reply-check rounds a HUMAN_GATE gets before the session suspends
# for the next wake-up, and how many "yes, but change it" revision cycles
# one gate absorbs before the model takes over. Fixed, not authorable:
# the patience budget is the conductor's contract with the user, not a
# per-playbook knob.
GATE_MAX_CHECKS = 3
GATE_MAX_REVISIONS = 2

# `{root.name}` — the playbook's own ref grammar, always dotted
# (`inputs.` / `item.` / an earlier decide node's id). Dotted refs are
# deliberately NOT part of the macro template layer (its tokenizer
# rejects them); same `{{`/`}}` escapes, same
# fail-at-load-on-stray-brace rule.
REF_RE = re.compile(r"\{\{|\}\}|\{([a-z][a-z0-9_]*\.[a-z][a-z0-9_]*)\}|[{}]")
# A string that is EXACTLY one input ref (the mandate's string form).
_SOLE_REF = re.compile(r"^\{inputs\.([a-z][a-z0-9_]*)\}$")

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

# The ref grammar's global roots — rejected as node ids so a decide's
# `{node.field}` outputs can never shadow `{inputs.*}` or the ledger's
# `{item.*}`. (`gate` is not global: it exists only inside a payment
# gate's own messages, where the money slots win.)
RESERVED_REF_ROOTS = frozenset({"inputs", "item"})

_INPUT_KEYS = {"description", "default", "example", "kind"}
_INPUT_KINDS = ("scalar", "list")
_MANDATE_KEYS = {"max_amount", "expires_minutes"}
_NODE_COMMON = {"id", "type"}
_NODE_KEYS = {
    "LEG": _NODE_COMMON
    | {"macro", "with", "enter", "verify", "compensate", "irreversible"},
    "DECIDE": _NODE_COMMON
    | {"call", "with", "context", "outcomes", "routes", "max_visits"},
    "RECONCILE": _NODE_COMMON | {"page"},
    "CONFIRM": _NODE_COMMON | {"compose", "with", "message"},
    "HUMAN_GATE": _NODE_COMMON
    | {"gate", "compose", "with", "return", "message", "over_message", "revise"},
}

PACK_MACROS_DIRNAME = "macros"

_PLAY_KEYS = {"description", "enabled", "inputs", "mandate", "nodes", "parse_context"}


class PlaybookError(ValueError):
    """A playbook (or its pack wiring) is invalid. Message is user-facing:
    `physiclaw playbooks check` prints it verbatim. All-or-nothing."""


_require_str, _prose, _opt_prose, check_name = _spec.bind(PlaybookError)
INPUT_NAME_RE = _spec.INPUT_NAME_RE


@dataclass(frozen=True)
class InputRef:
    """A `{inputs.name}` reference resolved at parse time — the consumer
    (the mandate check, of all places) must never re-derive the brace
    grammar."""

    name: str


@dataclass(frozen=True)
class PlaybookInput:
    name: str
    description: str
    default: str | None = None  # present → optional
    example: str | None = None
    # "scalar" (a template string) or "list" — the buying-list ledger:
    # its VALUE is a JSON array string ([{query, qty}], validated at
    # arm/activation), consumed by the next_item loop as {item.*} refs,
    # never referenced as an {inputs.name} template.
    kind: str = "scalar"

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
    outcomes: tuple[str, ...]  # resolved: fixed for choose_item, authored for decide
    routes: dict[str, str]  # out → node id | reserved target
    max_visits: int


@dataclass(frozen=True)
class ReconcileNode:
    """Desired-state convergence, zero LLM: on the cart page, diff the
    cart rows (observed) against the ledger (desired) in code and act —
    quantity via the row's +/− steppers, removal is minus-to-zero, a
    missing picked item re-enters the shopping loop. Cart rows matching
    no ledger item are LEFT ALONE: they may be the user's own, and the
    conductor never destroys what it cannot attribute to itself."""

    id: str
    page: str  # this pack's cart page — re-read after every action


@dataclass(frozen=True)
class ConfirmNode:
    id: str
    compose: str
    args: dict[str, Any]
    # The authored ask ({inputs.name}/{node.field} refs, parse-validated,
    # runtime-filled), sent VERBATIM and REQUIRED: only the playbook
    # author knows the user's language, so the conductor composes no
    # prose — ever.
    message: str


@dataclass(frozen=True)
class HumanGateNode:
    """Ask-and-hold: compose the full-context message, send it over the
    user channel, and fall through only on a micro-call-confirmed reply.
    GATE_MAX_CHECKS unconfirmed rounds suspend the session (regular wake-up
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
    # Ledger playbooks only: a "yes, but change it" reply routes HERE
    # (linted: the next_item node) after revise_list updates the ledger
    # — shop the additions, reconcile the changes, and re-ask with the
    # fresh total. None = a revise hands over (non-list behavior).
    revise: str | None


Node = LegNode | DecideNode | ReconcileNode | ConfirmNode | HumanGateNode


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
    # Memory slices the ACTIVATION's parse_task receives (fail-closed,
    # the decide `context:` contract) — e.g. `memory.shopping_prefs`.
    parse_context: tuple[str, ...] = ()
    # The embedded macros — LEG bodies (`<playbook>.<node>`) and
    # compensate/return bodies (`<playbook>.<node>.<role>`). The node
    # field holds the same synthesized name, so dispatch is name-keyed
    # either way. `qualified_inline` is the registry door.
    inline_macros: dict[str, Macro] = field(default_factory=dict)

    # The ledger stack's two anchors, derived HERE so lint (playbook)
    # and runtime (program) can never disagree on what counts as "the"
    # ledger input or "the" loop — `_check_ledger` guarantees at most
    # one of each.

    @property
    def ledger_input(self) -> PlaybookInput | None:
        return next((i for i in self.inputs if i.kind == "list"), None)

    @property
    def loop(self) -> DecideNode | None:
        return next(
            (
                n
                for n in self.nodes
                if isinstance(n, DecideNode) and CALLS[n.call].loop_arm is not None
            ),
            None,
        )


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
    # The `playbooks:` map, raw — parsed per-entry by `scan_playbooks`
    # so one broken walk excludes itself, never the pack.
    playbook_docs: dict = field(default_factory=dict)


def qualified_macro(app: str, name: str) -> str:
    """The qualified `app/name` dispatch key — the ONE spelling of the
    convention the run_macro handler resolves (user macro names can
    never contain "/", so no collision). Lives beside `Pack`, the owner
    of macro dicts — every pack site (channel included) consumes it."""
    return f"{app}/{name}"


def macro_app(name: str) -> str:
    """The app half of a qualified dispatch key — `qualified_macro`'s
    inverse, kept beside it so the "/" convention has one spelling.
    "" for an unqualified name (user macros never carry an app)."""
    app, sep, _ = name.partition("/")
    return app if sep else ""


def qualified_pack(app: str, pack: Pack) -> dict[str, Macro]:
    """A pack's macros under their qualified dispatch keys."""
    return {qualified_macro(app, n): m for n, m in pack.macros.items()}


def qualified_inline(app: str, spec: Playbook) -> dict[str, Macro]:
    """A playbook's inline macros under their qualified dispatch keys —
    `qualified_pack`'s sibling for the hands that live in the playbook
    itself. Every registry a walk can dispatch through takes both."""
    return {qualified_macro(app, n): m for n, m in spec.inline_macros.items()}


# ---------- pack loading ----------


def load_pack(app: str) -> Pack:
    """The app pack, whole, from its one spec file: validated meta, page
    declarations, the raw `playbooks:` docs (parsed per-entry by
    `scan_playbooks`), and the recorded macros. A broken pack macro is
    carried as its error string so the playbook referencing it fails
    with the cause."""
    doc = _spec.load_pack_doc(app, PlaybookError)
    if doc is None:
        raise PlaybookError(f"no pack {app!r} on disk (missing {PACK_FILENAME})")
    _check_pack_meta(doc, app)
    try:
        pages = parse_pages_data(doc.get("pages"), app)
    except PagesError as e:
        raise PlaybookError(f"{app}/{PACK_FILENAME} `pages`: {e}") from e
    raw_playbooks = doc.get("playbooks") or {}
    if not isinstance(raw_playbooks, dict):
        raise PlaybookError("`playbooks` must be a mapping of name → playbook")
    macros: dict[str, Macro] = {}
    errors: dict[str, str] = {}
    root = paths.pack_root(app) / PACK_MACROS_DIRNAME
    if root.is_dir():
        # One scanner for both macro roots: traversal guard, dot-dir
        # convention, and the broad-except lesson live in `store.scan`.
        for entry in macro_store.scan(root):
            if entry.spec is not None:
                macros[entry.dir_name] = entry.spec
            else:
                errors[entry.dir_name] = entry.error or "invalid"
    return Pack(
        app=app,
        pages=pages,
        macros=macros,
        macro_errors=errors,
        playbook_docs=dict(raw_playbooks),
    )


def _check_pack_meta(doc: dict, app: str) -> None:
    """The manifest half of the pack file: `name` equals the directory,
    `description` is real prose, `placeholders` (install-time constants,
    validated here so `check` catches a malformed map before install
    prompts read it) is name → {description, [example]}."""
    declared = _require_str(doc.get("name"), "`name`")
    if declared != app:
        raise PlaybookError(f"name {declared!r} must equal the pack directory {app!r}")
    if app == "pages":
        raise PlaybookError(
            "a pack cannot be named 'pages' — it is the page-reference root"
        )
    _prose(doc.get("description"), "`description`")
    ph = doc.get("placeholders")
    if ph is None:
        return
    if not isinstance(ph, dict):
        raise PlaybookError("`placeholders` must be a mapping of TOKEN → spec")
    for key, spec in ph.items():
        where = f"placeholder {key!r}"
        if not isinstance(spec, dict) or set(spec) - {"description", "example"}:
            raise PlaybookError(f"{where} must be a {{description, example}} mapping")
        _prose(spec.get("description"), f"{where}: `description`")


def scan_playbooks(app: str, pack: Pack | None = None) -> list[PlaybookEntry]:
    """Every entry of the pack's `playbooks:` map, parsed against the
    pack. Callers that already hold the Pack thread it through so the
    spec file and macros are not re-read."""
    if pack is None:
        if not (paths.pack_root(app) / PACK_FILENAME).exists():
            return []
        pack = load_pack(app)
    out: list[PlaybookEntry] = []
    for name, data in pack.playbook_docs.items():
        name = str(name)
        try:
            spec = _parse_playbook_data(data, name, pack)
            out.append(PlaybookEntry(app=app, name=name, spec=spec))
        except Exception as e:  # broad: exclude whole, never take a session down
            out.append(
                PlaybookEntry(app=app, name=name, error=str(e) or type(e).__name__)
            )
    return out


def list_apps() -> list[str]:
    """Packs across the search path (the `paths.playbooks_dirs` layering),
    sorted — a PLAYBOOK.yml marks a pack."""
    return sorted(paths.marked_subdirs(paths.playbooks_dirs(), PACK_FILENAME))


# ---------- parsing ----------


def parse_playbook(text: str, name: str, pack: Pack) -> Playbook:
    """One playbook given as YAML text — the text-shaped door tests and
    tooling use; the live path is `scan_playbooks` over the pack file's
    `playbooks:` map. Raises PlaybookError naming the offending field;
    never a partial spec."""
    data = _spec.load_yaml(text, PlaybookError)
    return _parse_playbook_data(data, name, pack)


def _parse_playbook_data(data, name: str, pack: Pack) -> Playbook:
    """One entry of the `playbooks:` map → a validated Playbook. The map
    key IS the name — there is no inner `name:` key to drift from it."""
    if not isinstance(data, dict):
        raise PlaybookError("a playbook must be a YAML mapping (key: value pairs)")

    unknown = sorted(set(map(str, data.keys())) - _PLAY_KEYS)
    if unknown:
        raise PlaybookError(f"unknown key(s): {', '.join(unknown)}")

    check_name(name, "playbook key")

    description = _prose(data.get("description"), "`description`")

    enabled = data.get("enabled", True)
    if not isinstance(enabled, bool):
        raise PlaybookError("`enabled` must be true or false")

    raw_ctx = data.get("parse_context", [])
    if not isinstance(raw_ctx, list):
        raise PlaybookError("`parse_context` must be a list")
    parse_context = []
    for c in raw_ctx:
        # The decide `context:` grammar (CONTEXT_RE), memory root only —
        # activation runs before any walk, so `inputs.*` cannot exist.
        if not isinstance(c, str) or not (
            CONTEXT_RE.match(c) and c.startswith("memory.")
        ):
            raise PlaybookError(
                f"`parse_context` entry {c!r} must look like `memory.<slug>` "
                "(activation runs before any walk, so only memory slices exist)"
            )
        parse_context.append(c)

    inputs = _parse_inputs(data.get("inputs", {}))
    # Refs resolve SCALAR inputs only: a `kind: list` value is a JSON
    # ledger, consumed by the next_item loop as {item.*} — splicing it
    # into a template would paste raw JSON into a gesture argument.
    ref_names = {i.name for i in inputs if i.kind == "scalar"}
    has_ledger = any(i.kind == "list" for i in inputs)
    mandate = _parse_mandate(data["mandate"], ref_names) if "mandate" in data else None
    inline: dict[str, Macro] = {}
    resolve = _macro_resolver(name, pack, inline)
    nodes = _parse_nodes(
        data.get("nodes"), ref_names, pack, resolve, has_ledger=has_ledger
    )
    ids = {n.id: i for i, n in enumerate(nodes)}
    _check_graph(nodes, ids)
    _check_money(nodes, ids, mandate)
    _check_ledger(nodes, ids, inputs)
    return Playbook(
        app=pack.app,
        name=name,
        description=description,
        enabled=enabled,
        inputs=inputs,
        mandate=mandate,
        nodes=tuple(nodes),
        parse_context=tuple(parse_context),
        inline_macros=inline,
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
        kind = spec.get("kind", "scalar")
        if kind not in _INPUT_KINDS:
            raise PlaybookError(
                f"input {name!r}: `kind` must be one of {', '.join(_INPUT_KINDS)} "
                f"(got {kind!r})"
            )
        out.append(
            PlaybookInput(
                name=name,
                description=_prose(
                    spec.get("description"), f"input {name!r}: `description`"
                ),
                default=_opt_prose(spec.get("default"), f"input {name!r}: `default`"),
                example=_opt_prose(spec.get("example"), f"input {name!r}: `example`"),
                kind=kind,
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
                f"{where}.max_amount: placeholder {{inputs.{name}}} not "
                "declared under `inputs`"
            )
        max_amount = InputRef(name=name)
    else:
        raise PlaybookError(
            f"{where}.max_amount must be a number or exactly one `{{inputs.name}}` ref"
        )
    expires = raw.get("expires_minutes")
    if expires is not None and (
        isinstance(expires, bool) or not isinstance(expires, int) or expires <= 0
    ):
        raise PlaybookError(f"{where}.expires_minutes must be a positive whole number")
    return Mandate(max_amount=max_amount, expires_minutes=expires)


# ---------- nodes ----------


def _parse_nodes(
    raw: Any,
    input_names: set[str],
    pack: Pack,
    resolve: "_MacroResolve",
    *,
    has_ledger: bool,
) -> list[Node]:
    if not isinstance(raw, list) or not raw:
        raise PlaybookError("`nodes` must be a non-empty list")
    if len(raw) > MAX_NODES:
        raise PlaybookError(f"too many nodes ({len(raw)} > {MAX_NODES})")
    seen: dict[str, int] = {}
    # Grows as the single pass advances, so `{node.field}` refs are
    # defined-before-use by construction (list order). With a ledger,
    # the loop-scoped `{item.*}` refs are available everywhere as a
    # pseudo-payload (the loop closer sits AFTER the body in list
    # order, so per-body scoping cannot ride the single pass); no node
    # can shadow it — `item` is a reserved ref root.
    payload_so_far: dict[str, tuple[str, ...]] = (
        {"item": LEDGER_FIELDS} if has_ledger else {}
    )
    out: list[Node] = []
    for i, node in enumerate(raw, start=1):
        parsed = _parse_node(i, node, input_names, pack, seen, payload_so_far, resolve)
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
    resolve: "_MacroResolve",
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
    if nid in RESERVED_REF_ROOTS:
        raise PlaybookError(
            f"{where}: id {nid!r} is a reserved ref root — {{inputs.*}} and "
            "{item.*} always read the declared inputs and the ledger item"
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
        return _parse_leg(where, nid, node, args, pack, resolve)
    if ntype == "DECIDE":
        return _parse_decide(where, nid, node, args, input_names)
    if ntype == "RECONCILE":
        page = _page_ref(node.get("page"), f"{where}: `page`", pack, own_pack_only=True)
        return ReconcileNode(id=nid, page=page)
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
    return_macro = _optional_pack_macro(node, "return", where, nid, resolve)
    revise = node.get("revise")
    if revise is not None:
        revise = _require_str(revise, f"{where}: `revise`")
        check_name(revise, f"{where}: `revise`")
    return HumanGateNode(
        id=nid,
        gate=gate,
        compose=compose,
        args=dict(args),
        message=message,
        over_message=over,
        return_macro=return_macro,
        revise=revise,
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


# The shape `_macro_resolver` returns; a string annotation at the use
# sites keeps the forward reference cheap.
_MacroResolve = Callable[..., Macro]


def _macro_resolver(
    playbook: str, pack: Pack, inline: dict[str, Macro]
) -> _MacroResolve:
    """The name-or-inline resolution every macro-carrying slot shares —
    a LEG's `macro:`, a leg's `compensate:`, a gate's `return:`. ONE
    home for the whole idiom: the synthesized-name rule
    (`<playbook>.<node>[.<role>]`, dot-joined so it can never collide
    with a directory macro — `check_name` rejects dots), the MacroError
    framing, the inline registry, and the directory validation (a
    broken macro reports its cause, an unknown one lists what exists) —
    so the three slots can never drift. Returns the resolved Macro; its
    `.name` is the dispatch name either way (a directory macro's name
    IS its directory)."""

    def resolve(raw: Any, where: str, nid: str, role: str | None = None) -> Macro:
        slot = role or "macro"
        if isinstance(raw, dict):
            mname = f"{playbook}.{nid}" + (f".{role}" if role else "")
            try:
                spec = parse_inline_macro(raw, mname)
            except MacroError as e:
                raise PlaybookError(f"{where}: inline `{slot}`: {e}") from e
            inline[mname] = spec
            return spec
        if raw is not None and not isinstance(raw, str):
            raise PlaybookError(
                f"{where}: `{slot}` must be a pack macro name or an inline "
                "mapping with `steps:`"
            )
        mname = _require_str(raw, f"{where}: `{slot}`")
        if mname in pack.macro_errors:
            raise PlaybookError(
                f"{where}: pack macro {mname!r} is invalid: {pack.macro_errors[mname]}"
            )
        if mname not in pack.macros:
            available = ", ".join(sorted(pack.macros)) or "(none)"
            raise PlaybookError(
                f"{where}: {slot} {mname!r} not found in this pack's "
                f"{PACK_MACROS_DIRNAME}/ — playbooks reference only their own "
                f"pack's macros. Available: {available}"
            )
        return pack.macros[mname]

    return resolve


def _optional_pack_macro(
    node: dict, key: str, where: str, nid: str, resolve: _MacroResolve
) -> str | None:
    """An optional node key holding one of THIS pack's macros — the
    compensate/return idiom, resolved exactly like a LEG's `macro:`
    (directory name or embedded body). One lint of its own, on the
    RESOLVED macro so it holds for either spelling: both roles dispatch
    with no arguments, so a required input could only abort at run time
    — right after a confirmed ask, at the worst moment."""
    raw = node.get(key)
    if raw is None:
        return None
    spec = resolve(raw, where, nid, key)
    required = sorted(i.name for i in spec.inputs if i.required)
    if required:
        raise PlaybookError(
            f"{where}: `{key}` macro {spec.name!r} requires input(s) "
            f"{', '.join(required)} — the walk dispatches {key} with no "
            "arguments"
        )
    return spec.name


def _parse_leg(
    where: str,
    nid: str,
    node: dict,
    args: dict,
    pack: Pack,
    resolve: _MacroResolve,
) -> LegNode:
    spec = resolve(node.get("macro"), where, nid)
    macro = spec.name
    declared = {inp.name: inp for inp in spec.inputs}
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
    compensate = _optional_pack_macro(node, "compensate", where, nid, resolve)
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
    if decl.deterministic and ("context" in node or "max_visits" in node):
        raise PlaybookError(
            f"{where}: {call} is deterministic — `context`/`max_visits` "
            "would be dead config (no prompt, no visit budget)"
        )
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

    if decl.outcomes:
        if "outcomes" in node:
            raise PlaybookError(
                f"{where}: {call} declares its outcomes itself "
                f"({', '.join(decl.outcomes)}) — remove `outcomes`"
            )
        outcomes = decl.outcomes
    else:
        raw_outcomes = node.get("outcomes")
        if not isinstance(raw_outcomes, list) or len(raw_outcomes) < 2:
            raise PlaybookError(
                f"{where}: {call} needs `outcomes` — the answers this question "
                "can have (at least 2, including `escalate`)"
            )
        outcomes_list = []
        for o in raw_outcomes:
            o = _require_str(o, f"{where}: `outcomes` item")
            check_name(o, f"{where}: `outcomes` item")
            if o in outcomes_list:
                raise PlaybookError(f"{where}: duplicate outcome {o!r}")
            outcomes_list.append(o)
        if ESCALATE not in outcomes_list:
            raise PlaybookError(
                f"{where}: `outcomes` must include {ESCALATE!r} — every closed "
                "choice needs the concrete escape arm"
            )
        outcomes = tuple(outcomes_list)

    routes_raw = node.get("routes")
    if not isinstance(routes_raw, dict) or not routes_raw:
        raise PlaybookError(
            f"{where}: `routes` must map every out to a node id (or {ESCALATE!r})"
        )
    extra = sorted(set(routes_raw.keys()) - set(outcomes))
    if extra:
        raise PlaybookError(
            f"{where}: `routes` names unknown outcome(s): {', '.join(map(str, extra))}"
        )
    unrouted = sorted(set(outcomes) - set(routes_raw.keys()))
    if unrouted:
        raise PlaybookError(
            f"{where}: `routes` must route EVERY outcome — missing: {', '.join(unrouted)}"
        )
    routes = {
        out: _require_str(target, f"{where}: `routes.{out}`")
        for out, target in routes_raw.items()
    }
    for out, target in routes.items():
        # A self-route is sanctioned only on the call's re-ask arm — the
        # conductor refreshes the screen (swipes) between those visits.
        # Any other self-route would re-ask the identical screen with the
        # identical prompt: a lint-free playbook that can never converge.
        if target == nid and out != decl.reask_arm:
            raise PlaybookError(
                f"{where}: `routes.{out}` routes back to this node — only "
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
        outcomes=outcomes,
        routes=routes,
        max_visits=max_visits,
    )


def _page_ref(
    value: Any, where: str, pack: Pack, *, own_pack_only: bool = False
) -> str:
    """A page reference — always `<root>.<page>`, one uniform shape over
    a closed root vocabulary: `pages` (this pack file's own `pages:`
    section) or a reserved built-in namespace (`ios`/`channel`). One
    spelling, required: a bare name is rejected with the fix spelled
    out, so every ref a reader meets points at its section. Own-pack
    pages must be declared; reserved pages resolve against built-ins
    later, so only the namespace is checked here. `own_pack_only` closes
    the reserved door (RECONCILE acts on the page — a built-in it cannot
    act on is out)."""
    ref = _require_str(value, where)
    if "." not in ref:
        raise PlaybookError(
            f"{where}: page references are written `pages.<name>` — write pages.{ref}"
        )
    app, _, page = ref.partition(".")
    if app == "pages":
        ref = page
    elif own_pack_only:
        raise PlaybookError(
            f"{where}: must be THIS pack's page (`pages.<name>`) — the "
            "node re-reads and acts on it, so a reserved namespace "
            "cannot serve"
        )
    elif app in RESERVED_APPS:
        check_name(page, f"{where}: page")
        return ref
    else:
        raise PlaybookError(
            f"{where}: {ref!r} — page references are `pages.<name>` "
            f"(this pack) or a reserved namespace "
            f"({', '.join(sorted(RESERVED_APPS))}).<page>"
        )
    if ref not in pack.pages:
        declared = ", ".join(sorted(pack.pages)) or "(none)"
        raise PlaybookError(
            f"{where}: page {ref!r} not declared under `pages:` in "
            f"{pack.app}/{PACK_FILENAME}. Declared: {declared}"
        )
    return ref


# ---------- refs (`{inputs.name}` / `{node.field}`) ----------


def _refs(text: str, where: str) -> set[str]:
    """The ref names in one string. Raises on a stray brace — a typo'd
    template must fail at load, not resolve wrong later."""
    names: set[str] = set()
    for m in REF_RE.finditer(text):
        if m.group(0) in ("{{", "}}"):
            continue
        if m.group(1) is None:
            raise PlaybookError(
                f"{where}: stray {m.group(0)!r} — every ref is dotted: "
                "{inputs.name} for an input, {node.field} for an earlier "
                "decision's output, {{ / }} for a literal brace"
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
        root, _, fld = ref.partition(".")
        if root == "inputs":
            if fld not in input_names:
                raise PlaybookError(f"{where}: {{{ref}}} not declared under `inputs`")
        elif root not in payloads:
            raise PlaybookError(
                f"{where}: {{{ref}}} references node {root!r}, which "
                "is not an EARLIER decide node — outputs wire forward "
                "only, in list order"
            )
        elif fld not in payloads[root]:
            raise PlaybookError(
                f"{where}: {{{ref}}}: node {root!r} has no output "
                f"{fld!r} (has: {', '.join(payloads[root]) or '(none)'})"
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
    braces. `values` is keyed by the dotted ref spellings themselves
    (`inputs.name`, `node.field`, `item.field`). Recurses into
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
    """Referenced pack macros still disabled — the live-readiness rule:
    `playbooks check` warns about it and the overture will not offer such
    a playbook at all. Covers legs,
    `compensate:`, and gate `return:` (all dispatch at walk time). Safe
    unguarded access: parse validated every directory name against
    `pack.macros`."""
    named: set[str] = set()
    for n in spec.nodes:
        if isinstance(n, LegNode):
            named.add(n.macro)
            if n.compensate is not None:
                named.add(n.compensate)
        elif isinstance(n, HumanGateNode) and n.return_macro is not None:
            named.add(n.return_macro)
    # One rule, no inline special case: each name resolves through the
    # merged view, and an inline macro is enabled by construction — its
    # gate is the playbook's own `enabled:` — so only directory names
    # can report.
    return sorted(
        m for m in named if not (spec.inline_macros.get(m) or pack.macros[m]).enabled
    )


# ---------- graph lints ----------


def _successors(nodes: list[Node], idx: int) -> list[str]:
    """Outgoing edges of one node: DECIDE routes explicitly; everything
    else (HUMAN_GATE included — its unconfirmed path suspends the session,
    which is no edge) falls through to the next node in list order."""
    node = nodes[idx]
    if isinstance(node, DecideNode):
        return [t for t in node.routes.values() if t != ESCALATE]
    if idx + 1 < len(nodes):
        return [nodes[idx + 1].id]
    return []


def _check_graph(nodes: list[Node], ids: dict[str, int]) -> None:
    # Targets resolve. (A non-DECIDE can never self-route: its one
    # successor is the NEXT node by construction, and ids are unique —
    # so "only a DECIDE may self-loop" needs no check of its own.)
    for node in nodes:
        if isinstance(node, DecideNode):
            for out, target in node.routes.items():
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
    """DFS cycle detection. Excluded as sanctioned: self-edges (the
    bounded re-ask loop) and next_item's `next` arm — the ledger loop,
    which terminates by construction (each `next` consumes a pending
    item; `_check_ledger` pins it backward). Wider cycles are the
    model's job."""
    visiting: set[str] = set()
    done: set[str] = set()

    def visit(nid: str) -> None:
        visiting.add(nid)
        node = nodes[ids[nid]]
        loop_edge = None
        if isinstance(node, DecideNode):
            arm = CALLS[node.call].loop_arm
            if arm is not None:
                loop_edge = node.routes.get(arm)
        for target in _successors(nodes, ids[nid]):
            if target == nid or target == loop_edge:
                continue
            if target in visiting:
                raise PlaybookError(
                    f"cycle via {nid!r} → {target!r} — playbooks are "
                    "forward-only except a DECIDE's bounded re-ask "
                    "self-loop and next_item's ledger loop; wider loops "
                    "belong to the model"
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
    """An irreversible-tagged node must be unreachable except through a
    HUMAN_GATE, and a payment playbook must carry a mandate. Walked over
    the real edges with a gate-passed flag, so the guarantee is
    structural."""
    if not any(isinstance(n, LegNode) and n.irreversible for n in nodes):
        return
    money = [
        n.id for n in nodes if isinstance(n, LegNode) and n.irreversible == "payment"
    ]
    if money and mandate is None:
        raise PlaybookError(
            f"node(s) {', '.join(money)} are irreversible: payment — the "
            "playbook must declare a `mandate` (max_amount)"
        )
    # Adjacency, not just reachability: the conductor reads the payment
    # sheet AT the gate (the ask quotes its total) and fires the leg as
    # the gate's fall-through — a node in between would desynchronize
    # the consent from the sheet. The gate must also DECLARE the class
    # (`gate: payment`), so the runtime keys off the declaration. Other
    # irreversible classes take any gate: the human said go, adjacently.
    for i, n in enumerate(nodes):
        if not (isinstance(n, LegNode) and n.irreversible):
            continue
        prev = nodes[i - 1] if i > 0 else None
        if n.irreversible == "payment":
            if not (isinstance(prev, HumanGateNode) and prev.gate == "payment"):
                raise PlaybookError(
                    f"node {n.id!r} (irreversible: payment) must DIRECTLY "
                    "follow a HUMAN_GATE with `gate: payment` — the gate "
                    "reads the sheet its ask quotes, and consent must not "
                    "desynchronize from it"
                )
            # After the ask the phone shows the IM thread; the walk must
            # re-enter the app on a VERIFIED page before money fires, or
            # the fire-time predicates would read the thread — where the
            # ask itself quotes the consented total.
            if prev.return_macro is None and n.enter is None:
                raise PlaybookError(
                    f"gate {prev.id!r} → {n.id!r}: after the ask the phone "
                    "is on the IM thread — declare `return:` on the gate "
                    "or `enter:` on the payment leg"
                )
        elif not isinstance(prev, HumanGateNode):
            raise PlaybookError(
                f"node {n.id!r} (irreversible: {n.irreversible}) must "
                "DIRECTLY follow a HUMAN_GATE — an irreversible act runs "
                "human-approved or not at all"
            )
        # The gate's fall-through is the leg's ONLY door: a routed
        # in-edge (a DECIDE arm, the ledger loop) would enter it with
        # another gate's consent still bound — or with none.
        for other in nodes:
            if isinstance(other, DecideNode) and n.id in other.routes.values():
                raise PlaybookError(
                    f"node {other.id!r} routes to {n.id!r} "
                    f"(irreversible: {n.irreversible}) — an irreversible "
                    "leg is entered ONLY as its own gate's fall-through"
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
        passed = gated or (isinstance(node, HumanGateNode) and node.gate == "payment")
        for target in _successors(nodes, ids[nid]):
            stack.append((target, passed))


def _check_ledger(
    nodes: list[Node], ids: dict[str, int], inputs: tuple[PlaybookInput, ...]
) -> None:
    """The ledger stack is all-or-nothing: the ONE `kind: list` input,
    the ONE next_item loop consuming it, and the optional RECONCILE /
    gate `revise:` that both lean on the loop. Half a stack is a walk
    that stalls at runtime — rejected here instead."""
    list_inputs = [i.name for i in inputs if i.kind == "list"]
    if len(list_inputs) > 1:
        raise PlaybookError(
            f"at most one `kind: list` input (the ledger) — got: "
            f"{', '.join(list_inputs)}"
        )
    loops = [
        n
        for n in nodes
        if isinstance(n, DecideNode) and CALLS[n.call].loop_arm is not None
    ]
    if len(loops) > 1:
        raise PlaybookError(
            f"at most one {NEXT_ITEM} node — RECONCILE re-shop and gate "
            f"`revise:` both route through THE loop"
        )
    if loops and not list_inputs:
        raise PlaybookError(
            f"node {loops[0].id!r}: {NEXT_ITEM} walks the ledger — declare "
            "a `kind: list` input"
        )
    if list_inputs and not loops:
        raise PlaybookError(
            f"input {list_inputs[0]!r} is `kind: list` but no {NEXT_ITEM} "
            "node consumes it"
        )
    if not loops:
        stranded = next((n for n in nodes if isinstance(n, ReconcileNode)), None)
        if stranded is not None:
            raise PlaybookError(
                f"node {stranded.id!r}: RECONCILE needs the {NEXT_ITEM} loop "
                "— a missing item re-enters it to be shopped again"
            )
    else:
        loop = loops[0]
        idx = ids[loop.id]
        arm = CALLS[loop.call].loop_arm
        assert arm is not None
        for out, target in loop.routes.items():
            if out == arm:
                if target == ESCALATE or ids[target] >= idx:
                    raise PlaybookError(
                        f"node {loop.id!r}: `on.{out}` must route BACKWARD "
                        "to the loop body head (the closer sits at the "
                        "loop's bottom)"
                    )
            elif target != ESCALATE and ids[target] <= idx:
                raise PlaybookError(
                    f"node {loop.id!r}: `on.{out}` must route FORWARD — "
                    "the ledger is spent"
                )
    for n in nodes:
        if isinstance(n, HumanGateNode) and n.revise is not None:
            if not loops or n.revise != loops[0].id:
                raise PlaybookError(
                    f"node {n.id!r}: `revise` must target the {NEXT_ITEM} "
                    "node — a revision re-enters the loop for the added "
                    "items, then reconciles the rest"
                )
            if n.return_macro is None:
                raise PlaybookError(
                    f"node {n.id!r}: `revise` needs `return:` — the reply "
                    "was read on the IM thread, and the loop must re-enter "
                    "the app before it can shop"
                )
