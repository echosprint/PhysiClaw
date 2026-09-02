"""PLAYBOOK.yml → a validated `Playbook` — the model, its parser and
lints, and the ref grammar's two halves (validate + `fill_refs`).

A playbook is one app task written as a ROUTE: a top-down alternation
of waypoints (`page:` — where the walk must BE, checked every time) and
moves (what it DOES). `do` runs a gesture macro, `decide` brokers one
bounded question to a model call (`calls.py`), `ask` messages the user
and holds for approval, `tell` messages and pauses, `sync` converges
the preceding page onto the shopping list in code. Execution lives in
`program.py`; this module's contract is the macro parser's:
all-or-nothing validation with errors that name the offending field, so
a walker never meets an unknown macro, a dangling page, or an unrouted
decision.

The grammar, top-down (YAML keys are the user vocabulary — the model
classes below keep their historical field names, each commented with
the key it parses from)::

    playbook  ::= description [enabled] [inputs] [budget] [context] route
    inputs    ::= {id: {description, [default], [example], [type]}}  # ≤ MAX_INPUTS
                  # type: text | list ("list" is the buying-list ledger)
    budget    ::= number | "{inputs.x}"                # scalar = max_amount
                | {max_amount, [expires_minutes]}
    context   ::= [memory.<slug>, ...]      # memory the task reading receives
    route     ::= [entry, ...]   # an optional prefix of pure-text agents
                                 # and one `start`, then the first page
    entry     ::= "page" name [anchors] [forbid] [scrollable] [open] [recover]
                | "start" name macro          # the unconditional cold-launch
                | "do" name [with] [macro] [undo] [irreversible]
                | "agent" name prompt [tools] [give] [returns] [limit]
                  [irreversible]             # the step handed to the model
                | "decide" name uses [with] [context] [answers] routes [max_asks]
                | "ask" name approve message [over_budget_message]
                  [resume] [revise]
                | "tell" name message
                | "sync" name
    macro     ::= name                    # macros/<name>/ — a directory macro
                | {[inputs] steps}        # inline — MACRO.yml grammar minus
                                          # name/description/enabled
    routes    ::= {answer: move-id | page-name | "escalate"}  # total over answers

Route semantics — how the alternation compiles onto the node graph:

  - The FIRST entry is the start page. It IS the first move's derived
    enter, so a wrong screen at walk start climbs the init ladder
    (go_back, then force_quit + the start page's `open:` body — or the
    pack's directory `open` macro) via the rescue machinery before the
    first `do` runs. (A decide/ask/tell first move reads the screen as
    it is — those moves have no precondition, as before the route
    grammar.) `open:` is legal only on the start page.
  - A `do`'s precondition (`enter`) is the nearest preceding page; its
    landing check (`verify`) is the page entry that MUST immediately
    follow it — the reflector, enforced by shape.
  - A `page:` entry either DECLARES the page in place (anchors/forbid/
    scrollable beside it — merged into the pack's page namespace by
    `pages.collect_page_decls`, so the matcher sees it through every
    door) or references one declared elsewhere (this pack's `pages:`
    appendix, or another waypoint).
  - A route target may name a move, `escalate`, or a page — a page
    target means "this answer lands there" and resolves to the move
    after that waypoint (ambiguous if the page recurs on the route:
    name the move instead).
  - `sync` acts on the page it follows; `ask`/`tell`/`decide` sit on
    the current page and fall through (decide: route) to the next entry.
  - A `do`'s name IS its macro: the directory macro of that name, or
    the `macro:` body beside it (`macro: <other-name>` overrides when
    one directory macro serves two moves).

An inline macro is single-use by construction — an anonymous body has
no name for another site to reference: its name is synthesized
`<playbook>.<move>` (a do body) or `<playbook>.<name>.<role>` (an
undo/return/open body) — dot-joined, a spelling no directory macro or
move can take, so the pack-wide dispatch namespace stays collision-free
by construction. It is enabled iff its playbook is, and it records
stats/runs under that synthesized name like any pack macro. Grammar
boundary: everything under an inline `macro:` IS a macro (single-name
`{x}` templates, fed by the move's `with:`), everything outside stays
dotted — the same split as the file boundary, moved to the key.

The ledger stack (`type: list` input + `next_item` loop + `sync` +
gate `revise:`) is one unit — `_check_ledger` holds its pieces
together: the ONE `type: list` input is the buying list (desired
state), the `next_item` decide closes the shopping loop over it (its
`next` arm is the one sanctioned backward edge; body moves read
loop-scoped `{item.query}`/`{item.qty}`), `sync` diffs the cart
(observed state) against the ledger in code, and a payment ask's
`revise:` routes a "yes, but change it" reply back into the loop
instead of handing over.

Control flow: moves fall through in route order (past the last entry =
done); a decide routes every one of its answers explicitly. An `ask`
falls through only once the user has approved: it sends the authored
message over the user channel, waits for a reply, and a micro-call
judges whether the reply confirms. Unconfirmed → wait and check again,
GATE_MAX_CHECKS times in all; still unconfirmed → the session suspends
for the next wake-up, the regular-session contract. `escalate` is a
reserved target — the conductor goes quiet and the model takes over.
The only legal cycles are a decide self-routing its call's re-ask arm
(choose_item's `scroll` — the conductor swipes between re-asks, bounded
by `max_asks`) and the ledger loop's declared backward arm
(`CallDecl.loop_arm`, terminating by item consumption); anything wider
is the model's job, not a playbook's.

Wiring is by placeholder, and every ref is dotted: `{inputs.name}`
reads a declared input, `{move.field}` reads an EARLIER decide's
declared payload field, `{item.field}` the ledger loop's current item,
`{ask.total}`/`{ask.cap}` a payment ask's money slots. A bare `{name}`
is a load error. Waypoints name this pack's pages bare (declaration and
reference share the route's context); the reserved built-ins stay
dotted (`ios.<page>`/`channel.<page>`). Dotted refs are playbook-level
— resolved to plain strings before any macro sees them, so pack macros
keep the stock single-name template grammar.

Agent steps: an `agent` with no `tools` is a pure-text call (prompt in,
`returns` fields out — outputs read downstream as `{name.field}` refs,
legal before the first page); with `tools` it is a screen EPISODE framed
by the adjacent waypoints exactly like a `do` — `give` grants it
declared landmarks by name, `limit` bounds its calls/scrolls, and its
exit is the following page, audited by the matcher. A page's `recover:`
declares its recovery hand (one gesture or an argument-less macro); ANY
recover in the playbook turns the hidden rescue ladder off — what you
declare is what you get.

Money is a parse-time lint, not doctrine: a move tagged
`irreversible: payment` (a leg or an agent episode) must be unreachable
except through an `ask` that approves payment. A `budget:` is optional —
without one the consented total is the fire-time bound, and there is no
over-budget branch to author.
"""

import re
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import Any

from physiclaw.common import paths
from physiclaw.common.paths import PACK_FILENAME
from physiclaw.conductor import _spec
from physiclaw.conductor.calls import (
    AGENT_TOOLS,
    CALLS,
    ESCALATE,
    LEDGER_FIELDS,
    NEXT_ITEM,
)
from physiclaw.conductor.pages import (
    PAGE_DECL_FIELDS,
    RESERVED_APPS,
    Landmark,
    PageDecl,
    PagesError,
    collect_page_decls,
    pack_landmarks,
    parse_pages_data,
    route_decl,
)
from physiclaw.macros import store as macro_store
from physiclaw.macros.model import Macro, MacroError
from physiclaw.macros.parse import parse_inline_macro

# The cap counts MOVES (compiled nodes) — waypoints ride free, bounded
# by the pack's own MAX_PAGES.
MAX_NODES = 20
MAX_INPUTS = 8
MAX_VISITS_CAP = 10
DEFAULT_MAX_VISITS = 3
# How many reply-check rounds an `ask` gets before the session suspends
# for the next wake-up, and how many "yes, but change it" revision cycles
# one ask absorbs before the model takes over. Fixed, not authorable:
# the patience budget is the conductor's contract with the user, not a
# per-playbook knob.
GATE_MAX_CHECKS = 3
GATE_MAX_REVISIONS = 2

# `{root.name}` — the playbook's own ref grammar, always dotted
# (`inputs.` / `item.` / an earlier decide or agent node's id). The root
# follows the move-name grammar (hyphens included — `{pick-into-cart.message}`);
# the field stays input-shaped. Dotted refs are deliberately NOT part of
# the macro template layer (its tokenizer rejects them); same `{{`/`}}`
# escapes, same fail-at-load-on-stray-brace rule.
REF_RE = re.compile(r"\{\{|\}\}|\{([a-z0-9][a-z0-9_-]*\.[a-z][a-z0-9_]*)\}|[{}]")
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

# The ref grammar's global roots — rejected as move names so a decide's
# `{move.field}` outputs can never shadow `{inputs.*}` or the ledger's
# `{item.*}`. (`ask` is not global: it exists only inside a payment
# ask's own messages, where the money slots win.)
RESERVED_REF_ROOTS = frozenset({"inputs", "item"})

_INPUT_KEYS = {"description", "default", "example", "type"}
_INPUT_KINDS = ("text", "list")
_BUDGET_KEYS = {"max_amount", "expires_minutes"}

# Agent-step bounds. `tools` is the closed per-episode gesture allowlist
# (`calls.AGENT_TOOLS`, the keys of the tool → verbs map the runner
# reads); `limit.calls` bounds the LLM calls an episode may spend (each
# action is one call), `limit.scrolls` the scrolling subset — the
# classic degenerate loop gets its own tighter cap.
MAX_AGENT_CALLS = 30
DEFAULT_AGENT_CALLS = 12
DEFAULT_AGENT_SCROLLS = 6
MAX_PROMPT_LEN = 2000
MAX_RETURNS = 6
_AGENT_LIMIT_KEYS = {"calls", "scrolls"}

# The declared-recovery hand: one gesture (`tool`, with `with:
# landmarks.<name>` for a tap) or a macro. Closed tool vocabulary — a
# recover hand resets state, it does not navigate (the route does that).
RECOVER_TOOLS = ("force_quit", "go_back", "home_screen", "tap")
_RECOVER_KEYS = {"tool", "with", "macro"}

# Route entry vocabularies. An entry's KIND is its leading key and the
# value is the entry's name — the map-key-is-the-name doctrine, applied
# to the route. Page-declaration fields come from `pages.py`'s ONE
# spelling (PAGE_DECL_FIELDS) — their content is validated there; they
# appear here only so the unknown-key check names them as legal.
ENTRY_KINDS = ("page", "start", "do", "agent", "decide", "ask", "tell", "sync")
_ENTRY_KEYS = {
    "page": {"page", "open", "recover", *PAGE_DECL_FIELDS},
    "start": {"start", "macro"},
    "do": {"do", "with", "macro", "undo", "irreversible"},
    "agent": {"agent", "prompt", "tools", "give", "returns", "limit", "irreversible"},
    "decide": {"decide", "uses", "with", "context", "answers", "routes", "max_asks"},
    "ask": {
        "ask",
        "approve",
        "message",
        "over_budget_message",
        "resume",
        "return",
        "revise",
    },
    "tell": {"tell", "message"},
    "sync": {"sync"},
}

PACK_MACROS_DIRNAME = "macros"

_PLAY_KEYS = {"description", "enabled", "inputs", "budget", "route", "context"}


class PlaybookError(ValueError):
    """A playbook (or its pack wiring) is invalid. Message is user-facing:
    `physiclaw playbooks check` prints it verbatim. All-or-nothing."""


_require_str, _prose, _opt_prose, check_name = _spec.bind(PlaybookError)
INPUT_NAME_RE = _spec.INPUT_NAME_RE


@dataclass(frozen=True)
class InputRef:
    """A `{inputs.name}` reference resolved at parse time — the consumer
    (the budget check, of all places) must never re-derive the brace
    grammar."""

    name: str


@dataclass(frozen=True)
class PlaybookInput:
    name: str
    description: str
    default: str | None = None  # present → optional
    example: str | None = None
    # `type:` — "text" (a template string) or "list", the buying-list
    # ledger: its VALUE is a JSON array string ([{query, qty}],
    # validated at arm/activation), consumed by the next_item loop as
    # {item.*} refs, never referenced as an {inputs.name} template.
    kind: str = "text"

    @property
    def required(self) -> bool:
        return self.default is None


@dataclass(frozen=True)
class Mandate:
    """`budget:` — the user-authorized spend bound, enforced by the
    conductor in code at checkout-class moves — never trusted to any
    model call."""

    max_amount: float | InputRef
    expires_minutes: int | None


@dataclass(frozen=True)
class LegNode:
    """A `do` entry: run one gesture macro, land on the next waypoint."""

    id: str
    macro: str  # the do-name itself, a `macro:` override, or a synthesized inline name
    args: dict[str, Any]  # `with:`
    enter: str  # derived: the nearest preceding waypoint; "" on a `start`
    #   leg, which runs unconditionally from wherever the phone is
    verify: str  # derived: the waypoint that follows — the reflector, mandatory
    compensate: str | None  # `undo:` — the macro that reverses this move
    irreversible: str | None  # one of IRREVERSIBLE_CLASSES


@dataclass(frozen=True)
class DecideNode:
    """A `decide` entry: one bounded question about the current screen."""

    id: str
    call: str  # `uses:` — the registered decision call
    args: dict[str, Any]  # `with:`
    context: tuple[str, ...]
    outcomes: tuple[str, ...]  # `answers:` — fixed for choose_item, authored for decide
    routes: dict[
        str, str
    ]  # answer → move id | reserved target (page targets resolved at parse)
    max_visits: int  # `max_asks:`


@dataclass(frozen=True)
class ReconcileNode:
    """A `sync` entry — desired-state convergence, zero LLM: on the cart
    page it follows, diff the cart rows (observed) against the ledger
    (desired) in code and act — quantity via the row's +/− steppers,
    removal is minus-to-zero, a missing picked item re-enters the
    shopping loop. Cart rows matching no ledger item are LEFT ALONE:
    they may be the user's own, and the conductor never destroys what it
    cannot attribute to itself."""

    id: str
    page: str  # derived: the waypoint this sync follows — re-read after every action


@dataclass(frozen=True)
class ConfirmNode:
    """A `tell` entry: message the user, then pause until any reply."""

    id: str
    # The authored text ({inputs.name}/{move.field} refs, parse-validated,
    # runtime-filled), sent VERBATIM and REQUIRED: only the playbook
    # author knows the user's language, so the conductor composes no
    # prose — ever.
    message: str


@dataclass(frozen=True)
class HumanGateNode:
    """An `ask` entry — ask-and-hold: send the authored message over the
    user channel, and fall through only on a micro-call-confirmed reply.
    GATE_MAX_CHECKS unconfirmed rounds suspend the session (regular wake-up
    contract) — so the move after an ask runs human-approved or not at all."""

    id: str
    gate: str  # `approve:` — what a yes authorizes, e.g. "payment"
    # The authored ask, REQUIRED and sent verbatim like
    # ConfirmNode.message. The consent contract is enforced as lints on
    # the template, not as appended code prose: a payment ask's
    # `message` must reference {ask.total} (the ask IS the consent
    # record), and `over_budget_message` — sent instead when the sheet
    # total exceeds the budget cap — must reference {ask.total} AND
    # {ask.cap} (the breach must be disclosed). Both slots are
    # runtime-filled from the payment sheet.
    message: str
    over_message: str | None  # `over_budget_message:`
    # `return:` — macro run after a confirmed reply to get BACK into the
    # app (the ask left it for the IM thread); the next move's derived
    # enter judges the landing. None = the walk tries the next move
    # directly and hands over if its enter check fails.
    return_macro: str | None
    # Ledger playbooks only: a "yes, but change it" reply routes HERE
    # (linted: the next_item move) after revise_list updates the ledger
    # — shop the additions, sync the changes, and re-ask with the fresh
    # total. None = a revise hands over (non-list behavior).
    revise: str | None


@dataclass(frozen=True)
class AgentNode:
    """An `agent` entry — the step handed to the model, inside a fence
    the author draws. No tools = a pure-text call (prompt in, `returns`
    fields out, no screen); tools = a screen-driving EPISODE whose exit
    is the adjacent `verify` waypoint (audited by the matcher, never by
    the model's claim). The prompt's refs fill ONCE when the step opens
    and the episode context is append-only, so every call's prefix is
    byte-identical to the previous call's whole request."""

    id: str
    prompt: str
    tools: tuple[str, ...]  # () = pure-text call
    give: tuple[str, ...]  # granted landmark names the model may name blind
    returns: tuple[tuple[str, str], ...]  # (field, description) pairs
    enter: str  # derived like a leg's; "" for a pure-text call
    verify: str  # derived like a leg's; "" for a pure-text call
    max_calls: int
    max_scrolls: int
    irreversible: str | None  # one of IRREVERSIBLE_CLASSES

    @property
    def return_fields(self) -> tuple[str, ...]:
        return tuple(name for name, _ in self.returns)


@dataclass(frozen=True)
class RecoverHand:
    """A page's declared `recover:` — what runs when the page is needed
    and does not read. One gesture (`tool`, a tap taking a landmark
    target) or one argument-less macro; after it runs the walk
    re-locates on the route. Declared, not a hidden ladder — a page
    without one hands over."""

    tool: str | None = None  # one of RECOVER_TOOLS
    landmark: str | None = None  # the tap target's landmark name
    macro: str | None = None  # resolved macro name


Node = LegNode | AgentNode | DecideNode | ReconcileNode | ConfirmNode | HumanGateNode


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
    mandate: Mandate | None  # `budget:`
    nodes: tuple[Node, ...]  # the route's MOVES, compiled (waypoints derived away)
    # The route's first waypoint — where the walk must be at start (it
    # is also the first move's derived enter, which is what the runtime
    # actually checks; a mismatch runs the init ladder — back, then
    # force_quit + open — before the first `do`).
    start: str = ""
    # The start waypoint's `open:` body under its synthesized name
    # (`<playbook>.<start>.open`), or None — the walk's cold-launch,
    # preferred over the pack's directory `open` macro by the rescue
    # reset rung and the boot init alike.
    open_macro: str | None = None
    # `context:` — memory slices the ACTIVATION's parse_task receives
    # (fail-closed, the decide contract) — e.g. `memory.shopping_prefs`.
    context: tuple[str, ...] = ()
    # The embedded macros — do bodies (`<playbook>.<move>`) and
    # undo/return/open bodies (`<playbook>.<name>.<role>`). The node
    # field holds the same synthesized name, so dispatch is name-keyed
    # either way. `qualified_inline` is the registry door.
    inline_macros: dict[str, Macro] = field(default_factory=dict)
    # Declared recovery, page name → hand. Non-empty flips the walk to
    # declared mode: the hidden rescue ladder is OFF, a mismatched page
    # runs ITS hand (or hands over when it declares none) after the
    # implicit unlock/settle.
    recovers: dict[str, RecoverHand] = field(default_factory=dict)

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
    # The pack's declared fixed spots (`landmarks:`) — recover hands
    # and agent grants name them, the rescue ladder consults `back` and
    # `dismiss`. See `pages.Landmark`.
    landmarks: dict[str, Landmark] = field(default_factory=dict)


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
        # Appendix + route-declared waypoints, one namespace — the
        # matcher and every playbook validate against the same set.
        pages = parse_pages_data(collect_page_decls(doc), app)
    except PagesError as e:
        raise PlaybookError(f"{app}/{PACK_FILENAME} pages: {e}") from e
    try:
        landmarks = pack_landmarks(doc)
    except PagesError as e:
        raise PlaybookError(f"{app}/{PACK_FILENAME} landmarks: {e}") from e
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
        landmarks=landmarks,
    )


def _check_pack_meta(doc: dict, app: str) -> None:
    """The manifest half of the pack file: `app` (which app this pack
    automates) equals the directory, `description` is real prose,
    `placeholders` (install-time constants, validated here so `check`
    catches a malformed map before install prompts read it) is
    name → {description, [example]}."""
    declared = _require_str(doc.get("app"), "`app`")
    if declared != app:
        raise PlaybookError(f"app {declared!r} must equal the pack directory {app!r}")
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

    raw_ctx = data.get("context", [])
    if not isinstance(raw_ctx, list):
        raise PlaybookError("`context` must be a list")
    context = []
    for c in raw_ctx:
        # The decide `context:` grammar (CONTEXT_RE), memory root only —
        # activation runs before any walk, so `inputs.*` cannot exist.
        if not isinstance(c, str) or not (
            CONTEXT_RE.match(c) and c.startswith("memory.")
        ):
            raise PlaybookError(
                f"`context` entry {c!r} must look like `memory.<slug>` "
                "(activation runs before any walk, so only memory slices exist)"
            )
        context.append(c)

    inputs = _parse_inputs(data.get("inputs", {}))
    # Refs resolve TEXT inputs only: a `type: list` value is a JSON
    # ledger, consumed by the next_item loop as {item.*} — splicing it
    # into a template would paste raw JSON into a gesture argument.
    ref_names = {i.name for i in inputs if i.kind == "text"}
    has_ledger = any(i.kind == "list" for i in inputs)
    mandate = _parse_budget(data["budget"], ref_names) if "budget" in data else None
    inline: dict[str, Macro] = {}
    resolve = _macro_resolver(name, pack, inline)
    nodes, start, open_macro, recovers = _parse_route(
        data.get("route"),
        ref_names,
        pack,
        resolve,
        has_ledger=has_ledger,
        has_mandate=mandate is not None,
    )
    ids = {n.id: i for i, n in enumerate(nodes)}
    _check_graph(nodes, ids)
    _check_money(nodes, ids)
    _check_ledger(nodes, ids, inputs)
    return Playbook(
        app=pack.app,
        name=name,
        description=description,
        enabled=enabled,
        inputs=inputs,
        mandate=mandate,
        nodes=tuple(nodes),
        start=start,
        open_macro=open_macro,
        context=tuple(context),
        inline_macros=inline,
        recovers=recovers,
    )


def _parse_inputs(raw: Any) -> tuple[PlaybookInput, ...]:
    if not isinstance(raw, dict):
        raise PlaybookError("`inputs` must be a mapping of input names")
    if len(raw) > MAX_INPUTS:
        raise PlaybookError(f"too many inputs ({len(raw)} > {MAX_INPUTS})")
    out: list[PlaybookInput] = []
    for name, spec in raw.items():
        _field_name(name, "input name")
        if not isinstance(spec, dict):
            raise PlaybookError(f"input {name!r} must be a mapping")
        unknown = sorted(set(spec.keys()) - _INPUT_KEYS)
        if unknown:
            raise PlaybookError(f"input {name!r}: unknown key(s): {', '.join(unknown)}")
        kind = spec.get("type", "text")
        if kind not in _INPUT_KINDS:
            raise PlaybookError(
                f"input {name!r}: `type` must be one of {', '.join(_INPUT_KINDS)} "
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


def _field_name(name: Any, what: str) -> str:
    """The one naming rule for the values `{x.y}` refs read — declared
    inputs and an agent's return fields."""
    if not isinstance(name, str) or not INPUT_NAME_RE.match(name):
        raise PlaybookError(
            f"{what} {name!r} must be lowercase, start with a "
            "letter, and contain only letters/digits/underscores"
        )
    return name


def _unique_list(raw: Any, where: str, check: Callable[[Any], str]) -> list[str]:
    """A list of distinct entries, each validated (and normalized) by
    `check` — the one shape `tools`, `give` and `answers` share."""
    if not isinstance(raw, list):
        raise PlaybookError(f"{where} must be a list")
    out: list[str] = []
    for item in raw:
        value = check(item)
        if value in out:
            raise PlaybookError(f"{where}: duplicate entry {value!r}")
        out.append(value)
    return out


def _irreversible_class(entry: dict, where: str) -> str | None:
    """A move's optional `irreversible:` class — the same closed
    vocabulary on a `do` and an `agent`."""
    irreversible = entry.get("irreversible")
    if irreversible is not None and irreversible not in IRREVERSIBLE_CLASSES:
        raise PlaybookError(
            f"{where}: `irreversible` must be one of "
            f"{', '.join(IRREVERSIBLE_CLASSES)} (got {irreversible!r})"
        )
    return irreversible


def _parse_budget(raw: Any, input_names: set[str]) -> Mandate:
    """`budget:` — a bare number or `{inputs.x}` ref IS the max amount
    (the common case reads as one line); the mapping form adds
    `expires_minutes`."""
    where = "`budget`"
    if not isinstance(raw, dict):
        return Mandate(
            max_amount=_budget_amount(raw, where, input_names), expires_minutes=None
        )
    unknown = sorted(set(raw.keys()) - _BUDGET_KEYS)
    if unknown:
        raise PlaybookError(f"{where}: unknown key(s): {', '.join(map(str, unknown))}")
    if "max_amount" not in raw:
        raise PlaybookError(f"{where} needs `max_amount`")
    max_amount = _budget_amount(raw["max_amount"], f"{where}.max_amount", input_names)
    expires = raw.get("expires_minutes")
    if expires is not None and (
        isinstance(expires, bool) or not isinstance(expires, int) or expires <= 0
    ):
        raise PlaybookError(f"{where}.expires_minutes must be a positive whole number")
    return Mandate(max_amount=max_amount, expires_minutes=expires)


def _budget_amount(
    amount: Any, where: str, input_names: set[str]
) -> "float | InputRef":
    if isinstance(amount, (int, float)) and not isinstance(amount, bool):
        if amount <= 0:
            raise PlaybookError(f"{where} must be positive (got {amount})")
        return float(amount)
    if isinstance(amount, str) and (m := _SOLE_REF.match(amount.strip())):
        name = m.group(1)
        if name not in input_names:
            raise PlaybookError(
                f"{where}: placeholder {{inputs.{name}}} not declared under `inputs`"
            )
        return InputRef(name=name)
    raise PlaybookError(
        f"{where} must be a number or exactly one `{{inputs.name}}` ref"
    )


# ---------- the route ----------


def _parse_route(
    raw: Any,
    input_names: set[str],
    pack: Pack,
    resolve: "_MacroResolve",
    *,
    has_ledger: bool,
    has_mandate: bool,
) -> "tuple[list[Node], str, str | None, dict[str, RecoverHand]]":
    """`route:` → (compiled moves, start page, open-macro name, recovers).

    Waypoints do not become nodes — they become the adjacent moves'
    checks: a `do`'s (and an acting `agent`'s) enter is the nearest
    preceding page, its verify the page that must immediately follow it
    (the reflector, enforced by shape), a `sync`'s page is the one it
    follows. The route opens with an optional prefix of pure-text
    `agent` steps (no screen — they may run before the phone is
    touched) and an optional `start` move (the unconditional
    cold-launch, verified by the page that follows it); the first page
    waypoint is the start contract either way. A `page:` may carry
    `recover:` — its declared recovery hand. A decide route target
    naming a page resolves to the move after that waypoint, so
    `pick: detail` reads as the landing it is."""
    if not isinstance(raw, list) or not raw:
        raise PlaybookError("`route` must be a non-empty list")
    entries = [_classify_entry(i, e) for i, e in enumerate(raw, start=1)]
    first_page = next((i for i, (k, _, _) in enumerate(entries) if k == "page"), None)
    if first_page is None:
        raise PlaybookError(
            "the route needs a page waypoint — the walk's start contract"
        )
    starts = [i for i, (k, _, _) in enumerate(entries) if k == "start"]
    if len(starts) > 1:
        raise PlaybookError("at most one `start` — a route cold-launches once")
    if starts and starts[0] != first_page - 1:
        raise PlaybookError(
            "`start` must sit immediately before the first page — the page "
            "that follows it is the landing it must reach"
        )
    for i in range(first_page):
        kind, _, _ = entries[i]
        if kind == "start":
            continue
        if kind != "agent":
            # (An acting agent up here fails in `_parse_agent`: it has
            # no page to start on.)
            raise PlaybookError(
                f"route entry {i + 1}: only pure-text `agent` steps (no "
                "tools) and the `start` move may precede the first page — "
                f"a `{kind}` needs a screen the route has not reached yet"
            )
    if all(kind == "page" for kind, _, _ in entries):
        raise PlaybookError(
            "the route needs at least one move (start/do/agent/decide/ask/tell/sync)"
        )

    # Waypoint prepass: every page id resolved and its in-place
    # declaration validated up front (`_waypoint_id` — one grammar at
    # every door), so a `do` can read the page that follows it in one
    # forward look. `pages.route_decl` is the one declaration predicate,
    # shared with `collect_page_decls` so the two doors cannot disagree.
    declared_here = {
        name
        for kind, name, entry in entries
        if kind == "page" and route_decl(entry) is not None
    }
    wp_ids: list[str | None] = []
    page_pos: dict[str, list[int]] = {}
    for i, (kind, name, entry) in enumerate(entries):
        if kind != "page":
            wp_ids.append(None)
            continue
        pid = _waypoint_id(i + 1, name, entry, pack, declared_here)
        wp_ids.append(pid)
        page_pos.setdefault(pid, []).append(i)
    start = wp_ids[first_page]
    assert start is not None  # first_page indexes a page entry

    moves: list[Node] = []
    seen: dict[str, int] = {}
    recovers: dict[str, RecoverHand] = {}
    # Grows as the single pass advances, so `{move.field}` refs are
    # defined-before-use by construction (route order). With a ledger,
    # the loop-scoped `{item.*}` refs are available everywhere as a
    # pseudo-payload; no move can shadow it — `item` is a reserved root.
    payloads: dict[str, tuple[str, ...]] = {"item": LEDGER_FIELDS} if has_ledger else {}
    current_page: str | None = None
    # Legacy `open:` on the start page — retired by the `start` move
    # (the two would launch twice), kept for packs that predate it.
    open_macro = None
    if not starts:
        open_macro = _optional_pack_macro(
            entries[first_page][2], "open", f"start page {start!r}", start, resolve
        )
    for i, (kind, name, entry) in enumerate(entries):
        pos = i + 1
        if kind == "page":
            current_page = wp_ids[i]
            if "open" in entry and (starts or i != first_page):
                raise PlaybookError(
                    f"route entry {pos}: `open` belongs to the START page "
                    "only, and a route with a `start` move has no use for "
                    "it — the start IS the cold-launch"
                )
            if "recover" in entry:
                rpage = wp_ids[i]
                assert rpage is not None
                if "." in rpage:
                    raise PlaybookError(
                        f"route entry {pos}: {rpage!r} is a reserved built-in "
                        "— packs declare recovery for their own pages only"
                    )
                hand = _parse_recover(
                    entry["recover"], f"route entry {pos}", rpage, pack, resolve
                )
                if rpage in recovers and recovers[rpage] != hand:
                    raise PlaybookError(
                        f"route entry {pos}: page {rpage!r} declares `recover` "
                        "twice with different hands — declare it once"
                    )
                recovers[rpage] = hand
            continue
        where = f"route entry {pos}"
        check_name(name, f"{where}: `{kind}`")
        if name in RESERVED_TARGETS:
            raise PlaybookError(
                f"{where}: name {name!r} is a reserved routing target — a "
                "move must not shadow the escalation sink"
            )
        if name in RESERVED_REF_ROOTS:
            raise PlaybookError(
                f"{where}: name {name!r} is a reserved ref root — "
                "{inputs.*} and {item.*} always read the declared inputs "
                "and the ledger item"
            )
        if name in page_pos:
            raise PlaybookError(
                f"{where}: {name!r} is also a page on this route — moves "
                "and pages share the routing namespace, so the names must "
                "not collide"
            )
        if name in seen:
            raise PlaybookError(
                f"{where}: duplicate move name {name!r} (entry {seen[name]} "
                "already uses it) — routing addresses moves by name, so "
                "they must be unique"
            )
        seen[name] = pos
        where = f"move {name!r}"
        args = entry.get("with", {})
        if not isinstance(args, dict):
            raise PlaybookError(f"{where}: `with` must be a mapping of arguments")
        _check_arg_refs(args, input_names, payloads, where)
        if kind == "do":
            nxt = wp_ids[i + 1] if i + 1 < len(entries) else None
            if nxt is None:
                raise PlaybookError(
                    f"{where}: a `do` must be followed by the page it lands "
                    "on — the landing check is what proves the move ran"
                )
            assert current_page is not None  # checked by the prefix rule
            moves.append(
                _parse_do(where, name, entry, args, current_page, nxt, resolve)
            )
        elif kind == "start":
            nxt = wp_ids[i + 1] if i + 1 < len(entries) else None
            assert nxt is not None  # `start` sits immediately before a page
            spec = resolve(entry.get("macro"), where, name)
            moves.append(
                LegNode(
                    id=name,
                    macro=spec.name,
                    args={},
                    enter="",  # unconditional: the start runs from anywhere
                    verify=nxt,
                    compensate=None,
                    irreversible=None,
                )
            )
        elif kind == "agent":
            nxt = wp_ids[i + 1] if i + 1 < len(entries) else None
            agent = _parse_agent(
                where,
                name,
                entry,
                input_names,
                payloads,
                pack,
                current_page,
                nxt,
            )
            payloads[agent.id] = agent.return_fields
            moves.append(agent)
        elif kind == "decide":
            node = _parse_decide(where, name, entry, args, input_names)
            payloads[node.id] = CALLS[node.call].payload
            moves.append(node)
        elif kind == "ask":
            moves.append(
                _parse_ask(
                    where,
                    name,
                    entry,
                    input_names,
                    payloads,
                    resolve,
                    has_mandate=has_mandate,
                )
            )
        elif kind == "tell":
            message, _ = _entry_message(where, entry, input_names, payloads)
            moves.append(ConfirmNode(id=name, message=message))
        else:  # sync
            assert current_page is not None  # the route starts at a page
            if "." in current_page:
                raise PlaybookError(
                    f"{where}: `sync` acts on the page it follows, and "
                    f"{current_page!r} is a reserved built-in it cannot act on"
                )
            moves.append(ReconcileNode(id=name, page=current_page))
    if len(moves) > MAX_NODES:
        raise PlaybookError(f"too many moves ({len(moves)} > {MAX_NODES})")
    _resolve_targets(moves, entries, page_pos)
    return moves, start, open_macro, recovers


def _classify_entry(i: int, entry: Any) -> tuple[str, str, dict]:
    """(kind, name, entry) for one route entry — the kind is its leading
    key, the value the name; exactly one kind key, and only that kind's
    field vocabulary beside it."""
    where = f"route entry {i}"
    if not isinstance(entry, dict):
        raise PlaybookError(f"{where} must be a mapping")
    kinds = [k for k in ENTRY_KINDS if k in entry]
    if len(kinds) != 1:
        raise PlaybookError(
            f"{where} must carry exactly one of {', '.join(ENTRY_KINDS)} "
            f"(got: {', '.join(map(str, sorted(entry))) or '(empty)'})"
        )
    kind = kinds[0]
    unknown = sorted(set(map(str, entry.keys())) - _ENTRY_KEYS[kind])
    if unknown:
        raise PlaybookError(
            f"{where}: unknown key(s) for `{kind}`: {', '.join(unknown)}"
        )
    return kind, _require_str(entry.get(kind), f"{where}: `{kind}`"), entry


def _waypoint_id(pos: int, name: str, entry: dict, pack: Pack, declared: set) -> str:
    """One page waypoint's id. Own-pack pages are written bare (the route
    IS the pack's context); the reserved built-ins stay dotted
    (`ios.<page>`/`channel.<page>`) and can only be referenced, never
    declared or opened here."""
    where = f"route entry {pos}"
    if "." in name:
        app, _, page = name.partition(".")
        if app not in RESERVED_APPS:
            raise PlaybookError(
                f"{where}: page {name!r} — waypoints name this pack's pages "
                f"bare, or a reserved namespace "
                f"({', '.join(sorted(RESERVED_APPS))}).<page>"
            )
        check_name(page, f"{where}: page")
        if not entry.keys().isdisjoint(("open", *PAGE_DECL_FIELDS)):
            raise PlaybookError(
                f"{where}: {name!r} is a reserved built-in — it cannot be "
                "declared or opened from a pack"
            )
        return name
    check_name(name, f"{where}: `page`")
    decl = route_decl(entry)
    if decl is not None:
        # Validate the in-place declaration's CONTENT here too, so the
        # text door (`parse_playbook` — tests, tooling) enforces the
        # same page grammar the pack door does via `collect_page_decls`;
        # a playbook green at one door must not go red at the other.
        try:
            parse_pages_data({name: decl}, pack.app)
        except PagesError as e:
            raise PlaybookError(f"{where}: {e}") from e
    elif name not in pack.pages and name not in declared:
        known = ", ".join(sorted(pack.pages)) or "(none)"
        raise PlaybookError(
            f"{where}: page {name!r} is not declared — declare it here "
            "(anchors beside the waypoint), in this pack's `pages:` "
            f"section, or on another route. Declared: {known}"
        )
    return name


def _resolve_targets(
    moves: list[Node],
    entries: list[tuple],
    page_pos: dict[str, list[int]],
) -> None:
    """Decide route targets, resolved in place: a move name stands as
    written; a page name becomes the move after that waypoint ("this
    answer lands there"). Applied AFTER the full pass so a target may
    point anywhere on the route — including the ledger loop's backward
    arm."""
    move_ids = {n.id for n in moves}
    # entry index → the id of the first move at-or-after it
    next_move: "list[str | None]" = [None] * len(entries)
    later: str | None = None
    for i in range(len(entries) - 1, -1, -1):
        if entries[i][0] != "page":
            later = entries[i][1]
        next_move[i] = later
    for at, node in enumerate(moves):
        if not isinstance(node, DecideNode):
            continue
        resolved: dict[str, str] = {}
        for out, target in node.routes.items():
            if target == ESCALATE or target in move_ids:
                resolved[out] = target
                continue
            where = f"move {node.id!r}: `routes.{out}`"
            positions = page_pos.get(target)
            if positions is None:
                raise PlaybookError(
                    f"{where}: {target!r} is neither a move nor a page on "
                    f"this route (or {ESCALATE!r})"
                )
            if len(positions) > 1:
                raise PlaybookError(
                    f"{where}: page {target!r} appears more than once on the "
                    "route — name the move this answer should land on instead"
                )
            landing = next_move[positions[0]]
            if landing is None:
                raise PlaybookError(
                    f"{where}: nothing follows page {target!r} — route this "
                    "answer to a move, or to `escalate`"
                )
            resolved[out] = landing
        # A self-route is sanctioned only on the call's re-ask arm — the
        # conductor refreshes the screen (swipes) between those visits.
        # Any other self-route would re-ask the identical screen with the
        # identical prompt: a lint-free playbook that can never converge.
        reask = CALLS[node.call].reask_arm
        for out, target in resolved.items():
            if target == node.id and out != reask:
                raise PlaybookError(
                    f"move {node.id!r}: `routes.{out}` routes back to this "
                    f"move — only {node.call}'s re-ask arm "
                    f"({reask or '(none for this call)'}) may self-loop; the "
                    "conductor scrolls between those re-asks"
                )
        moves[at] = replace(node, routes=resolved)


def _parse_ask(
    where: str,
    nid: str,
    entry: dict,
    input_names: set[str],
    payloads: dict[str, tuple[str, ...]],
    resolve: "_MacroResolve",
    *,
    has_mandate: bool,
) -> HumanGateNode:
    approve = _require_str(entry.get("approve"), f"{where}: `approve`")
    check_name(approve, f"{where}: `approve`")
    g_payloads = payloads
    if approve == "payment":
        # The consent slots, runtime-filled from the payment sheet and
        # available only here. {ask.cap} exists only when a `budget`
        # does. (A move literally named `ask` would be shadowed in this
        # message — the money slots win, both at parse and at fill.)
        slots = ("total", "cap") if has_mandate else ("total",)
        g_payloads = {**payloads, "ask": slots}
    message, msg_refs = _entry_message(where, entry, input_names, g_payloads)
    over = _opt_prose(
        entry.get("over_budget_message"), f"{where}: `over_budget_message`"
    )
    if approve == "payment":
        if "ask.total" not in msg_refs:
            raise PlaybookError(
                f"{where}: a payment ask's `message` must quote the sheet "
                "total — reference {ask.total} (the ask IS the consent "
                "record)"
            )
        # The over-budget branch exists exactly when a `budget` does: no
        # budget → the consented total is the only bound (nothing to
        # author); a budget → the breach message is mandatory.
        if over is not None and not has_mandate:
            raise PlaybookError(
                f"{where}: `over_budget_message` needs a `budget` — "
                "without a cap there is no over-budget branch"
            )
        if over is None and has_mandate:
            raise PlaybookError(
                f"{where}: a payment ask under a `budget` needs "
                "`over_budget_message` — sent instead when the total "
                "exceeds the cap; it must reference {ask.total} and {ask.cap}"
            )
        if over is not None:
            over_refs = _refs(over, f"{where}: `over_budget_message`")
            _check_refs(
                over_refs, input_names, g_payloads, f"{where}: `over_budget_message`"
            )
            missing = sorted({"ask.total", "ask.cap"} - over_refs)
            if missing:
                raise PlaybookError(
                    f"{where}: `over_budget_message` must reference "
                    + " and ".join("{" + m + "}" for m in missing)
                    + " — an over-budget ask must disclose the total AND the cap"
                )
    elif over is not None:
        raise PlaybookError(
            f"{where}: `over_budget_message` is only for `approve: payment`"
        )
    # `resume:` is the current spelling; `return:` the legacy one.
    if "resume" in entry and "return" in entry:
        raise PlaybookError(
            f"{where}: `resume` and `return` are one field — use `resume`"
        )
    resume_key = "resume" if "resume" in entry else "return"
    return_macro = _optional_pack_macro(entry, resume_key, where, nid, resolve)
    revise = entry.get("revise")
    if revise is not None:
        revise = _require_str(revise, f"{where}: `revise`")
        check_name(revise, f"{where}: `revise`")
    return HumanGateNode(
        id=nid,
        gate=approve,
        message=message,
        over_message=over,
        return_macro=return_macro,
        revise=revise,
    )


def _entry_message(
    where: str,
    entry: dict,
    input_names: set[str],
    payloads: dict[str, tuple[str, ...]],
) -> tuple[str, set[str]]:
    """A REQUIRED authored `message:` — the exact text sent to the user;
    only the author knows the user's language, so the conductor composes
    no prose around it. Refs held to the same defined-before-use rules
    as `with:` values; returned with them so the ask lints can inspect."""
    text = _prose(entry.get("message"), f"{where}: `message`")
    refs = _refs(text, f"{where}: `message`")
    _check_refs(refs, input_names, payloads, f"{where}: `message`")
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
    return _argless_macro(raw, key, where, nid, resolve)


def _argless_macro(
    raw: Any, key: str, where: str, nid: str, resolve: _MacroResolve
) -> str:
    """The value form: resolve one argument-less pack macro. The
    helper-hand spelling may wrap its body one level (`resume:` and
    `recover:` carry `macro:` inside their value) — unwrapped HERE, the
    rule's one home, so the resolver sees the same shapes a `do` does."""
    if isinstance(raw, dict) and set(raw) == {"macro"}:
        raw = raw["macro"]
    spec = resolve(raw, where, nid, key)
    required = sorted(i.name for i in spec.inputs if i.required)
    if required:
        raise PlaybookError(
            f"{where}: `{key}` macro {spec.name!r} requires input(s) "
            f"{', '.join(required)} — the walk dispatches {key} with no "
            "arguments"
        )
    return spec.name


def _parse_do(
    where: str,
    nid: str,
    entry: dict,
    args: dict,
    enter: str,
    verify: str,
    resolve: _MacroResolve,
) -> LegNode:
    """A `do` move. Its name IS its macro — the directory macro of that
    name, or the `macro:` beside it (an inline body, or another name
    when one directory macro serves two moves). `enter`/`verify` arrive
    derived from the route's waypoints."""
    spec = resolve(entry.get("macro", nid), where, nid)
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
    compensate = _optional_pack_macro(entry, "undo", where, nid, resolve)
    irreversible = _irreversible_class(entry, where)
    return LegNode(
        id=nid,
        macro=macro,
        args=dict(args),
        enter=enter,
        verify=verify,
        compensate=compensate,
        irreversible=irreversible,
    )


def _parse_agent(
    where: str,
    nid: str,
    entry: dict,
    input_names: set[str],
    payloads: dict[str, tuple[str, ...]],
    pack: Pack,
    current_page: str | None,
    next_wp: str | None,
) -> AgentNode:
    """An `agent` move. No `tools` = a pure-text call (needs `returns`,
    no pages); tools = an acting episode framed by the adjacent
    waypoints exactly like a `do`. The prompt is the author's whole
    brief — refs validated here, filled once when the step opens."""
    prompt = _require_str(entry.get("prompt"), f"{where}: `prompt`")
    if len(prompt) > MAX_PROMPT_LEN:
        raise PlaybookError(
            f"{where}: `prompt` is {len(prompt)} characters (max {MAX_PROMPT_LEN})"
        )

    def _tool(t: Any) -> str:
        if not isinstance(t, str) or t not in AGENT_TOOLS:
            raise PlaybookError(
                f"{where}: tool {t!r} — the episode vocabulary is "
                f"{', '.join(AGENT_TOOLS)}"
            )
        return t

    tools = _unique_list(entry.get("tools", []), f"{where}: `tools`", _tool)
    if not tools:
        # Everything below `tools` is about the screen an episode acts
        # on; on a pure-text call it is dead config.
        for key in ("give", "irreversible", "limit"):
            if key in entry:
                raise PlaybookError(
                    f"{where}: `{key}` is for acting episodes — a pure-text "
                    "call has no screen"
                )
    give = _unique_list(
        entry.get("give", []),
        f"{where}: `give`",
        lambda g: _landmark_name(g, f"{where}: `give` entry", pack),
    )

    raw_returns = entry.get("returns")
    returns: list[tuple[str, str]] = []
    if raw_returns is not None:
        if not isinstance(raw_returns, dict) or not raw_returns:
            raise PlaybookError(
                f"{where}: `returns` must be a mapping of field → description"
            )
        if len(raw_returns) > MAX_RETURNS:
            raise PlaybookError(
                f"{where}: {len(raw_returns)} return fields > max {MAX_RETURNS}"
            )
        for fname, desc in raw_returns.items():
            _field_name(fname, f"{where}: return field")
            returns.append((fname, _prose(desc, f"{where}: `returns.{fname}`")))

    if not tools and not returns:
        raise PlaybookError(
            f"{where}: an agent with neither `tools` nor `returns` can do "
            "nothing — give it hands, fields to fill, or both"
        )
    irreversible = _irreversible_class(entry, where)

    enter = verify = ""
    if tools:
        if current_page is None:
            raise PlaybookError(
                f"{where}: an acting agent needs the page it starts on — "
                "put a page waypoint before it"
            )
        if next_wp is None:
            raise PlaybookError(
                f"{where}: an acting agent must be followed by the page it "
                "finishes on — the landing check is its exit contract"
            )
        if "." in current_page or "." in next_wp:
            raise PlaybookError(
                f"{where}: an agent episode runs on this pack's own pages — "
                "reserved built-ins cannot frame it"
            )
        enter, verify = current_page, next_wp

    raw_limit = entry.get("limit")
    max_calls, max_scrolls = DEFAULT_AGENT_CALLS, DEFAULT_AGENT_SCROLLS
    if raw_limit is not None:
        if not isinstance(raw_limit, dict):
            raise PlaybookError(f"{where}: `limit` must be a mapping")
        unknown = sorted(set(map(str, raw_limit)) - _AGENT_LIMIT_KEYS)
        if unknown:
            raise PlaybookError(
                f"{where}: `limit`: unknown key(s): {', '.join(unknown)}"
            )
        max_calls = _limit_int(
            raw_limit.get("calls", DEFAULT_AGENT_CALLS),
            f"{where}: `limit.calls`",
            1,
            MAX_AGENT_CALLS,
        )
        max_scrolls = _limit_int(
            raw_limit.get("scrolls", min(DEFAULT_AGENT_SCROLLS, max_calls)),
            f"{where}: `limit.scrolls`",
            0,
            MAX_AGENT_CALLS,
        )
    if "scroll" not in tools:
        max_scrolls = 0

    g_payloads = payloads
    if irreversible == "payment":
        # The consented total — the ONE gate slot a payment episode may
        # quote (its ask directly precedes it, `_check_money`).
        g_payloads = {**payloads, "ask": ("total",)}
    refs = _refs(prompt, f"{where}: `prompt`")
    _check_refs(refs, input_names, g_payloads, f"{where}: `prompt`")

    return AgentNode(
        id=nid,
        prompt=prompt,
        tools=tuple(tools),
        give=tuple(give),
        returns=tuple(returns),
        enter=enter,
        verify=verify,
        max_calls=max_calls,
        max_scrolls=max_scrolls,
        irreversible=irreversible,
    )


def _limit_int(value: Any, where: str, lo: int, hi: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not lo <= value <= hi:
        raise PlaybookError(f"{where} must be {lo}–{hi} (got {value!r})")
    return value


def _parse_recover(
    raw: Any, where: str, page: str, pack: Pack, resolve: "_MacroResolve"
) -> RecoverHand:
    """A page's `recover:` hand — one gesture or one argument-less
    macro. A `tap` takes its target as `with: landmarks.<name>` (the
    declared spot, label-healed at run time); the other tools take no
    target."""
    if not isinstance(raw, dict):
        raise PlaybookError(
            f"{where}: `recover` must be a mapping — `tool:` (one gesture) or `macro:`"
        )
    unknown = sorted(set(map(str, raw)) - _RECOVER_KEYS)
    if unknown:
        raise PlaybookError(f"{where}: `recover`: unknown key(s): {', '.join(unknown)}")
    tool, macro_raw = raw.get("tool"), raw.get("macro")
    if (tool is None) == (macro_raw is None):
        raise PlaybookError(
            f"{where}: `recover` takes exactly one of `tool` or `macro`"
        )
    if macro_raw is not None:
        if "with" in raw:
            raise PlaybookError(f"{where}: `recover.with` goes with `tool: tap`")
        return RecoverHand(
            macro=_argless_macro(macro_raw, "recover", where, page, resolve)
        )
    if tool not in RECOVER_TOOLS:
        raise PlaybookError(
            f"{where}: `recover.tool` must be one of {', '.join(RECOVER_TOOLS)} "
            f"(got {tool!r})"
        )
    target = raw.get("with")
    if tool != "tap":
        if target is not None:
            raise PlaybookError(
                f"{where}: `recover.with` goes with `tool: tap` — "
                f"{tool} takes no target"
            )
        return RecoverHand(tool=tool)
    if target is None:
        raise PlaybookError(
            f"{where}: a recover tap needs `with: landmarks.<name>` — the "
            "declared spot it presses"
        )
    return RecoverHand(
        tool=tool, landmark=_landmark_name(target, f"{where}: `recover.with`", pack)
    )


def _landmark_name(value: Any, where: str, pack: Pack) -> str:
    """A `landmarks.<name>` reference resolved to its bare name — ONE
    spelling for `give:` entries and a recover tap's `with:`. The name
    half rides the shared name grammar (`check_name`), so a landmark
    reference can never drift from the section's own naming rule."""
    prefix, _, name = value.partition(".") if isinstance(value, str) else ("", "", "")
    if prefix != "landmarks" or not name:
        raise PlaybookError(f"{where}: {value!r} must look like `landmarks.<name>`")
    check_name(name, where)
    if name not in pack.landmarks:
        known = ", ".join(sorted(pack.landmarks)) or "(none)"
        raise PlaybookError(
            f"{where}: names landmark {name!r} — not declared under "
            f"`landmarks`. Declared: {known}"
        )
    return name


def _parse_decide(
    where: str, nid: str, entry: dict, args: dict, input_names: set[str]
) -> DecideNode:
    """A `decide` move. Its `routes:` are returned RAW — page-name
    targets resolve to moves in `_resolve_targets`, after the whole
    route is known."""
    call = entry.get("uses")
    if not isinstance(call, str) or call not in CALLS:
        raise PlaybookError(
            f"{where}: `uses` must be one of {', '.join(sorted(CALLS))} (got {call!r})"
        )
    decl = CALLS[call]
    if decl.deterministic and ("context" in entry or "max_asks" in entry):
        raise PlaybookError(
            f"{where}: {call} is deterministic — `context`/`max_asks` "
            "would be dead config (no prompt, no ask budget)"
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

    context_raw = entry.get("context", [])
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
        if "answers" in entry:
            raise PlaybookError(
                f"{where}: {call} declares its answers itself "
                f"({', '.join(decl.outcomes)}) — remove `answers`"
            )
        outcomes = decl.outcomes
    else:
        raw_answers = entry.get("answers")
        if not isinstance(raw_answers, list) or len(raw_answers) < 2:
            raise PlaybookError(
                f"{where}: {call} needs `answers` — what this question can "
                "answer (at least 2, including `escalate`)"
            )

        def _answer(o: Any) -> str:
            o = _require_str(o, f"{where}: `answers` item")
            check_name(o, f"{where}: `answers` item")
            return o

        answers_list = _unique_list(raw_answers, f"{where}: `answers`", _answer)
        if ESCALATE not in answers_list:
            raise PlaybookError(
                f"{where}: `answers` must include {ESCALATE!r} — every closed "
                "choice needs the concrete escape arm"
            )
        outcomes = tuple(answers_list)

    routes_raw = entry.get("routes")
    if not isinstance(routes_raw, dict) or not routes_raw:
        raise PlaybookError(
            f"{where}: `routes` must map every answer to a move, a page, "
            f"or {ESCALATE!r}"
        )
    extra = sorted(set(routes_raw.keys()) - set(outcomes))
    if extra:
        raise PlaybookError(
            f"{where}: `routes` names unknown answer(s): {', '.join(map(str, extra))}"
        )
    unrouted = sorted(set(outcomes) - set(routes_raw.keys()))
    if unrouted:
        raise PlaybookError(
            f"{where}: `routes` must route EVERY answer — missing: {', '.join(unrouted)}"
        )
    routes = {
        out: _require_str(target, f"{where}: `routes.{out}`")
        for out, target in routes_raw.items()
    }

    max_asks = _limit_int(
        entry.get("max_asks", DEFAULT_MAX_VISITS),
        f"{where}: `max_asks`",
        1,
        MAX_VISITS_CAP,
    )
    return DecideNode(
        id=nid,
        call=call,
        args=dict(args),
        context=tuple(context),
        outcomes=outcomes,
        routes=routes,
        max_visits=max_asks,
    )


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
                f"{where}: {{{ref}}} references move {root!r}, which "
                "is not an EARLIER decide move — outputs wire forward "
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
    a playbook at all. Covers every dispatching role: do moves, `undo:`,
    ask `return:`, and the start page's `open:` (the init ladder's hand).
    Safe unguarded access: parse validated every directory name against
    `pack.macros`."""
    named: set[str] = set()
    if spec.open_macro is not None:
        named.add(spec.open_macro)
    named.update(h.macro for h in spec.recovers.values() if h.macro is not None)
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
                        f"move {node.id!r}: `routes.{out}` routes to unknown "
                        f"move {target!r} (or a reserved target: "
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
            f"unreachable move(s): {', '.join(unreachable)} — every move "
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


def _check_money(nodes: list[Node], ids: dict[str, int]) -> None:
    """An irreversible-tagged move (a leg OR an agent episode) must be
    unreachable except through an `ask` that approves its class. Walked
    over the real edges with an approved flag, so the guarantee is
    structural. A `budget:` is optional: without one, the consented
    total IS the bound the fire-time predicates enforce. (The route's
    derived `enter` — always the waypoint before the ask — is what
    guarantees money fires on a verified app page, never on the IM
    thread the ask left the phone on.)"""

    def _irr(n: Node) -> str | None:
        return n.irreversible if isinstance(n, (LegNode, AgentNode)) else None

    if not any(_irr(n) for n in nodes):
        return
    # Adjacency, not just reachability: the conductor reads the payment
    # sheet AT the ask (the message quotes its total) and fires the move
    # as the ask's fall-through — a move in between would desynchronize
    # the consent from the sheet. The ask must also DECLARE the class
    # (`approve: payment`), so the runtime keys off the declaration.
    # Other irreversible classes take any ask: the human said go,
    # adjacently.
    for i, n in enumerate(nodes):
        irr = _irr(n)
        if not irr:
            continue
        prev = nodes[i - 1] if i > 0 else None
        if irr == "payment":
            if not (isinstance(prev, HumanGateNode) and prev.gate == "payment"):
                raise PlaybookError(
                    f"move {n.id!r} (irreversible: payment) must DIRECTLY "
                    "follow an `ask` with `approve: payment` — the ask "
                    "reads the sheet its message quotes, and consent must "
                    "not desynchronize from it"
                )
        elif not isinstance(prev, HumanGateNode):
            raise PlaybookError(
                f"move {n.id!r} (irreversible: {irr}) must "
                "DIRECTLY follow an `ask` — an irreversible act runs "
                "human-approved or not at all"
            )
        # The ask's fall-through is the move's ONLY door: a routed
        # in-edge (a decide arm, the ledger loop) would enter it with
        # another ask's consent still bound — or with none.
        for other in nodes:
            if isinstance(other, DecideNode) and n.id in other.routes.values():
                raise PlaybookError(
                    f"move {other.id!r} routes to {n.id!r} "
                    f"(irreversible: {irr}) — an irreversible "
                    "move is entered ONLY as its own ask's fall-through"
                )
    # DFS over (move, approved) states; a money move seen with the flag
    # down is reachable around the human.
    seen: set[tuple[str, bool]] = set()
    stack: list[tuple[str, bool]] = [(nodes[0].id, False)]
    while stack:
        nid, gated = stack.pop()
        if (nid, gated) in seen:
            continue
        seen.add((nid, gated))
        node = nodes[ids[nid]]
        if _irr(node) == "payment" and not gated:
            raise PlaybookError(
                f"move {nid!r} (irreversible: payment) is reachable without "
                "passing an `ask` — money always goes through the human"
            )
        passed = gated or (isinstance(node, HumanGateNode) and node.gate == "payment")
        for target in _successors(nodes, ids[nid]):
            stack.append((target, passed))


def _check_ledger(
    nodes: list[Node], ids: dict[str, int], inputs: tuple[PlaybookInput, ...]
) -> None:
    """The ledger stack is all-or-nothing: the ONE `type: list` input,
    the ONE next_item loop consuming it, and the optional `sync` /
    ask `revise:` that both lean on the loop. Half a stack is a walk
    that stalls at runtime — rejected here instead."""
    list_inputs = [i.name for i in inputs if i.kind == "list"]
    if len(list_inputs) > 1:
        raise PlaybookError(
            f"at most one `type: list` input (the ledger) — got: "
            f"{', '.join(list_inputs)}"
        )
    loops = [
        n
        for n in nodes
        if isinstance(n, DecideNode) and CALLS[n.call].loop_arm is not None
    ]
    if len(loops) > 1:
        raise PlaybookError(
            f"at most one {NEXT_ITEM} move — `sync` re-shop and ask "
            f"`revise:` both route through THE loop"
        )
    if loops and not list_inputs:
        raise PlaybookError(
            f"move {loops[0].id!r}: {NEXT_ITEM} walks the ledger — declare "
            "a `type: list` input"
        )
    if list_inputs and not loops:
        raise PlaybookError(
            f"input {list_inputs[0]!r} is `type: list` but no {NEXT_ITEM} "
            "move consumes it"
        )
    if not loops:
        stranded = next((n for n in nodes if isinstance(n, ReconcileNode)), None)
        if stranded is not None:
            raise PlaybookError(
                f"move {stranded.id!r}: `sync` needs the {NEXT_ITEM} loop "
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
                        f"move {loop.id!r}: `routes.{out}` must route "
                        "BACKWARD to the loop body head (the closer sits "
                        "at the loop's bottom)"
                    )
            elif target != ESCALATE and ids[target] <= idx:
                raise PlaybookError(
                    f"move {loop.id!r}: `routes.{out}` must route FORWARD — "
                    "the ledger is spent"
                )
    for n in nodes:
        if isinstance(n, HumanGateNode) and n.revise is not None:
            if not loops or n.revise != loops[0].id:
                raise PlaybookError(
                    f"move {n.id!r}: `revise` must target the {NEXT_ITEM} "
                    "move — a revision re-enters the loop for the added "
                    "items, then syncs the rest"
                )
            if n.return_macro is None:
                raise PlaybookError(
                    f"move {n.id!r}: `revise` needs `return:` — the reply "
                    "was read on the IM thread, and the loop must re-enter "
                    "the app before it can shop"
                )
