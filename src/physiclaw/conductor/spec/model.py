"""The playbook model — the grammar as dataclasses, and the pack it
validates against. A leaf: the compiler (`route.py`), the loader
(`pack.py`), the scaffold, and the walk all import it; it imports none
of them.

A pack is a folder: `PLAYBOOK.yml`, the MANIFEST (what the app is and
what its routes share — meta, placeholders, landmarks, pages; every
section optional, the file may be empty), one `<name>.yml` per
playbook beside it (the file body is the playbook: name, description,
enabled, inputs, route; the stem is the name, referenced as
`<app>/<name>`, and `name:` must agree with it), and
`macros/<name>/MACRO.yml` for the recorded hands
routes share. The manifest never carries a route.

A playbook is one app task written as a ROUTE: a top-down alternation
of waypoints (`page:` — where the walk must BE, checked every time) and
moves (what it DOES). The grammar, top-down (the YAML keys are the user
vocabulary and the model classes below carry the same names)::

    playbook  ::= description [enabled] [inputs] route
    inputs    ::= {id: {description, [default], [example]}}   # ≤ MAX_INPUTS
    route     ::= [agent...] [start] page (move page | ask | tell)*
    entry     ::= "page" name [anchors] [forbid] [scrollable] [recover] [tries]
                | "start" name macro          # the unconditional cold-launch
                | "do" name macro [with] [irreversible]
                | "agent" name prompt [tools] [give] [returns] [limit]
                  [context] [irreversible]   # the step handed to the model
                | "ask" name approve message yes no [total_label] [wait]
                  [rounds] [resume]           # payment: resume required when
                                              # a screen move follows
                | "tell" name message
                | "select" name [limit]       # channel/boot only, and last:
                                              # read the thread, hand a
                                              # playbook the baton
    recover   ::= hand                        # one hand for any deviation
                | {[covered: hand] [elsewhere: hand] [locked: hand]}
                # `tries:` beside it bounds the page; a manifest page's
                # recover: is inherited by every route, a route's own
                # replaces it whole for that walk
    hand      ::= go_back | force_quit | home_screen | unlock_phone
                | {tap: landmarks.<name>} | {macro: name | body}
    macro     ::= name                    # macros/<name>/ — a directory macro
                | {[inputs] steps}        # inline — MACRO.yml grammar minus
                                          # name/description/enabled

The route's shape IS the contract: an optional prefix of pure-text
`agent` steps and one `start` opens it, the first page is the start
contract, every `do` and every acting `agent` is followed by the page
it lands on (its landing check — the reflector, enforced by shape),
and a page may declare its own `recover:` hand — one hand for any
deviation, or one per reading (`covered:` a sheet over the page
itself, `locked:` the phone's lock screen, `elsewhere:` any other
screen), each page bounded by its own `tries:` under the walk-wide
ceiling. Moves fall through in
route order — there is no routing and no loop; whatever needs judgment
is an `agent` step inside the author's fence, and whatever needs a
human is an `ask`. Money runs in code: an `irreversible: payment` move
directly follows the `ask` that approves it (`lints.check_money`).

What the playbook declares is what runs — no more, no less. A page
without `recover:` hands over; an agent step runs the author's prompt
with the tools, landmarks, macros, and context the author listed and
nothing else; an ask's reply is read against the `yes:`/`no:` words
the ask declares, and anything they do not cover is the model's to
read; a payment ask names the label its `total_label:` sits beside, and
`wait:` and `rounds:` are its own patience.

Wiring is by placeholder, and every ref is dotted: `{inputs.name}` reads
a declared input, `{move.field}` reads an EARLIER agent step's declared
return field, `{ask.total}` a payment ask's quoted total. A
bare `{name}` is a load error. Dotted refs are playbook-level — resolved
to plain strings before any macro sees them, so pack macros keep the
stock single-name template grammar.

The loader (`pack.py`) turns files into these; the compiler (`route.py`)
and its lints turn a `route:` into the nodes.
"""

from dataclasses import dataclass, field
from typing import Any, Protocol

from physiclaw.conductor.spec import specfile
from physiclaw.conductor.spec.limits import (
    DEFAULT_ASK_ROUNDS,
    DEFAULT_ASK_WAIT_SECONDS,
    DEFAULT_RECOVER_LIMIT,
)
from physiclaw.conductor.spec.pages import Landmark, PageDecl
from physiclaw.macros.model import Macro, MacroInput

# The three readings a page's `recover:` may key its hands by: the page
# itself under an overlay, the phone's lock screen (where taps do not
# land — only `unlock_phone` helps), or any other screen.
READING_COVERED = "covered"
READING_ELSEWHERE = "elsewhere"
READING_LOCKED = "locked"
RECOVER_READINGS = (READING_COVERED, READING_ELSEWHERE, READING_LOCKED)
# The one irreversible class: money. A payment move is entered only as
# the fall-through of an `ask` with `approve: payment` (`lints.py`).
IRREVERSIBLE_CLASSES = ("payment",)

# The ref grammar's one global root — `{inputs.name}` — rejected as a
# move name so an agent's `{move.field}` outputs can never shadow it.
# (`ask` is not global: it exists only inside a payment ask's own
# messages, where the money slots win.)
INPUTS_ROOT = "inputs"

PACK_MACROS_DIRNAME = "macros"


class PlaybookError(specfile.SpecError):
    """A playbook (or its pack wiring) is invalid. Message is user-facing:
    `physiclaw playbooks check` prints it verbatim. All-or-nothing."""


# The scalar terminals, bound once for the whole playbook grammar —
# `route.py` (the compiler) reads them from here.
require_str, prose, opt_prose, check_name = specfile.bind(PlaybookError)


# ---------- the model ----------


# A playbook's `inputs:` IS the macro grammar's — one shape, one parser
# (`macros.parse.parse_inputs`), and the resolver reads either.
PlaybookInput = MacroInput


@dataclass(frozen=True)
class DoNode:
    """A `do` move (and the `start` move — a do with no `enter`, run
    unconditionally): one recorded macro, framed by the pages before
    and after it."""

    id: str
    macro: str  # the dispatch name — a directory macro or the inline body's
    args: dict  # `with:` — ref templates, filled at run time
    enter: str  # the page the move starts on; "" = unconditional (`start`)
    verify: str  # the page it must land on
    irreversible: str | None = None

    @property
    def start(self) -> bool:
        return not self.enter


@dataclass(frozen=True)
class AgentNode:
    """An `agent` move — the model's step, inside the author's fence.
    No `tools` = one pure-text call (`prompt` in, `returns` out); tools =
    an acting EPISODE framed by `enter`/`verify` exactly like a `do`."""

    id: str
    prompt: str
    tools: tuple[str, ...]
    give: tuple[str, ...]  # granted landmark names
    returns: tuple[tuple[str, str], ...]  # (field, description)
    enter: str
    verify: str
    max_calls: int
    max_scrolls: int
    irreversible: str | None = None
    context: tuple[str, ...] = ()  # `context:` — what to load (`context.py`)
    macros: tuple[str, ...] = ()  # granted pack macros (`give: [macros.<name>]`)

    @property
    def return_fields(self) -> tuple[str, ...]:
        return tuple(f for f, _ in self.returns)


@dataclass(frozen=True)
class AskNode:
    """An `ask` move — message the user and hold for approval. `approve`
    names the class the reply consents to (`payment` binds the quoted
    total); `yes`/`no` are the whole-message replies that open or close
    the gate, in `reply.normalize` space (anything else is the model's);
    `resume` is the macro that re-enters the app afterwards."""

    id: str
    approve: str
    message: str
    yes: tuple[str, ...]
    no: tuple[str, ...]
    resume: str | None = None
    # The waypoint before the ask — the page a payment ask reads its
    # total off ("" when none precedes it; a payment ask requires one).
    enter: str = ""
    # A payment ask's `total_label:` — the label readings the sheet total
    # sits beside (`money.declared_total` reads the amount off that row).
    total_label: tuple[str, ...] = ()
    # The ask's own patience: the in-session poll cadence and how many
    # silent rounds before the session suspends for the next wake.
    wait_seconds: int = DEFAULT_ASK_WAIT_SECONDS
    silence_rounds: int = DEFAULT_ASK_ROUNDS


@dataclass(frozen=True)
class TellNode:
    """A `tell` move — message the user; the walk moves on once the
    send lands (a reply, if any, is the next wake's boot to read)."""

    id: str
    message: str


@dataclass(frozen=True)
class ActivateNode:
    """The `select` step — the channel boot's own, and its last: on
    the thread (its `enter`, the page before it), ONE parse_task call
    over the enabled playbooks; a positive answer becomes the walk's
    baton (the program the conductor drives next), anything else ends
    the boot quietly with the model standing on the thread. `max_scrolls`
    bounds parse_task's scroll-for-history escape."""

    id: str
    enter: str
    max_scrolls: int
    irreversible: str | None = None  # `Checked`'s obligation; never set here


class Checked(Protocol):
    """A node the walk checks a page for before it runs — a `do`, an
    acting `agent`, the boot's `select`. What `enter_gate` and
    `recover_or_handover` read: where it must be, and whether money
    forbids recovering it."""

    @property
    def id(self) -> str: ...

    @property
    def enter(self) -> str: ...

    @property
    def irreversible(self) -> str | None: ...


@dataclass(frozen=True)
class RecoverHand:
    """One recovery hand — a bare gesture (`tool`), a tap of a declared
    `landmark`, or one argument-less `macro`."""

    tool: str | None = None
    landmark: str | None = None
    macro: str | None = None


@dataclass(frozen=True)
class Recovery:
    """A page's declared `recover:` — which hand runs for which reading
    of the deviation, and how many `tries` this page gets in one walk.
    `covered` fires when the page itself reads under a sheet or popup;
    `locked` when the phone shows its lock screen; `elsewhere` for any
    other screen. The flat form (`recover: go_back`) declares one hand
    for all three."""

    covered: RecoverHand | None = None
    elsewhere: RecoverHand | None = None
    locked: RecoverHand | None = None
    tries: int = DEFAULT_RECOVER_LIMIT

    def hand_for(self, reading: str) -> RecoverHand | None:
        """The hand declared for one of `RECOVER_READINGS`."""
        if reading == READING_COVERED:
            return self.covered
        if reading == READING_LOCKED:
            return self.locked
        return self.elsewhere

    @property
    def hands(self) -> tuple[RecoverHand, ...]:
        return tuple(
            h for h in (self.covered, self.elsewhere, self.locked) if h is not None
        )


Node = DoNode | AgentNode | AskNode | TellNode | ActivateNode


@dataclass(frozen=True)
class Playbook:
    """One validated playbook. `parse_playbook` is the only producer, so a
    Playbook is correct by construction against its pack (declared pages,
    pack macros, landmarks)."""

    app: str
    name: str
    description: str
    enabled: bool
    inputs: tuple[PlaybookInput, ...]
    nodes: tuple[Node, ...]  # the route's MOVES, compiled (waypoints derived away)
    # The route's first waypoint — where the walk must be at start (it
    # is also the first move's derived enter, which is what the runtime
    # actually checks).
    start: str = ""
    # The embedded macros — do bodies (`<playbook>.<move>`) and
    # resume/recover bodies (`<playbook>.<name>.<role>`). The node field
    # holds the same synthesized name, so dispatch is name-keyed either
    # way. `qualified_inline` is the registry door.
    inline_macros: dict[str, Macro] = field(default_factory=dict)
    # Declared recovery, page name → its hands: a mismatched page runs
    # ITS hand for the reading, or hands over when it declares none.
    recovers: dict[str, Recovery] = field(default_factory=dict)

    @property
    def activates(self) -> bool:
        """Whether this is the boot — a route ending in `select`, the
        one walk that hands a baton on (the compiler admits the step in
        the channel pack's boot file only)."""
        return any(isinstance(n, ActivateNode) for n in self.nodes)

    def first_unsettled(self, outputs: dict[str, str]) -> int:
        """Where a walk (re)starts: the route top, past any COMPLETED
        pure-text agent — its outputs are recorded, and re-deriving them
        (a recover hand's walk-from-the-top) could silently change them.
        Never further: a page that happens to match a later move's
        landing proves nothing about the moves before it."""
        for i, node in enumerate(self.nodes):
            settled = (
                isinstance(node, AgentNode)
                and not node.tools
                and all(f"{node.id}.{f}" in outputs for f in node.return_fields)
            )
            if not settled:
                return i
        return len(self.nodes)


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
    """What a playbook validates against: the app's declared pages, its
    private macros (name → parsed Macro, or the parse error string), and
    its landmarks."""

    app: str
    pages: dict[str, PageDecl]
    macros: dict[str, Macro]
    macro_errors: dict[str, str]
    # The raw playbook files (`<name>.yml` beside the manifest) — parsed
    # per entry by `scan_playbooks`, so one broken walk excludes itself,
    # never the pack; the files that would not load ride as errors.
    playbook_docs: dict = field(default_factory=dict)
    playbook_errors: dict[str, str] = field(default_factory=dict)
    # The manifest's `pages: <name>: recover:` hands, RAW by page name —
    # the route compiler resolves them (its grammar, its resolver) and
    # every route inherits them for a shared page unless it declares
    # its own.
    page_recovers: dict[str, Any] = field(default_factory=dict)
    # The pack's declared fixed spots (`landmarks:`) — recover hands and
    # agent grants name them. See `pages.Landmark`.
    landmarks: dict[str, Landmark] = field(default_factory=dict)
