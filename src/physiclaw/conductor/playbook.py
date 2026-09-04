"""A pack → validated `Playbook`s — the model, the pack, and the ref
grammar's two halves (validate at parse, `fill_refs` at run).

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
    entry     ::= "page" name [anchors] [forbid] [scrollable] [recover]
                | "start" name macro          # the unconditional cold-launch
                | "do" name [with] [macro] [irreversible]
                | "agent" name prompt [tools] [give] [returns] [limit]
                  [context] [irreversible]   # the step handed to the model
                | "ask" name approve message yes no [total] [wait] [resume]
                                              # payment: resume required when
                                              # a screen move follows
                | "tell" name message
                | "activate" name [limit]     # channel/boot only, and last:
                                              # read the thread, hand a
                                              # playbook the baton
    recover   ::= hand [limit]                # one hand for any deviation
                | {[occluded: hand] [elsewhere: hand] [locked: hand] [limit]}
                # a manifest page's recover: is inherited by every
                # route; a route's own replaces it whole for that walk
    hand      ::= {tool [with]} | {macro}
    macro     ::= name                    # macros/<name>/ — a directory macro
                | {[inputs] steps}        # inline — MACRO.yml grammar minus
                                          # name/description/enabled

The route's shape IS the contract: an optional prefix of pure-text
`agent` steps and one `start` opens it, the first page is the start
contract, every `do` and every acting `agent` is followed by the page
it lands on (its landing check — the reflector, enforced by shape),
and a page may declare its own `recover:` hand — one hand for any
deviation, or one per reading (`occluded:` a sheet over the page
itself, `locked:` the phone's lock screen, `elsewhere:` any other
screen), each page bounded by its own `limit:` under the walk-wide
ceiling. Moves fall through in
route order — there is no routing and no loop; whatever needs judgment
is an `agent` step inside the author's fence, and whatever needs a
human is an `ask`. Money runs in code: an `irreversible: payment` move
directly follows the `ask` that approves it (`route.py` lints it).

What the playbook declares is what runs — no more, no less. A page
without `recover:` hands over; an agent step runs the author's prompt
with the tools, landmarks, macros, and context the author listed and
nothing else; an ask's reply is read against the `yes:`/`no:` words
the ask declares, and anything they do not cover is the model's to
read; a payment ask names the label its `total:` sits beside, and
`wait:` is its own patience.

Wiring is by placeholder, and every ref is dotted: `{inputs.name}` reads
a declared input, `{move.field}` reads an EARLIER agent step's declared
return field, `{ask.total}` a payment ask's quoted total. A
bare `{name}` is a load error. Dotted refs are playbook-level — resolved
to plain strings before any macro sees them, so pack macros keep the
stock single-name template grammar.

This module owns the model and the pack; the route compiler and its
lints live in `route.py`.
"""

import re
from dataclasses import dataclass, field
from typing import Any, Protocol

from physiclaw.common import paths
from physiclaw.common.paths import PACK_FILENAME
from physiclaw.conductor import specfile
from physiclaw.conductor.limits import (
    DEFAULT_ASK_ROUNDS,
    DEFAULT_ASK_WAIT_SECONDS,
    DEFAULT_RECOVER_LIMIT,
)
from physiclaw.conductor.pages import (
    CHANNEL_APP,
    Landmark,
    PageDecl,
    PagesError,
    collect_page_decls,
    collect_page_recovers,
    pack_landmarks,
    parse_pages_data,
)
from physiclaw.macros import parse as macro_parse
from physiclaw.macros import store as macro_store
from physiclaw.macros.model import Macro, MacroError, MacroInput

# The three readings a page's `recover:` may key its hands by: the page
# itself under an overlay, the phone's lock screen (where taps do not
# land — only `unlock_phone` helps), or any other screen.
READING_OCCLUDED = "occluded"
READING_ELSEWHERE = "elsewhere"
READING_LOCKED = "locked"
RECOVER_READINGS = (READING_OCCLUDED, READING_ELSEWHERE, READING_LOCKED)
# `{root.name}` — the playbook's own ref grammar, always dotted
# (`inputs.` / an earlier agent step's id). The root follows the
# move-name grammar (hyphens included — `{pick-into-cart.message}`);
# the field stays input-shaped. Dotted refs are deliberately NOT part
# of the macro template layer (its tokenizer rejects them); same
# `{{`/`}}` escapes, same fail-at-load-on-stray-brace rule.
REF_RE = re.compile(r"\{\{|\}\}|\{([a-z0-9][a-z0-9_-]*\.[a-z][a-z0-9_]*)\}|[{}]")

# The one irreversible class: money. A payment move is entered only as
# the fall-through of an `ask` with `approve: payment` (`route.py`).
IRREVERSIBLE_CLASSES = ("payment",)

# The ref grammar's one global root — `{inputs.name}` — rejected as a
# move name so an agent's `{move.field}` outputs can never shadow it.
# (`ask` is not global: it exists only inside a payment ask's own
# messages, where the money slots win.)
INPUTS_ROOT = "inputs"

PACK_MACROS_DIRNAME = "macros"

_PLAY_KEYS = {"name", "description", "enabled", "inputs", "route"}


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
    # A payment ask's `total:` — the label readings the sheet total sits
    # beside (`money.declared_total` reads the amount off that row).
    total: tuple[str, ...] = ()
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
    """The `activate` step — the channel boot's own, and its last: on
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
    acting `agent`, the boot's `activate`. What `enter_gate` and
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
    """One recovery hand — ONE gesture (`tool`, a `tap` taking its
    `landmark`) or one argument-less `macro`."""

    tool: str | None = None
    landmark: str | None = None
    macro: str | None = None


@dataclass(frozen=True)
class Recovery:
    """A page's declared `recover:` — which hand runs for which reading
    of the deviation, and how many times this page may recover in one
    walk. `occluded` fires when the page itself reads under an overlay
    (a sheet, a popup); `locked` when the phone shows its lock screen;
    `elsewhere` for any other screen. The flat form
    (`recover: {tool: ...}`) declares one hand for all three."""

    occluded: RecoverHand | None = None
    elsewhere: RecoverHand | None = None
    locked: RecoverHand | None = None
    limit: int = DEFAULT_RECOVER_LIMIT

    def hand_for(self, reading: str) -> RecoverHand | None:
        """The hand declared for one of `RECOVER_READINGS`."""
        if reading == READING_OCCLUDED:
            return self.occluded
        if reading == READING_LOCKED:
            return self.locked
        return self.elsewhere

    @property
    def hands(self) -> tuple[RecoverHand, ...]:
        return tuple(
            h for h in (self.occluded, self.elsewhere, self.locked) if h is not None
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
        """Whether this is the boot — a route ending in `activate`, the
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


def qualified_macro(app: str, name: str) -> str:
    """The qualified `app/name` dispatch key — the ONE spelling of the
    convention the run_macro handler resolves (user macro names can
    never contain "/", so no collision). Lives beside `Pack`, the owner
    of macro dicts — every pack site (channel included) consumes it."""
    return f"{app}/{name}"


def split_ref(ref: str) -> tuple[str, str]:
    """`<app>/<playbook>` → (app, playbook) — the one parse of the ref
    every skin takes (the CLI exits on the error, the studio answers
    400). Raises PlaybookError."""
    app, sep, name = ref.partition("/")
    if not sep or not app or not name or "/" in name:
        raise PlaybookError(f"{ref!r} is not <app>/<playbook>")
    return app, name


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


def qualified_all(app: str, pack: Pack) -> dict[str, Macro]:
    """Everything a walk of this pack can dispatch: the directory macros
    plus every playbook's inline bodies — disabled playbooks included
    (gating is the caller's filter, never the dispatch table)."""
    macros = qualified_pack(app, pack)
    for entry in scan_playbooks(app, pack):
        if entry.spec is not None:
            macros.update(qualified_inline(app, entry.spec))
    return macros


# ---------- pack loading ----------


def load_pack(app: str) -> Pack:
    """The app pack, whole: the manifest (`PLAYBOOK.yml` — what the app
    is and what its routes share: meta, placeholders, landmarks,
    pages), the playbook files beside it (raw, parsed per entry by
    `scan_playbooks`), and the recorded macros. A broken pack macro is
    carried as its error string so the playbook referencing it fails
    with the cause; a broken playbook file rides the same way."""
    doc = specfile.load_pack_doc(app, PlaybookError)
    if doc is None:
        raise PlaybookError(f"no pack {app!r} on disk (missing {PACK_FILENAME})")
    _check_pack_meta(doc, app)
    root = paths.pack_root(app)
    if app == CHANNEL_APP:
        # The boot file is a template the user owns, materialized beside
        # an existing channel pack on first look (the ios pack's
        # pattern): a channel recorded before the boot was a file keeps
        # its wake, and every door — wake, step, run, check — sees the
        # same pack. (Lazy: scaffold imports this module.)
        from physiclaw.conductor import scaffold

        scaffold.ensure_channel_boot(root)
    docs, pb_errors = specfile.load_playbook_docs(app, PlaybookError, root)
    try:
        # Appendix + route-declared waypoints across every playbook
        # file, one namespace — the matcher and every playbook validate
        # against the same set.
        pages = parse_pages_data(collect_page_decls(doc, docs), app)
    except PagesError as e:
        raise PlaybookError(f"{app}/{PACK_FILENAME} pages: {e}") from e
    try:
        landmarks = pack_landmarks(doc)
    except PagesError as e:
        raise PlaybookError(f"{app}/{PACK_FILENAME} landmarks: {e}") from e
    macros: dict[str, Macro] = {}
    errors: dict[str, str] = {}
    macros_root = root / PACK_MACROS_DIRNAME
    if macros_root.is_dir():
        # One scanner for both macro roots: traversal guard, dot-dir
        # convention, and the broad-except lesson live in `store.scan`.
        for entry in macro_store.scan(macros_root):
            if entry.spec is not None:
                macros[entry.dir_name] = entry.spec
            else:
                errors[entry.dir_name] = entry.error or "invalid"
    return Pack(
        app=app,
        pages=pages,
        macros=macros,
        macro_errors=errors,
        playbook_docs=docs,
        playbook_errors=pb_errors,
        landmarks=landmarks,
        page_recovers=collect_page_recovers(doc),
    )


def _check_pack_meta(doc: dict, app: str) -> None:
    """The manifest's meta, every field optional: `app` (which app this
    pack automates) must equal the directory when present — the folder
    IS the app, the field catches a pack copied under the wrong name;
    `description` is real prose when present (`install` prints it);
    `placeholders` (install-time constants, validated here so `check`
    catches a malformed map before install prompts read it) is
    name → {description, [example]}."""
    if "app" in doc:
        declared = require_str(doc.get("app"), "`app`")
        if declared != app:
            raise PlaybookError(
                f"app {declared!r} must equal the pack directory {app!r}"
            )
    if app == "pages":
        raise PlaybookError(
            "a pack cannot be named 'pages' — it is the page-reference root"
        )
    if "description" in doc:
        prose(doc.get("description"), "`description`")
    ph = doc.get("placeholders")
    if ph is None:
        return
    if not isinstance(ph, dict):
        raise PlaybookError("`placeholders` must be a mapping of TOKEN → spec")
    for key, spec in ph.items():
        where = f"placeholder {key!r}"
        if not isinstance(spec, dict) or set(spec) - {"description", "example"}:
            raise PlaybookError(f"{where} must be a {{description, example}} mapping")
        prose(spec.get("description"), f"{where}: `description`")


def scan_playbooks(app: str, pack: Pack | None = None) -> list[PlaybookEntry]:
    """Every playbook file of the pack, parsed against it — plus the
    files that would not load, as invalid entries. Callers that already
    hold the Pack thread it through so the spec file and macros are
    not re-read."""
    if pack is None:
        if not (paths.pack_root(app) / PACK_FILENAME).exists():
            return []
        pack = load_pack(app)
    out: list[PlaybookEntry] = [
        PlaybookEntry(app=app, name=n, error=e)
        for n, e in sorted(pack.playbook_errors.items())
    ]
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


def stray_dirs() -> list[str]:
    """Folders under a playbooks root holding YAML beside no manifest —
    an author who wrote a route and forgot the marker. Skips the `_`
    and `.` prefixes every lister does, and a name a home pack already
    claims (the home layer shadows the tree's)."""
    packs = set(list_apps())
    out: list[str] = []
    for root in paths.playbooks_dirs():
        if not root.is_dir():
            continue
        for d in sorted(root.iterdir()):
            if (
                d.is_dir()
                and not d.name.startswith(("_", "."))
                and d.name not in packs
                and any(d.glob("*.yml"))
            ):
                out.append(f"{root.name}/{d.name}")
    return out


def list_apps() -> list[str]:
    """Packs across the search path (the `paths.playbooks_dirs` layering),
    sorted — a PLAYBOOK.yml marks a pack."""
    return sorted(paths.marked_subdirs(paths.playbooks_dirs(), PACK_FILENAME))


# ---------- parsing ----------


def parse_playbook(text: str, name: str, pack: Pack) -> Playbook:
    """One playbook given as YAML text — the text-shaped door tests and
    tooling use; the live path is `scan_playbooks` over the pack's
    playbook files. Raises PlaybookError naming the offending field;
    never a partial spec."""
    data = specfile.load_yaml(text, PlaybookError)
    return _parse_playbook_data(data, name, pack)


def _parse_playbook_data(data: Any, name: str, pack: Pack) -> Playbook:
    """One playbook file's document → a validated Playbook. The file
    stem IS the name; the `name:` inside must agree with it."""
    # The compiler imports this module's model; the one import in the
    # other direction is deferred to the call so the two files stay a
    # pair without a cycle at load.
    from physiclaw.conductor.route import compile_route

    if not isinstance(data, dict):
        raise PlaybookError("a playbook must be a YAML mapping (key: value pairs)")
    unknown = sorted(set(map(str, data.keys())) - _PLAY_KEYS)
    if unknown:
        raise PlaybookError(f"unknown key(s): {', '.join(unknown)}")
    # A playbook names itself, like a macro and a skill do — and the
    # name must be the file's, so a copied file cannot lie about what
    # it is (the same rule `app` keeps with the pack folder).
    if "name" not in data:
        raise PlaybookError(
            f"every .yml beside the manifest is a playbook, and this one has "
            f"no `name:` — a playbook starts `name: {name}`"
        )
    check_name(name, "playbook name")
    declared = require_str(data.get("name"), "`name`")
    if declared != name:
        raise PlaybookError(
            f"name {declared!r} must equal the file name {name!r} ({name}.yml)"
        )
    description = prose(data.get("description"), "`description`")
    enabled = data.get("enabled", True)
    if not isinstance(enabled, bool):
        raise PlaybookError("`enabled` must be true or false")
    inputs = _parse_inputs(data.get("inputs", {}))
    input_names = {i.name for i in inputs}
    route = compile_route(
        data.get("route"), playbook=name, input_names=input_names, pack=pack
    )
    return Playbook(
        app=pack.app,
        name=name,
        description=description,
        enabled=enabled,
        inputs=inputs,
        nodes=tuple(route.nodes),
        start=route.start,
        inline_macros=route.inline,
        recovers=route.recovers,
    )


def field_name(name: Any, what: str) -> str:
    """The one naming rule for the values `{x.y}` refs read — an agent's
    return fields, spelled like the inputs they sit beside."""
    if not isinstance(name, str) or not specfile.INPUT_NAME_RE.match(name):
        raise PlaybookError(
            f"{what} {name!r} must be lowercase, start with a "
            "letter, and contain only letters/digits/underscores"
        )
    return name


def _parse_inputs(raw: Any) -> tuple[PlaybookInput, ...]:
    """`inputs:` through the macro grammar's parser, the error class
    translated at this one seam."""
    try:
        return macro_parse.parse_inputs(raw)
    except MacroError as e:
        raise PlaybookError(str(e)) from e


# ---------- the ref grammar ----------


def refs_in(text: str, where: str) -> set[str]:
    """Every dotted ref a template names; a stray brace is a load error."""
    names: set[str] = set()
    for m in REF_RE.finditer(text):
        if m.group(0) in ("{{", "}}"):
            continue
        if m.group(1) is None:
            raise PlaybookError(
                f"{where}: stray {m.group(0)!r} — every ref is dotted: "
                "{inputs.name} for an input, {node.field} for an earlier "
                "agent step's output, {{ / }} for a literal brace"
            )
        names.add(m.group(1))
    return names


def check_refs(
    refs: set[str],
    input_names: set[str],
    payloads: dict[str, tuple[str, ...]],
    where: str,
) -> None:
    """Every ref resolves: an `inputs.*` to a declared input, anything
    else to an EARLIER move's declared output field."""
    for ref in sorted(refs):
        root, _, fld = ref.partition(".")
        if root == INPUTS_ROOT:
            if fld not in input_names:
                raise PlaybookError(f"{where}: {{{ref}}} not declared under `inputs`")
        elif root not in payloads:
            raise PlaybookError(
                f"{where}: {{{ref}}} references move {root!r}, which "
                "is not an EARLIER agent step — outputs wire forward "
                "only, in route order"
            )
        elif fld not in payloads[root]:
            raise PlaybookError(
                f"{where}: {{{ref}}}: move {root!r} has no output "
                f"{fld!r} (has: {', '.join(payloads[root]) or '(none)'})"
            )


def check_arg_refs(
    value: Any,
    input_names: set[str],
    payloads: dict[str, tuple[str, ...]],
    where: str,
) -> None:
    """`check_refs` over a `with:` value — recursing into lists and
    mappings exactly as `fill_refs` fills them."""
    if isinstance(value, str):
        check_refs(refs_in(value, where), input_names, payloads, where)
    elif isinstance(value, list):
        for v in value:
            check_arg_refs(v, input_names, payloads, where)
    elif isinstance(value, dict):
        for v in value.values():
            check_arg_refs(v, input_names, payloads, where)


def fill_refs(value: Any, values: dict[str, str], where: str) -> Any:
    """A `with:` value with its refs resolved from `values` — the runtime
    half of the ref grammar, beside REF_RE so no consumer re-derives the
    braces. `values` is keyed by the dotted ref spellings themselves
    (`inputs.name`, `node.field`). Recurses into lists/dicts exactly as
    `check_arg_refs` validates them. Raises PlaybookError on a ref with
    no value (e.g. an agent output not yet recorded)."""
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


def require_live(spec: Playbook, pack: Pack) -> None:
    """The live rule, spelled once: what a real wake needs of a playbook
    — enabled, with every referenced pack macro enabled. A resuming
    suspension and the boot must satisfy it; a rehearsal deliberately
    need not (you rehearse BEFORE you enable). Raises PlaybookError
    naming the gap."""
    ref = f"{spec.app}/{spec.name}"
    if not spec.enabled:
        raise PlaybookError(f"{ref} is disabled — set `enabled: true` once rehearsed")
    disabled = disabled_macros(spec, pack)
    if disabled:
        raise PlaybookError(
            f"{ref} references disabled pack macro(s): {', '.join(disabled)} — "
            "rehearse, then enable"
        )


def disabled_macros(spec: Playbook, pack: Pack) -> list[str]:
    """Referenced pack macros still disabled — the live-readiness rule:
    `playbooks check` warns about it and the boot will not offer such
    a playbook at all. Covers every dispatching role: do moves, an ask's
    `resume:`, a page's `recover:` hands, and an agent's granted macros.
    Safe unguarded access: parse
    validated every directory name against `pack.macros`."""
    named: set[str] = set()
    for recovery in spec.recovers.values():
        named.update(h.macro for h in recovery.hands if h.macro is not None)
    for n in spec.nodes:
        if isinstance(n, DoNode):
            named.add(n.macro)
        elif isinstance(n, AskNode) and n.resume is not None:
            named.add(n.resume)
        elif isinstance(n, AgentNode):
            named.update(n.macros)
    # One rule, no inline special case: each name resolves through the
    # merged view, and an inline macro is enabled by construction — its
    # gate is the playbook's own `enabled:` — so only directory names
    # can report.
    return sorted(
        m for m in named if not (spec.inline_macros.get(m) or pack.macros[m]).enabled
    )
