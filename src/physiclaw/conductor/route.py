"""The route compiler — `route:` → compiled moves, the start page, and
the declared recovery hands, with the lints that make the shape a
contract.

Waypoints do not become nodes — they become the adjacent moves'
checks: a `do`'s (and an acting `agent`'s) enter is the nearest
preceding page, its verify the page that must immediately follow it
(the reflector, enforced by shape). The route opens with an optional
prefix of pure-text `agent` steps (no screen — they may run before the
phone is touched) and an optional `start` move (the unconditional
cold-launch, verified by the page that follows it); the first page
waypoint is the start contract either way. A `page:` may carry
`recover:` — its declared recovery hand.

Control flow is route order, full stop: moves fall through (past the
last entry = done), an `ask` falls through once the user approved, a
`tell` suspends the session and the next wake continues past it. The
one thing the compiler must PROVE is about money: an
`irreversible: payment` move directly follows the `ask` that approves
it, so consent never desynchronizes from the sheet it quotes.

An inline macro is single-use by construction — an anonymous body has
no name for another site to reference: its name is synthesized
`<playbook>.<move>` (a do body) or `<playbook>.<name>.<role>` (a
resume/recover body) — dot-joined, a spelling no directory macro or
move can take, so the pack-wide dispatch namespace stays collision-free
by construction. Grammar boundary: everything under an inline `macro:`
IS a macro (single-name `{x}` templates, fed by the move's `with:`),
everything outside stays dotted — the same split as the file boundary,
moved to the key.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, TypeVar

from physiclaw.conductor import context, reply
from physiclaw.conductor.calls import AGENT_TOOLS, CONTRACT_FIELDS, RESERVED_KEYS
from physiclaw.conductor.limits import (
    DEFAULT_AGENT_CALLS,
    DEFAULT_AGENT_SCROLLS,
    DEFAULT_ASK_ROUNDS,
    DEFAULT_ASK_WAIT_SECONDS,
    DEFAULT_RECOVER_LIMIT,
    MAX_AGENT_CALLS,
    MAX_ASK_ROUNDS,
    MAX_ASK_WAIT_SECONDS,
    MAX_NODES,
    MAX_PROMPT_LEN,
    MAX_RECOVER_ACTIONS,
    MAX_RETURNS,
    MIN_ASK_WAIT_SECONDS,
)
from physiclaw.conductor.pages import (
    PAGE_DECL_FIELDS,
    RESERVED_APPS,
    PagesError,
    parse_pages_data,
    route_decl,
)
from physiclaw.conductor.playbook import (
    INPUTS_ROOT,
    IRREVERSIBLE_CLASSES,
    PACK_MACROS_DIRNAME,
    READING_ELSEWHERE,
    READING_OCCLUDED,
    RECOVER_READINGS,
    AgentNode,
    AskNode,
    DoNode,
    Node,
    Pack,
    PlaybookError,
    RecoverHand,
    Recovery,
    TellNode,
    check_arg_refs,
    check_name,
    check_refs,
    field_name,
    prose,
    refs_in,
    require_str,
)
from physiclaw.macros.model import Macro, MacroError, checked_readings
from physiclaw.macros.parse import parse_inline_macro

# Agent-step grammar. `tools` is the closed per-episode gesture allowlist
# (`calls.AGENT_TOOLS`, the keys of the tool → verbs map the runner
# reads); the bounds on calls, scrolls, prompt, and returns are
# `limits.py`'s.
_AGENT_LIMIT_KEYS = {"calls", "scrolls"}
_ASK_WAIT_KEYS = {"seconds", "rounds"}

# The declared-recovery hand: one gesture (`tool`, with `with:
# landmarks.<name>` for a tap) or a macro. Closed tool vocabulary — a
# recover hand resets state, it does not navigate (the route does that).
# A page declares one hand for any deviation, or one per reading
# (`occluded:` / `elsewhere:`), plus its own `limit:`.
RECOVER_TOOLS = ("force_quit", "go_back", "home_screen", "unlock_phone", "tap")
_HAND_KEYS = {"tool", "with", "macro"}
_RECOVER_KEYS = _HAND_KEYS | {"limit", *RECOVER_READINGS}

# What an agent may be granted by name (`give:`): a landmark it may tap
# blind, or a pack macro it may run.
GRANT_LANDMARKS = "landmarks"
GRANT_MACROS = "macros"
_GRANT_ROOTS = (GRANT_LANDMARKS, GRANT_MACROS)

# Route entry vocabularies. An entry's KIND is its leading key and the
# value is the entry's name — the map-key-is-the-name doctrine, applied
# to the route. Page-declaration fields come from `pages.py`'s ONE
# spelling (PAGE_DECL_FIELDS) — their content is validated there; they
# appear here only so the unknown-key check names them as legal.
ENTRY_KINDS = ("page", "start", "do", "agent", "ask", "tell")
_ENTRY_KEYS = {
    "page": {"page", "recover", *PAGE_DECL_FIELDS},
    "start": {"start", "macro"},
    "do": {"do", "with", "macro", "irreversible"},
    "agent": {
        "agent",
        "prompt",
        "tools",
        "give",
        "returns",
        "limit",
        "context",
        "irreversible",
    },
    "ask": {"ask", "approve", "message", "yes", "no", "total", "wait", "resume"},
    "tell": {"tell", "message", "no"},
}

# The shape `_macro_resolver` returns.
_MacroResolve = Callable[..., Macro]

_T = TypeVar("_T")


@dataclass
class _Ctx:
    """What every entry parser reads — one object instead of the same
    five arguments threaded through each signature. `payloads` grows as
    the compile pass advances (an agent's return fields, in route
    order), so a `{move.field}` ref is defined-before-use by
    construction."""

    playbook: str
    pack: Pack
    input_names: set[str]
    resolve: "_MacroResolve"
    payloads: dict[str, tuple[str, ...]] = field(default_factory=dict)

    def payloads_with_total(self) -> dict[str, tuple[str, ...]]:
        """The refs a payment step may quote: every recorded return
        field plus the ONE gate slot, `{ask.total}` — the consented
        amount its ask binds (`_check_money` keeps the two adjacent)."""
        return {**self.payloads, "ask": ("total",)}


@dataclass(frozen=True)
class CompiledRoute:
    """What `compile_route` hands back: the moves, the start page, each
    page's declared recovery hand, and the inline macro bodies under
    their synthesized names."""

    nodes: list[Node]
    start: str
    recovers: dict[str, Recovery]
    inline: dict[str, Macro]


def compile_route(
    raw: Any,
    *,
    playbook: str,
    input_names: set[str],
    pack: Pack,
) -> CompiledRoute:
    """`route:` → the compiled route (see `CompiledRoute`): the shape
    prepass first (every rule about WHERE an entry may sit), then one
    forward pass compiling the moves against the waypoints around them,
    then the lints that need the whole route."""
    entries, wp_ids, start, page_names = _shape(raw, pack)
    inline: dict[str, Macro] = {}
    ctx = _Ctx(playbook, pack, input_names, _macro_resolver(playbook, pack, inline))
    moves: list[Node] = []
    seen: dict[str, int] = {}
    recovers: dict[str, Recovery] = {}
    current_page: str | None = None
    for i, (kind, name, entry) in enumerate(entries):
        pos = i + 1
        if kind == "page":
            current_page = wp_ids[i]
            if "recover" in entry:
                rpage = wp_ids[i]
                assert rpage is not None
                if "." in rpage:
                    raise PlaybookError(
                        f"route entry {pos}: {rpage!r} is a reserved built-in "
                        "— packs declare recovery for their own pages only"
                    )
                hand = _parse_recover(
                    ctx, entry["recover"], f"route entry {pos}", rpage
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
        if name == INPUTS_ROOT:
            raise PlaybookError(
                f"{where}: name {name!r} is a reserved ref root — "
                "{inputs.*} always reads the declared inputs"
            )
        if name in page_names:
            raise PlaybookError(
                f"{where}: {name!r} is also a page on this route — moves "
                "and pages share one namespace, so the names must not collide"
            )
        if name in seen:
            raise PlaybookError(
                f"{where}: duplicate move name {name!r} (entry {seen[name]} "
                "already uses it) — refs address moves by name, so they "
                "must be unique"
            )
        seen[name] = pos
        where = f"move {name!r}"
        args = entry.get("with", {})
        if not isinstance(args, dict):
            raise PlaybookError(f"{where}: `with` must be a mapping of arguments")
        check_arg_refs(args, input_names, ctx.payloads, where)
        nxt = wp_ids[i + 1] if i + 1 < len(entries) else None
        if kind == "do":
            if nxt is None:
                raise PlaybookError(
                    f"{where}: a `do` must be followed by the page it lands "
                    "on — the landing check is what proves the move ran"
                )
            assert current_page is not None  # checked by the prefix rule
            moves.append(_parse_do(ctx, where, name, entry, args, current_page, nxt))
        elif kind == "start":
            assert nxt is not None  # `start` sits immediately before a page
            spec = ctx.resolve(entry.get("macro"), where, name)
            moves.append(
                DoNode(
                    id=name,
                    macro=spec.name,
                    args={},
                    enter="",  # unconditional: the start runs from anywhere
                    verify=nxt,
                )
            )
        elif kind == "agent":
            agent = _parse_agent(ctx, where, name, entry, current_page, nxt)
            ctx.payloads[agent.id] = agent.return_fields
            moves.append(agent)
        elif kind == "ask":
            moves.append(_parse_ask(ctx, where, name, entry, current_page))
        else:  # tell
            message, _ = _entry_message(ctx, where, entry, ctx.payloads)
            no = _reply_words(entry, "no", where) if "no" in entry else []
            moves.append(TellNode(id=name, message=message, no=tuple(no)))
    if len(moves) > MAX_NODES:
        raise PlaybookError(f"too many moves ({len(moves)} > {MAX_NODES})")
    _check_money(moves)
    _check_resume(moves)
    return CompiledRoute(nodes=moves, start=start, recovers=recovers, inline=inline)


def _shape(
    raw: Any, pack: Pack
) -> tuple[list[tuple[str, str, dict]], list[str | None], str, set[str]]:
    """The route's shape, proved before any move is compiled: a
    non-empty list whose first page is the start contract, at most one
    `start` sitting right before it, only pure-text agents above it,
    and every waypoint's id resolved (`_waypoint_id`, one grammar at
    every door) so a move can read the page after it in one look.
    Returns (classified entries, the waypoint id per entry or None,
    the start page id, the set of page ids on the route)."""
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
            "the route needs at least one move (start/do/agent/ask/tell)"
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
    page_names: set[str] = set()
    for i, (kind, name, entry) in enumerate(entries):
        if kind != "page":
            wp_ids.append(None)
            continue
        pid = _waypoint_id(i + 1, name, entry, pack, declared_here)
        wp_ids.append(pid)
        page_names.add(pid)
    start = wp_ids[first_page]
    assert start is not None  # first_page indexes a page entry
    return entries, wp_ids, start, page_names


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
    return kind, require_str(entry.get(kind), f"{where}: `{kind}`"), entry


def _waypoint_id(pos: int, name: str, entry: dict, pack: Pack, declared: set) -> str:
    """One page waypoint's id. Own-pack pages are written bare (the route
    IS the pack's context); the reserved built-ins stay dotted
    (`ios.<page>`/`channel.<page>`) and can only be referenced, never
    declared here."""
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
        if not entry.keys().isdisjoint(PAGE_DECL_FIELDS):
            raise PlaybookError(
                f"{where}: {name!r} is a reserved built-in — it cannot be "
                "declared from a pack"
            )
        return name
    check_name(name, f"{where}: `page`")
    decl = route_decl(entry)
    if decl is not None and name not in pack.pages:
        # Validate the in-place declaration's CONTENT here too, so the
        # text door (`parse_playbook` — tests, tooling) enforces the
        # same page grammar the pack door does via `collect_page_decls`;
        # a playbook green at one door must not go red at the other.
        # (The pack door already parsed it when the pack knows the page.)
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


# ---------- shared field rules ----------


def _unique_list(raw: Any, where: str, check: Callable[[Any], _T]) -> list[_T]:
    """A list of distinct entries, each validated (and normalized) by
    `check` — the one shape `tools`, `give`, and the reply words share."""
    if not isinstance(raw, list):
        raise PlaybookError(f"{where} must be a list")
    out: list[_T] = []
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


def _limit_int(value: Any, where: str, lo: int, hi: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not lo <= value <= hi:
        raise PlaybookError(f"{where} must be {lo}–{hi} (got {value!r})")
    return value


def _entry_message(
    ctx: _Ctx, where: str, entry: dict, payloads: dict[str, tuple[str, ...]]
) -> tuple[str, set[str]]:
    """A REQUIRED authored `message:` — the exact text sent to the user;
    only the author knows the user's language, so the conductor composes
    no prose around it. Refs held to the same defined-before-use rules
    as `with:` values; returned with them so the ask lints can inspect."""
    text = prose(entry.get("message"), f"{where}: `message`")
    refs = refs_in(text, f"{where}: `message`")
    check_refs(refs, ctx.input_names, payloads, f"{where}: `message`")
    return text, refs


def _reply_words(entry: dict, key: str, where: str) -> list[str]:
    """An ask's `yes:` / `no:` — the whole-message replies it reads, a
    non-empty list of distinct strings, stored in `reply.normalize`
    space so every reader compares without re-normalizing."""
    out = _unique_list(
        entry.get(key),
        f"{where}: `{key}`",
        lambda w: reply.normalize(require_str(w, f"{where}: `{key}` entry")),
    )
    if not out:
        raise PlaybookError(f"{where}: `{key}` must list at least one reply word")
    return out


def _context_entries(entry: dict, where: str) -> list[str]:
    """An agent's `context:` — the sources it loads beside its prompt."""

    def _one(item: Any) -> str:
        bad = context.check_entry(item)
        if bad is not None:
            raise PlaybookError(f"{where}: `context` entry {item!r} {bad}")
        return item

    return _unique_list(entry.get("context", []), f"{where}: `context`", _one)


def _landmark_name(ctx: _Ctx, value: Any, where: str) -> str:
    """A `landmarks.<name>` reference resolved to its bare name — ONE
    spelling for `give:` entries and a recover tap's `with:`. The name
    half rides the shared name grammar (`check_name`), so a landmark
    reference can never drift from the section's own naming rule."""
    prefix, _, name = value.partition(".") if isinstance(value, str) else ("", "", "")
    if prefix != GRANT_LANDMARKS or not name:
        raise PlaybookError(f"{where}: {value!r} must look like `landmarks.<name>`")
    check_name(name, where)
    if name not in ctx.pack.landmarks:
        known = ", ".join(sorted(ctx.pack.landmarks)) or "(none)"
        raise PlaybookError(
            f"{where}: names landmark {name!r} — not declared under "
            f"`landmarks`. Declared: {known}"
        )
    return name


def _grant(ctx: _Ctx, value: Any, where: str, nid: str) -> tuple[str, str]:
    """One `give:` entry → (root, name): a `landmarks.<name>` the episode
    may tap blind, or a `macros.<name>` pack macro it may run — argument-
    less, like every helper hand, and never spelled like a fixed answer
    (`done`, `escalate`, a verb) that the episode legend already owns."""
    prefix, _, name = value.partition(".") if isinstance(value, str) else ("", "", "")
    if prefix not in _GRANT_ROOTS or not name:
        roots = " or ".join(f"`{r}.<name>`" for r in _GRANT_ROOTS)
        raise PlaybookError(f"{where}: {value!r} must look like {roots}")
    if name in RESERVED_KEYS:
        raise PlaybookError(
            f"{where}: {name!r} is a fixed episode answer — a granted "
            "landmark or macro cannot be spelled like one"
        )
    if prefix == GRANT_LANDMARKS:
        return prefix, _landmark_name(ctx, value, where)
    return prefix, _argless_macro(name, "give", where, nid, ctx.resolve)


# ---------- macros ----------


def _macro_resolver(
    playbook: str, pack: Pack, inline: dict[str, Macro]
) -> _MacroResolve:
    """The name-or-inline resolution every macro-carrying slot shares —
    a do's `macro:`, an ask's `resume:`, a page's `recover:`. ONE home
    for the whole idiom: the synthesized-name rule
    (`<playbook>.<node>[.<role>]`, dot-joined so it can never collide
    with a directory macro — `check_name` rejects dots), the MacroError
    framing, the inline registry, and the directory validation (a
    broken macro reports its cause, an unknown one lists what exists) —
    so the slots can never drift. Returns the resolved Macro; its
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
        mname = require_str(raw, f"{where}: `{slot}`")
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


def _argless_macro(
    raw: Any, key: str, where: str, nid: str, resolve: _MacroResolve
) -> str:
    """Resolve one argument-less pack macro for a helper-hand slot
    (`resume:`, `recover:`). The slot may wrap its body one level
    (`{macro: ...}`) — unwrapped HERE, the rule's one home, so the
    resolver sees the same shapes a `do` does. Both roles dispatch with
    no arguments, so a required input could only abort at run time —
    right after a confirmed ask, at the worst moment — hence the lint."""
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


# ---------- the moves ----------


def _parse_do(
    ctx: _Ctx,
    where: str,
    nid: str,
    entry: dict,
    args: dict,
    enter: str,
    verify: str,
) -> DoNode:
    """A `do` move. Its name IS its macro — the directory macro of that
    name, or the `macro:` beside it (an inline body, or another name
    when one directory macro serves two moves). `enter`/`verify` arrive
    derived from the route's waypoints."""
    if "." in enter or "." in verify:
        raise PlaybookError(
            f"{where}: a `do` runs on this pack's own pages — a reserved "
            "built-in cannot frame it"
        )
    spec = ctx.resolve(entry.get("macro", nid), where, nid)
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
    return DoNode(
        id=nid,
        macro=macro,
        args=dict(args),
        enter=enter,
        verify=verify,
        irreversible=_irreversible_class(entry, where),
    )


def _parse_agent(
    ctx: _Ctx,
    where: str,
    nid: str,
    entry: dict,
    current_page: str | None,
    next_wp: str | None,
) -> AgentNode:
    """An `agent` move. No `tools` = a pure-text call (needs `returns`,
    no pages); tools = an acting episode framed by the adjacent
    waypoints exactly like a `do`. The prompt is the author's whole
    brief — refs validated here, filled once when the step opens."""
    prompt = require_str(entry.get("prompt"), f"{where}: `prompt`")
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
    grants = _unique_list(
        entry.get("give", []),
        f"{where}: `give`",
        lambda g: _grant(ctx, g, f"{where}: `give` entry", nid),
    )
    give = tuple(n for root, n in grants if root == GRANT_LANDMARKS)
    macros = tuple(n for root, n in grants if root == GRANT_MACROS)
    shared = sorted(set(give) & set(macros))
    if shared:
        raise PlaybookError(
            f"{where}: `give` names {', '.join(shared)} as both a landmark and "
            "a macro — the model answers by name, so the two must differ"
        )
    if give and "tap" not in tools:
        raise PlaybookError(
            f"{where}: `give` grants landmarks, but without `tap` the episode "
            "cannot press one — grant `tap` or drop the landmarks"
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
            field_name(fname, f"{where}: return field")
            if fname in CONTRACT_FIELDS:
                raise PlaybookError(
                    f"{where}: return field {fname!r} is one of the reply "
                    f"contract's own fields ({', '.join(sorted(CONTRACT_FIELDS))})"
                    " — rename it"
                )
            returns.append((fname, prose(desc, f"{where}: `returns.{fname}`")))

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
    elif max_scrolls == 0:
        raise PlaybookError(
            f"{where}: `scroll` is granted but `limit.scrolls` is 0 — the "
            "first scroll would hand over; raise it or drop the tool"
        )

    g_payloads = (
        ctx.payloads_with_total() if irreversible == "payment" else ctx.payloads
    )
    check_refs(
        refs_in(prompt, f"{where}: `prompt`"),
        ctx.input_names,
        g_payloads,
        f"{where}: `prompt`",
    )

    return AgentNode(
        id=nid,
        prompt=prompt,
        tools=tuple(tools),
        give=give,
        returns=tuple(returns),
        enter=enter,
        verify=verify,
        max_calls=max_calls,
        max_scrolls=max_scrolls,
        irreversible=irreversible,
        context=tuple(_context_entries(entry, where)),
        macros=macros,
    )


def _parse_ask(
    ctx: _Ctx, where: str, nid: str, entry: dict, current_page: str | None
) -> AskNode:
    approve = require_str(entry.get("approve"), f"{where}: `approve`")
    check_name(approve, f"{where}: `approve`")
    if approve == "payment":
        if current_page is None or "." in current_page:
            raise PlaybookError(
                f"{where}: a payment ask reads its total off the page before "
                "it — put the sheet's page waypoint immediately before the ask"
            )
    # A payment ask may quote the consent slot (a move literally named
    # `ask` is shadowed in this message — the money slot wins, both at
    # parse and at fill).
    g_payloads = ctx.payloads_with_total() if approve == "payment" else ctx.payloads
    message, msg_refs = _entry_message(ctx, where, entry, g_payloads)
    total: tuple[str, ...] = ()
    if approve == "payment":
        if "ask.total" not in msg_refs:
            raise PlaybookError(
                f"{where}: a payment ask's `message` must quote the sheet "
                "total — reference {ask.total} (the ask IS the consent record)"
            )
        if "total" not in entry:
            raise PlaybookError(
                f"{where}: a payment ask declares `total:` — the label the "
                "sheet total sits beside (e.g. 合计), read off that row only"
            )
        total = checked_readings(entry, where, require_str, PlaybookError, key="total")
    elif "total" in entry:
        raise PlaybookError(f"{where}: `total` goes with `approve: payment`")
    wait_seconds, rounds = _ask_wait(entry.get("wait"), where)
    resume = None
    if entry.get("resume") is not None:
        resume = _argless_macro(entry["resume"], "resume", where, nid, ctx.resolve)
    return AskNode(
        id=nid,
        approve=approve,
        message=message,
        yes=tuple(_reply_words(entry, "yes", where)),
        no=tuple(_reply_words(entry, "no", where)),
        resume=resume,
        enter=current_page or "",
        total=total,
        wait_seconds=wait_seconds,
        silence_rounds=rounds,
    )


def _ask_wait(raw: Any, where: str) -> tuple[int, int]:
    """`wait: {seconds, rounds}` — both optional, defaults visible in
    the scaffold; bounded by the engine's single-sleep cap and a sane
    number of silent rounds."""
    raw = {} if raw is None else raw
    if not isinstance(raw, dict):
        raise PlaybookError(f"{where}: `wait` must be a mapping")
    unknown = sorted(set(map(str, raw)) - _ASK_WAIT_KEYS)
    if unknown:
        raise PlaybookError(f"{where}: `wait`: unknown key(s): {', '.join(unknown)}")
    seconds = _limit_int(
        raw.get("seconds", DEFAULT_ASK_WAIT_SECONDS),
        f"{where}: `wait.seconds`",
        MIN_ASK_WAIT_SECONDS,
        MAX_ASK_WAIT_SECONDS,
    )
    rounds = _limit_int(
        raw.get("rounds", DEFAULT_ASK_ROUNDS),
        f"{where}: `wait.rounds`",
        1,
        MAX_ASK_ROUNDS,
    )
    return seconds, rounds


def _parse_recover(ctx: _Ctx, raw: Any, where: str, page: str) -> Recovery:
    """A page's `recover:` — one hand for any deviation (`tool:`/`macro:`
    at the top), or one per reading (`occluded:` the page itself under
    a sheet or popup, `elsewhere:` any other screen), with the page's
    own `limit:` under the walk-wide ceiling."""
    if not isinstance(raw, dict):
        raise PlaybookError(
            f"{where}: `recover` must be a mapping — `tool:` (one gesture) or `macro:`"
        )
    unknown = sorted(set(map(str, raw)) - _RECOVER_KEYS)
    if unknown:
        raise PlaybookError(f"{where}: `recover`: unknown key(s): {', '.join(unknown)}")
    limit = _limit_int(
        raw.get("limit", DEFAULT_RECOVER_LIMIT),
        f"{where}: `recover.limit`",
        1,
        MAX_RECOVER_ACTIONS,
    )
    keyed = [k for k in RECOVER_READINGS if k in raw]
    flat = [k for k in _HAND_KEYS if k in raw]
    if keyed and flat:
        raise PlaybookError(
            f"{where}: `recover` declares one hand (`tool:`/`macro:`) OR one "
            f"per reading ({', '.join(RECOVER_READINGS)}), not both"
        )
    if keyed:
        hands = {
            k: _parse_hand(ctx, raw[k], f"{where}: `recover.{k}`", page) for k in keyed
        }
        return Recovery(
            occluded=hands.get(READING_OCCLUDED),
            elsewhere=hands.get(READING_ELSEWHERE),
            limit=limit,
        )
    hand = _parse_hand(ctx, {k: raw[k] for k in flat}, f"{where}: `recover`", page)
    return Recovery(occluded=hand, elsewhere=hand, limit=limit)


def _parse_hand(ctx: _Ctx, raw: Any, where: str, page: str) -> RecoverHand:
    """One recovery hand — one gesture or one argument-less macro. A
    `tap` takes its target as `with: landmarks.<name>` (the declared
    spot, label-healed at run time); the other tools take no target."""
    if not isinstance(raw, dict):
        raise PlaybookError(f"{where} must be a mapping — `tool:` or `macro:`")
    unknown = sorted(set(map(str, raw)) - _HAND_KEYS)
    if unknown:
        raise PlaybookError(f"{where}: unknown key(s): {', '.join(unknown)}")
    tool, macro_raw = raw.get("tool"), raw.get("macro")
    if (tool is None) == (macro_raw is None):
        raise PlaybookError(f"{where} takes exactly one of `tool` or `macro`")
    if macro_raw is not None:
        if "with" in raw:
            raise PlaybookError(f"{where}: `with` goes with `tool: tap`")
        return RecoverHand(
            macro=_argless_macro(macro_raw, "recover", where, page, ctx.resolve)
        )
    if tool not in RECOVER_TOOLS:
        raise PlaybookError(
            f"{where}: `tool` must be one of {', '.join(RECOVER_TOOLS)} (got {tool!r})"
        )
    target = raw.get("with")
    if tool != "tap":
        if target is not None:
            raise PlaybookError(
                f"{where}: `with` goes with `tool: tap` — {tool} takes no target"
            )
        return RecoverHand(tool=tool)
    if target is None:
        raise PlaybookError(
            f"{where}: a recover tap needs `with: landmarks.<name>` — the "
            "declared spot it presses"
        )
    return RecoverHand(
        tool=tool, landmark=_landmark_name(ctx, target, f"{where}: `with`")
    )


# ---------- the money and resume lints ----------


def _screen_move(node: Node) -> bool:
    """A move whose enter page must read before it runs — a `do` with an
    enter, or an acting agent."""
    return (isinstance(node, DoNode) and bool(node.enter)) or (
        isinstance(node, AgentNode) and bool(node.tools)
    )


def _check_resume(nodes: list[Node]) -> None:
    """An ask leaves the phone on the IM thread; a screen move right
    after it needs the app back first. For a payment ask that is not
    advisory: consent is bound, money never recovers, so a missing
    `resume:` is a certain handover the moment the user says yes."""
    for i, n in enumerate(nodes[:-1]):
        nxt = nodes[i + 1]
        if isinstance(n, AskNode) and n.approve == "payment" and n.resume is None:
            if _screen_move(nxt):
                raise PlaybookError(
                    f"ask {n.id!r} (approve: payment) is followed by "
                    f"{nxt.id!r}, which runs on the app — declare `resume:` "
                    "on the ask to re-enter the app after the reply"
                )


def _check_money(nodes: list[Node]) -> None:
    """A payment move (a do OR an agent episode) runs only as the
    fall-through of an `ask` with `approve: payment`. Adjacency, not
    just reachability — the route is linear, so adjacency IS the only
    door: the conductor reads the payment sheet AT the ask (the message
    quotes its total) and fires the move right after, so a move in
    between would desynchronize the consent from the sheet. (The
    route's derived `enter` — always the waypoint before the ask — is
    what guarantees money fires on a verified app page, never on the IM
    thread the ask left the phone on.)"""
    for i, n in enumerate(nodes):
        if not (isinstance(n, (DoNode, AgentNode)) and n.irreversible):
            continue
        prev = nodes[i - 1] if i > 0 else None
        if not (isinstance(prev, AskNode) and prev.approve == "payment"):
            raise PlaybookError(
                f"move {n.id!r} (irreversible: payment) must DIRECTLY "
                "follow an `ask` with `approve: payment` — the ask "
                "reads the sheet its message quotes, and consent must "
                "not desynchronize from it"
            )
