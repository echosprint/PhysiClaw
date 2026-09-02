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
from dataclasses import dataclass
from typing import Any

from physiclaw.conductor.calls import AGENT_TOOLS
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
    MAX_NODES,
    PACK_MACROS_DIRNAME,
    AgentNode,
    AskNode,
    DoNode,
    Node,
    Pack,
    PlaybookError,
    RecoverHand,
    TellNode,
    check_arg_refs,
    check_name,
    check_refs,
    field_name,
    opt_prose,
    prose,
    refs_in,
    require_str,
)
from physiclaw.macros.model import Macro, MacroError
from physiclaw.macros.parse import parse_inline_macro

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
ENTRY_KINDS = ("page", "start", "do", "agent", "ask", "tell")
_ENTRY_KEYS = {
    "page": {"page", "recover", *PAGE_DECL_FIELDS},
    "start": {"start", "macro"},
    "do": {"do", "with", "macro", "irreversible"},
    "agent": {"agent", "prompt", "tools", "give", "returns", "limit", "irreversible"},
    "ask": {"ask", "approve", "message", "over_budget_message", "resume"},
    "tell": {"tell", "message"},
}

# The shape `_macro_resolver` returns.
_MacroResolve = Callable[..., Macro]


@dataclass(frozen=True)
class CompiledRoute:
    """What `compile_route` hands back: the moves, the start page, each
    page's declared recovery hand, and the inline macro bodies under
    their synthesized names."""

    nodes: list[Node]
    start: str
    recovers: dict[str, RecoverHand]
    inline: dict[str, Macro]


def compile_route(
    raw: Any,
    *,
    playbook: str,
    input_names: set[str],
    pack: Pack,
    has_mandate: bool,
) -> CompiledRoute:
    """`route:` → the compiled route (see `CompiledRoute`)."""
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

    inline: dict[str, Macro] = {}
    resolve = _macro_resolver(playbook, pack, inline)
    moves: list[Node] = []
    seen: dict[str, int] = {}
    recovers: dict[str, RecoverHand] = {}
    # Grows as the single pass advances, so `{move.field}` refs are
    # defined-before-use by construction (route order).
    payloads: dict[str, tuple[str, ...]] = {}
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
        check_arg_refs(args, input_names, payloads, where)
        nxt = wp_ids[i + 1] if i + 1 < len(entries) else None
        if kind == "do":
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
            assert nxt is not None  # `start` sits immediately before a page
            spec = resolve(entry.get("macro"), where, name)
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
            agent = _parse_agent(
                where, name, entry, input_names, payloads, pack, current_page, nxt
            )
            payloads[agent.id] = agent.return_fields
            moves.append(agent)
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
        else:  # tell
            message, _ = _entry_message(where, entry, input_names, payloads)
            moves.append(TellNode(id=name, message=message))
    if len(moves) > MAX_NODES:
        raise PlaybookError(f"too many moves ({len(moves)} > {MAX_NODES})")
    _check_money(moves)
    return CompiledRoute(nodes=moves, start=start, recovers=recovers, inline=inline)


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


def _unique_list(raw: Any, where: str, check: Callable[[Any], str]) -> list[str]:
    """A list of distinct entries, each validated (and normalized) by
    `check` — the one shape `tools` and `give` share."""
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


def _limit_int(value: Any, where: str, lo: int, hi: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not lo <= value <= hi:
        raise PlaybookError(f"{where} must be {lo}–{hi} (got {value!r})")
    return value


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
    text = prose(entry.get("message"), f"{where}: `message`")
    refs = refs_in(text, f"{where}: `message`")
    check_refs(refs, input_names, payloads, f"{where}: `message`")
    return text, refs


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
    where: str,
    nid: str,
    entry: dict,
    args: dict,
    enter: str,
    verify: str,
    resolve: _MacroResolve,
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
    return DoNode(
        id=nid,
        macro=macro,
        args=dict(args),
        enter=enter,
        verify=verify,
        irreversible=_irreversible_class(entry, where),
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
            field_name(fname, f"{where}: return field")
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

    g_payloads = payloads
    if irreversible == "payment":
        # The consented total — the ONE gate slot a payment episode may
        # quote (its ask directly precedes it, `_check_money`).
        g_payloads = {**payloads, "ask": ("total",)}
    check_refs(
        refs_in(prompt, f"{where}: `prompt`"),
        input_names,
        g_payloads,
        f"{where}: `prompt`",
    )

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


def _parse_ask(
    where: str,
    nid: str,
    entry: dict,
    input_names: set[str],
    payloads: dict[str, tuple[str, ...]],
    resolve: _MacroResolve,
    *,
    has_mandate: bool,
) -> AskNode:
    approve = require_str(entry.get("approve"), f"{where}: `approve`")
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
    over = opt_prose(
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
            over_refs = refs_in(over, f"{where}: `over_budget_message`")
            check_refs(
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
    resume = None
    if entry.get("resume") is not None:
        resume = _argless_macro(entry["resume"], "resume", where, nid, resolve)
    return AskNode(
        id=nid, approve=approve, message=message, over_message=over, resume=resume
    )


def _parse_recover(
    raw: Any, where: str, page: str, pack: Pack, resolve: _MacroResolve
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


# ---------- the money lint ----------


def _check_money(nodes: list[Node]) -> None:
    """An irreversible-tagged move (a do OR an agent episode) runs only
    as the fall-through of an `ask` that approves its class. Adjacency,
    not just reachability — the route is linear, so adjacency IS the
    only door: the conductor reads the payment sheet AT the ask (the
    message quotes its total) and fires the move right after, so a move
    in between would desynchronize the consent from the sheet. (The
    route's derived `enter` — always the waypoint before the ask — is
    what guarantees money fires on a verified app page, never on the IM
    thread the ask left the phone on.)"""
    for i, n in enumerate(nodes):
        irr = n.irreversible if isinstance(n, (DoNode, AgentNode)) else None
        if not irr:
            continue
        prev = nodes[i - 1] if i > 0 else None
        if irr == "payment":
            if not (isinstance(prev, AskNode) and prev.approve == "payment"):
                raise PlaybookError(
                    f"move {n.id!r} (irreversible: payment) must DIRECTLY "
                    "follow an `ask` with `approve: payment` — the ask "
                    "reads the sheet its message quotes, and consent must "
                    "not desynchronize from it"
                )
        elif not isinstance(prev, AskNode):
            raise PlaybookError(
                f"move {n.id!r} (irreversible: {irr}) must "
                "DIRECTLY follow an `ask` — an irreversible act runs "
                "human-approved or not at all"
            )
