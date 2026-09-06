"""The lints — every rule `playbooks check` reports beyond the grammar.

Two kinds. The whole-route checks the compiler runs after compiling
the moves (a violation is a `PlaybookError`, the playbook is refused):
money (`check_money` — a payment move directly follows the ask that
approves it), resume (`check_resume` — a payment ask followed by an
app move declares `resume:`), and the boot's shape (`check_boot`).
And the advisories (`readiness_warnings`, `menu_warnings`): things
that let a walk start and then quietly under-perform — legal, and the
author is told the cost rather than refused.
"""

from physiclaw.common.bbox import center_of
from physiclaw.common.listing import Element, Screen, format_elements
from physiclaw.common.paths import PACK_PROMPTS_DIRNAME, PROMPT_SUFFIX
from physiclaw.conductor.spec import reply
from physiclaw.conductor.spec.conventions import BOOT_PLAYBOOK, CHANNEL_APP
from physiclaw.conductor.spec.match import SHORT_ANCHOR_MIN, normalize, score_page
from physiclaw.conductor.spec.model import (
    ActivateNode,
    AgentNode,
    AskNode,
    DoNode,
    Node,
    Pack,
    Playbook,
    PlaybookEntry,
    PlaybookError,
    TellNode,
)
from physiclaw.conductor.spec.pages import PageDecl, PagePrint

# ---------- the route's whole-shape checks ----------


def screen_move(node: Node) -> bool:
    """A move whose enter page must read before it runs — a `do` with an
    enter, or an acting agent."""
    return (isinstance(node, DoNode) and bool(node.enter)) or (
        isinstance(node, AgentNode) and bool(node.tools)
    )


def check_resume(nodes: list[Node]) -> None:
    """An ask leaves the phone on the IM thread; a screen move right
    after it needs the app back first. For a payment ask that is not
    advisory: consent is bound, money never recovers, so a missing
    `resume:` is a certain handover the moment the user says yes."""
    for i, n in enumerate(nodes[:-1]):
        nxt = nodes[i + 1]
        if isinstance(n, AskNode) and n.approve == "payment" and n.resume is None:
            if screen_move(nxt):
                raise PlaybookError(
                    f"ask {n.id!r} (approve: payment) is followed by "
                    f"{nxt.id!r}, which runs on the app — declare `resume:` "
                    "on the ask to re-enter the app after the reply"
                )


def check_money(nodes: list[Node]) -> None:
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


def check_boot(nodes: list[Node]) -> None:
    """The boot's shape: it ends in exactly one `select` (the baton
    IS its ending), and it never speaks to the user — before the request
    is even read, an ask would hold for a reply to nothing and a tell
    would report on nothing. Everything else on it is the ordinary route
    grammar."""
    activates = [n.id for n in nodes if isinstance(n, ActivateNode)]
    if not nodes or not isinstance(nodes[-1], ActivateNode):
        raise PlaybookError(
            f"{CHANNEL_APP}/{BOOT_PLAYBOOK} must end with a `select` step — "
            "reading the thread is what the boot is for, and its answer is the "
            "next walk"
        )
    if len(activates) > 1:
        raise PlaybookError(
            f"{CHANNEL_APP}/{BOOT_PLAYBOOK}: one `select` only — "
            f"{', '.join(activates)} would each end the boot"
        )
    for n in nodes:
        if isinstance(n, (AskNode, TellNode)):
            raise PlaybookError(
                f"{CHANNEL_APP}/{BOOT_PLAYBOOK}: {n.id!r} messages the user — the "
                "boot only reads the thread; asks and tells belong to playbooks"
            )


# ---------- the advisories ----------


def pack_warnings(pack: Pack, entries: "list[PlaybookEntry]") -> list[str]:
    """The pack-level advisories `playbooks check` prints, pack-relative:
    two pages one screen could read as both (one page's anchor texts a
    subset of another's — every screen that reads the larger also reads
    the smaller, and the matcher answers `ambiguous`), and prompt files
    no route names — a rename that left the old file behind, or prose
    written before its step."""
    out = _ambiguous_pages(pack)
    used_by = {e.name: e.spec.prompts_used for e in entries if e.spec is not None}
    shared_used: set[str] = set().union(*used_by.values())
    out += [
        f"{PACK_PROMPTS_DIRNAME}/{n}{PROMPT_SUFFIX} is read by no playbook"
        for n in sorted(set(pack.prompts.ok) - shared_used)
    ]
    for pb_name, files in sorted(pack.local.items()):
        for n in sorted(set(files.prompts.ok) - used_by.get(pb_name, frozenset())):
            out.append(
                f"{pb_name}/{PACK_PROMPTS_DIRNAME}/{n}{PROMPT_SUFFIX} is read by "
                f"no step of {pb_name}"
            )
    return out


def _ambiguous_pages(pack: Pack) -> list[str]:
    """Pages one screen would read as both: a screen showing exactly what
    one page declares also reads another whole. Judged by the matcher
    itself (`score_page` over a synthetic screen of the larger page's
    anchors, each at its band), so `within`, alternates, the fuzzy tiers,
    and `forbid` all decide — never a second spelling of the rule."""
    out = []
    pages = sorted(pack.pages.items())
    for large, l_decl in pages:
        screen = _declared_screen(l_decl)
        for small, s_decl in pages:
            if small == large:
                continue
            if score_page(PagePrint(app=pack.app, decl=s_decl), screen).passes:
                out.append(
                    f"pages {small!r} and {large!r}: a screen showing {large!r} "
                    f"whole also reads {small!r} — add an anchor or a `forbid` "
                    f"to {small!r}"
                )
    return out


def _declared_screen(decl: PageDecl) -> Screen:
    """The screen a page's declaration describes: one row per anchor,
    its canonical text, centred in its band (or mid-screen unpinned)."""
    els = []
    for i, a in enumerate(decl.anchors):
        cx, cy = (center_of(a.within) if a.within is not None else None) or (0.5, 0.5)
        els.append(
            Element(
                id=i,
                kind="text",
                label=a.text,
                bbox=(cx - 0.05, cy - 0.01, cx + 0.05, cy + 0.01),
                conf=0.9,
            )
        )
    return Screen.read(format_elements(els))


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
    that runs on the app: the send left the phone on the IM thread, so
    that move's enter check runs against the wrong screen and spends
    the page's recover hand, or hands over with none. Legal; the author
    is told. (A payment ask in that shape is refused at parse — consent
    never recovers.)"""
    out = []
    nodes = spec.nodes
    for i, n in enumerate(nodes[:-1]):
        nxt = nodes[i + 1]
        if not screen_move(nxt):
            continue
        if isinstance(n, AskNode) and n.resume is None:
            out.append(
                f"ask {n.id!r} has no `resume:` but {nxt.id!r} runs on the app "
                "next — the enter check will read the IM thread; declare "
                "`resume:` or expect the page's recover hand to spend"
            )
        elif isinstance(n, TellNode):
            out.append(
                f"tell {n.id!r} leaves the phone on the IM thread, and {nxt.id!r} "
                "runs on the app next — its enter check will read the thread; "
                "expect the page's recover hand to spend"
            )
    return out


def _anchor_warnings(spec: Playbook, pack: Pack) -> list[str]:
    """A page the route checks that identifies itself by a weak anchor: a
    reading under `match.SHORT_ANCHOR_MIN`, pinned to no band. The
    matcher reads such an anchor exactly (no one-substitution window at
    that length), so one OCR slip reads the page unknown and the walk
    spends its recover hand for nothing — and unpinned, it also matches
    the same characters anywhere on any screen. Pin it (`within:`), or
    choose a longer text."""
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
        if decl is None:
            continue
        weak = [
            a.text
            for a in decl.anchors
            if a.within is None
            and any(len(normalize(r)) < SHORT_ANCHOR_MIN for r in a.readings)
        ]
        if weak:
            out.append(
                f"page {name!r} anchor(s) {', '.join(weak)} are under "
                f"{SHORT_ANCHOR_MIN} characters and pinned to no band — read "
                "exactly, matched anywhere; pin them with `within:` or choose "
                "a longer text"
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
