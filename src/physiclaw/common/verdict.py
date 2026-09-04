"""Screen-change verdict — the shared vocabulary between core and agent.

After each mutating gesture the orchestrator diffs before/after camera
frames and appends one of these markers to the gesture's result string
(`core.vision.change` computes the diff). The engine's stuck guard
(`agent.engine.stuck`) parses the marker back out of the tool result to
decide whether a repeated gesture is making progress.

Dependency-free on purpose: core composes, engine parses, and neither
should drag the other's imports (cv2 on one side, provider stack on the
other) into a string constant.
"""

# Appended to gesture results as ` | <marker>`. The agent's doctrine
# (PHYSICLAW § Unchanged screen) explains how to read them, and tests on
# both sides pin the exact bytes — reword only with a coordinated change.
SCREEN_CHANGED = "screen: changed"
SCREEN_UNCHANGED = "screen: no visible change"

_SEPARATOR = " | "


def attach(result: str, changed: bool | None) -> str:
    """Append the verdict marker to a gesture result string.

    `None` means the diff couldn't run (camera hiccup, missing frame) —
    the result passes through unmarked and the guard fails open.

    The input is `defang`ed first, so downstream a marker exists IFF this
    function wrote it.
    """
    result = defang(result)
    if changed is None:
        return result
    return f"{result}{_SEPARATOR}{SCREEN_CHANGED if changed else SCREEN_UNCHANGED}"


def defang(text: str) -> str:
    """Neutralize verdict-marker bytes already inside `text` (`:` → ` -`).

    The forgery defense, named on its own: composed action text normally
    can't contain a marker, but echoed content can — an error message
    repr-ing an agent argument is enough — and a pre-existing marker would
    forge the verdict the engine parses. `attach` defangs before writing
    its own marker; the macro runner defangs its retained view's texts so
    the one marker in a composed result is always the runner's.
    """
    for marker in (SCREEN_UNCHANGED, SCREEN_CHANGED):
        if marker in text:
            text = text.replace(marker, marker.replace(":", " -"))
    return text


def action_text(blocks: list[dict]) -> str:
    """The core-composed action text of a raw MCP tool result — the half
    carrying the verdict marker, and the ONLY safe haystack for `parse`:
    action text is the one text the phone cannot forge, so scanning the
    listing instead would let on-screen text fake a `screen: changed`.
    `screen_text` is the opposite half; the block-shape rule they share
    is documented there."""
    if not blocks or blocks[0].get("type") != "text":
        return ""  # a view reply composes no action text
    return blocks[0].get("text") or ""


def all_text(blocks: list[dict]) -> str:
    """Every text block of a raw MCP tool result, joined — action text
    and screen alike, the whole a tool result carries into the
    transcript (`conductor.views.text_of` reads the same off the DTO).
    The one spelling of the text-block walk `screen_text` and
    `action_text` split."""
    return "\n".join(b.get("text") or "" for b in blocks if b.get("type") == "text")


def screen_text(blocks: list[dict]) -> str:
    """The screen half of a raw MCP tool result — the OCR listing, joined.
    What guards and macro `when`/`skip_when` clauses match against: they WANT
    whatever the phone displays. Never pass this to `parse`.

    `action_text`'s complement, and both halves are security boundaries
    pointing opposite ways: the verdict haystack must contain ONLY what
    core composed (see `action_text`), while the screen haystack must
    EXCLUDE it — the action text echoes the caller's own arguments back,
    so a macro guard requiring the string a previous step just staged
    would otherwise match its own echo and gate nothing.

    The shape rule both apply: a gesture replies `[action text, image,
    listing]`, a view replies `[image, listing]` with no action text at
    all — so block 0 is action text iff it is a text block. Must run on
    the RAW blocks — provider-side fusing merges text blocks and erases
    the boundary."""
    if blocks and blocks[0].get("type") == "text":
        blocks = blocks[1:]  # drop the action text — it is not screen content
    return all_text(blocks)


def has_image(blocks: list[dict]) -> bool:
    """Whether a reply carries a screen image. The image half of the same
    block-shape rule the text helpers above own — kept here so callers
    never re-spell `b.get("type") == "image"`."""
    return any(b.get("type") == "image" for b in blocks)


def parse(result_text: str) -> bool | None:
    """Read a verdict marker back out of a gesture's action text.

    Returns True (changed) / False (no visible change) / None (no marker
    — non-gesture tool, camera hiccup, or an old-format result).

    Contract: pass ONLY core-composed action text (the first text block
    of a tool result), never screen-derived text — the OCR listing
    echoes whatever the phone displays, which can contain marker-like
    text and forge a verdict (`action_text` above extracts exactly that).
    Neither marker is a substring of the other, so one marker parses
    unambiguously in either check order; UNCHANGED first is the
    conservative tiebreak should both ever appear — a false "changed"
    is the harmful direction (it resets the stuck guard and re-hides
    silent refusals).
    """
    if SCREEN_UNCHANGED in result_text:
        return False
    if SCREEN_CHANGED in result_text:
        return True
    return None
