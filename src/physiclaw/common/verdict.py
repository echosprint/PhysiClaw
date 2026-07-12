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

    Marker bytes already inside `result` are defanged first (`:` → ` -`),
    so downstream a marker exists IFF this function wrote it. Composed
    action text normally can't contain one, but echoed content can —
    an error message repr-ing an agent argument is enough — and a
    pre-existing marker would forge the verdict the engine parses.
    """
    for marker in (SCREEN_UNCHANGED, SCREEN_CHANGED):
        if marker in result:
            result = result.replace(marker, marker.replace(":", " -"))
    if changed is None:
        return result
    return f"{result}{_SEPARATOR}{SCREEN_CHANGED if changed else SCREEN_UNCHANGED}"


def parse(result_text: str) -> bool | None:
    """Read a verdict marker back out of a gesture's action text.

    Returns True (changed) / False (no visible change) / None (no marker
    — non-gesture tool, camera hiccup, or an old-format result).

    Contract: pass ONLY core-composed action text (the first text block
    of a tool result), never screen-derived text — the OCR listing
    echoes whatever the phone displays, which can contain marker-like
    text and forge a verdict (engine `_action_text` enforces this).
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
