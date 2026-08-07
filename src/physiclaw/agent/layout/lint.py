"""Pre-dispatch layout lint — blocks the wrong-box paste flows."""

from physiclaw.agent.layout import store
from physiclaw.agent.layout.store import inside_learned
from physiclaw.common.bbox import center_of
from physiclaw.common.gesture_vocab import (
    SEQUENCE,
    STEP_ACTIONS,
    STEP_ARG,
    STEP_TOOL,
)


def _step_center(step, tool: str) -> tuple[float, float] | None:
    """Center of a sequence step's bbox if it's the given press tool."""
    if not isinstance(step, dict) or step.get(STEP_TOOL) != tool:
        return None
    return center_of(step.get(STEP_ARG) or [])


def _taps_box(steps, box) -> bool:
    """True if any step is a tap landing inside `box` (None/garbage → False)."""
    if not isinstance(box, list):
        return False
    for s in steps:
        c = _step_center(s, "tap")
        if c is not None and inside_learned(c, box):
            return True
    return False


def lint_gesture(name: str, arguments: dict, *, keyboard_up: bool) -> str | None:
    """Pre-dispatch layout lint — blocking message, or None.

    Covers both shapes of the wrong-box failure: a `sequence` (checked
    always — in-batch evidence plus the caller's keyboard belief) and a
    STANDALONE `long_press` (checked only when the keyboard is believed
    up: the raising tap happened in an earlier call, so no in-batch
    evidence can exist).
    """
    if name == SEQUENCE:
        return lint_sequence(arguments.get(STEP_ACTIONS), keyboard_up=keyboard_up)
    if name != "long_press" or not keyboard_up:
        return None
    d = store._load()
    hidden = d.get("chat_input_kb_hidden")
    visible = d.get("chat_input_kb_visible")
    if not (isinstance(hidden, list) and isinstance(visible, list)):
        return None
    c = center_of(arguments.get("bbox") or [])
    if c is None or not inside_learned(c, hidden):
        return None
    return (
        f"BLOCKED — not executed: this long-press targets the chat input's "
        f"KEYBOARD-HIDDEN box {hidden}, but the keyboard is up (your earlier "
        "press on the input raised it), so that region is now the keyboard "
        "itself and no Paste popover can appear there. Long-press the "
        f"chat input's KEYBOARD-VISIBLE box {visible} instead "
        "(SYSTEM § Screen layout)."
    )


def lint_sequence(actions, *, keyboard_up: bool = False) -> str | None:
    """Pre-dispatch layout lint for a `sequence` batch — blocking message,
    or None.

    The one signature it refuses: a `long_press` on the chat input's
    KEYBOARD-HIDDEN box at a point in the batch where the keyboard must
    be up — because an earlier step tapped that box (which raises the
    keyboard), because the caller already believes the keyboard is up
    (`keyboard_up`, raised in an earlier call), or because a later step
    taps the Paste box (which only ever appears above the RISEN input).
    With the keyboard up, the kb-hidden region IS the keyboard's bottom
    rows, so the popover never opens and the batch "succeeds" into a
    no-op the stuck guard can't see (every press changes the screen).
    The im template's correct box is `chat_input_kb_visible`.

    Fail-open by design: unlearned layout, malformed steps, or missing
    fields return None — this lint must never invent a blocker.
    """
    if not isinstance(actions, list):
        return None
    d = store._load()
    hidden = d.get("chat_input_kb_hidden")
    visible = d.get("chat_input_kb_visible")
    paste = d.get("chat_paste")
    if not (isinstance(hidden, list) and isinstance(visible, list)):
        return None

    input_tapped_at: int | None = None
    for i, step in enumerate(actions, 1):
        tap_c = _step_center(step, "tap")
        if tap_c is not None and inside_learned(tap_c, hidden):
            if input_tapped_at is None:
                input_tapped_at = i
            continue
        lp_c = _step_center(step, "long_press")
        if lp_c is None or not inside_learned(lp_c, hidden):
            continue
        pastes_after = _taps_box(actions[i:], paste)
        if input_tapped_at is None and not keyboard_up and not pastes_after:
            continue  # bare long-press near the bottom — not the paste flow
        if input_tapped_at is not None:
            why = f"step {input_tapped_at} taps that box, which raises the keyboard"
        elif keyboard_up:
            why = "the keyboard is already up (raised by an earlier press on the input)"
        else:
            why = "the batch then taps the Paste box, which only appears above the RISEN input"
        return (
            f"BLOCKED — no step ran: step {i} long-presses the chat input's "
            f"KEYBOARD-HIDDEN box {hidden}, but {why} — that region is now "
            "the keyboard, so no Paste popover can appear. Re-issue the "
            f"batch with the long-press on the KEYBOARD-VISIBLE box "
            f"{visible} (SYSTEM § Screen layout)."
        )

    # Reused-paste-box guard: the batch taps the learned chat Paste box but
    # long-presses a box OTHER than the chat input it belongs to. chat_paste
    # only exists above chat_input_kb_visible after long-pressing IT; copying
    # the `im` bundled sequence into another app (a search field in Meituan /
    # JD / …) long-presses THAT app's box, so the reused chat_paste coord lands
    # on empty screen and the paste silently no-ops — the batch still "ok"s and
    # the agent loops. The correct im template long-presses chat_input_kb_visible,
    # so it never trips this.
    if isinstance(paste, list) and _taps_box(actions, paste):
        for step in actions:
            c = _step_center(step, "long_press")
            if c is not None and not inside_learned(c, visible):
                return (
                    f"BLOCKED — no step ran (clipboard unchanged): reuses IM "
                    f"chat Paste box {paste} after long-pressing a different "
                    "field — its Paste popover is elsewhere. Redo every "
                    "step: long-press the field ALONE (own turn), read Paste "
                    "from that view (`search-in-app`). Don't reuse "
                    "chat_paste."
                )
    return None
