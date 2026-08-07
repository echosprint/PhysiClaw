"""Cross-call belief about the on-screen keyboard, fed by gesture results."""

from dataclasses import dataclass

from physiclaw.agent.layout import store
from physiclaw.agent.layout.store import inside_learned
from physiclaw.common.bbox import center_of
from physiclaw.common.gesture_vocab import (
    NAV_TOOLS,
    PRESS_TOOLS,
    RUN_MACRO,
    SEQUENCE,
    STEP_ACTIONS,
    STEP_TOOL,
    SWIPE,
)

# Calls whose effect on the keyboard cannot be attributed: a swipe may or
# may not dismiss it, and `run_macro` hides a whole rehearsed sequence
# behind one result. Both decay the belief to "unknown" rather than guess.
KEYBOARD_OPAQUE = frozenset({SWIPE, RUN_MACRO})

# Boxes that are only ever pressed while typing/pasting — a press there
# neither raises nor dismisses the keyboard.
_KEYBOARD_REGION_FIELDS = (
    "chat_input_kb_visible",
    "send",
    "chat_paste",
    "backspace",
    "return",
    "space",
    "spotlight_input",
    "spotlight_paste",
)


@dataclass
class KeyboardTracker:
    """Conservative cross-call belief about the on-screen keyboard.

    "up" is claimed only when the camera verified the raising press (a
    changed-verdict press on the chat input's keyboard-hidden box) and
    every gesture since provably preserves it (typing/pasting boxes).
    Nav gestures mean "down" — except a camera-verified no-change nav
    (a missed edge swipe: the stuck guard's nav-miss case), which left
    the screen, keyboard included, exactly as it was. Swipes, batches
    with presses, and presses outside the keyboard region decay to
    "unknown", as does `run_macro` — the one local tool that replays real
    gestures, and so the exception to the rule below. Every OTHER local
    tool, plus views and clipboard syncs, never touches the screen.
    Consumers act only on "up" — "down"/"unknown" fail open.
    """

    state: str = "unknown"  # "up" | "down" | "unknown"

    def observe(self, name: str, arguments: dict, changed: bool | None) -> None:
        if name in NAV_TOOLS:
            if changed is not False:
                self.state = "down"
            return
        if name in KEYBOARD_OPAQUE:
            # RUN_MACRO is the SEQUENCE case wearing a local tool's name: a
            # macro replays up to 20 rehearsed gestures — `home_screen`
            # dismisses the keyboard, a chat-input tap raises it — behind ONE
            # result no one can attribute per step. Without this it falls to
            # the "views / local tools — screen untouched" return below and
            # carries a pre-macro belief across a macro that invalidated it,
            # so LayoutLint then blocks the agent's next long_press on the
            # box that is now correct. Decay, never keep.
            self.state = "unknown"
            return
        if name == SEQUENCE:
            actions = arguments.get(STEP_ACTIONS)
            steps = actions if isinstance(actions, list) else []
            if any(
                isinstance(x, dict) and x.get(STEP_TOOL) in (*PRESS_TOOLS, SWIPE)
                for x in steps
            ):
                # A batch verdict can't be attributed per step — any press
                # or swipe inside may have moved the keyboard.
                self.state = "unknown"
            return
        if name not in PRESS_TOOLS:
            return  # views / local tools / clipboard — screen untouched
        c = center_of(arguments.get("bbox") or [])
        if c is None:
            self.state = "unknown"
            return
        d = store._load()
        hidden = d.get("chat_input_kb_hidden")
        if isinstance(hidden, list) and inside_learned(c, hidden):
            # The raising press — but only the camera proves the keyboard
            # actually rose (a dead press must not claim "up").
            self.state = "up" if changed is True else "unknown"
            return
        for f in _KEYBOARD_REGION_FIELDS:
            box = d.get(f)
            if isinstance(box, list) and inside_learned(c, box):
                return  # typing/pasting — keyboard state preserved
        self.state = "unknown"  # a press elsewhere may have dismissed it
