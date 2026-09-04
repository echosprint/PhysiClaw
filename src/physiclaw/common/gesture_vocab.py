"""Gesture vocabulary — the shared tool-name contract between core and agent.

The engine's stuck guard, keyboard tracker, and layout lint classify
dispatched calls by tool name and unpack `sequence` steps; the
orchestrator executes those same steps. Both sides used to hold private
copies of these names held together by "keep in sync" comments — this
module is the single source. It also owns the non-gesture tool names the
agent layer references (PEEK, SEND_TO_CLIPBOARD, UNLOCK_PHONE — macro whitelists,
recovery views, and the conductor's boot), so those renames fail the same pin, and RUN_MACRO: the
one LOCAL (non-MCP) tool the classifiers must name, deliberately outside
that pin, with a test asserting the exclusion. MCPServer registration in
`core/server/tools.py` deliberately does NOT consume it: names there
are function identifiers, and `tests/common/test_gesture_vocab.py`
pins registration ⊇ vocabulary instead, so a rename fails red in CI
rather than eroding silently.

Dependency-free on purpose (the `verdict.py` precedent): string
constants only, importable from either side of the MCP boundary.
"""

# Press-family gestures — one target bbox each. Switching press type on
# a dead element is not "changing method" (stuck-guard doctrine).
PRESS_TOOLS = frozenset({"tap", "double_tap", "long_press"})

# Whole-screen navigation gestures — no target argument; each provably
# dismisses the keyboard (keyboard-tracker doctrine). GO_BACK is named
# because the conductor's rescue ladder synthesizes it directly (the
# UNLOCK_PHONE precedent); the others stay set-only until someone does.
GO_BACK = "go_back"
FORCE_QUIT = "force_quit"
NAV_TOOLS = frozenset({GO_BACK, "home_screen", FORCE_QUIT})

SWIPE = "swipe"
SWIPE_DIRECTIONS = ("up", "down", "left", "right")
SEQUENCE = "sequence"

# Non-gesture tools named beyond the server — the perception peek
# (macro recovery views), the phone's own screenshot (the studio's
# published surface), and the clipboard bridge (macro steps). Owned
# here so macros' whitelist and the registration ⊇ vocabulary pin cover
# them like any gesture name.
PEEK = "peek"
SCREENSHOT = "screenshot"
SEND_TO_CLIPBOARD = "send_to_clipboard"

# The swipe ladder: stroke length per `size` as a fraction of the screen,
# and the speed names. Dependency-free here so the studio page can be
# served the same ladder the orchestrator drives (`gestures.py` types
# them and consumes them).
SWIPE_DISTANCES: dict[str, float] = {
    "s": 0.1,
    "m": 0.3,
    "l": 0.5,
    "xl": 0.75,
    "xxl": 0.90,
}
SWIPE_SPEEDS = ("slow", "medium", "fast")
# Security-sensitive and deliberately NOT macro-able (see
# `macros.model.ALLOWED_STEP_TOOLS`), but the agent layer names it in
# two places — the engine's screen-tool classifier and the conductor's boot,
# which synthesizes the call directly. Here so a rename fails the
# registration pin instead of silently unclassifying it.
UNLOCK_PHONE = "unlock_phone"

# The one LOCAL engine tool the classifiers must name. NOT an MCP tool, so
# it is deliberately outside the registration ⊇ vocabulary pin — the test
# asserts that exclusion, so the distinction is stated rather than assumed.
# It belongs here anyway: a macro replays real gestures, so the classifiers
# that reason about gesture names have to reason about it too. (The macro
# settle step `wait` deliberately does NOT live here: nothing outside
# `physiclaw.macros` names it, and the engine has an unrelated local tool that
# also answers to "wait" — see `macros.model.WAIT`.)
RUN_MACRO = "run_macro"

# `sequence` argument shape: {"actions": [{"tool_name": ..., "arg": ...}]}.
STEP_ACTIONS = "actions"
STEP_TOOL = "tool_name"
STEP_ARG = "arg"
