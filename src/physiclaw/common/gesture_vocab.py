"""Gesture vocabulary — the shared tool-name contract between core and agent.

The engine's stuck guard, keyboard tracker, and layout lint classify
dispatched calls by tool name and unpack `sequence` steps; the
orchestrator executes those same steps. Both sides used to hold private
copies of these names held together by "keep in sync" comments — this
module is the single source. FastMCP registration in
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
# dismisses the keyboard (keyboard-tracker doctrine).
NAV_TOOLS = frozenset({"go_back", "home_screen", "force_quit"})

SWIPE = "swipe"
SEQUENCE = "sequence"

# `sequence` argument shape: {"actions": [{"tool_name": ..., "arg": ...}]}.
STEP_ACTIONS = "actions"
STEP_TOOL = "tool_name"
STEP_ARG = "arg"
