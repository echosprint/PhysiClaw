"""Every bound a walk runs under, in one place.

The conductor promises that everything it does is bounded: a route has
at most so many moves, an episode so many calls, a page so many
recoveries, a boot so many unlocks. Those numbers used to sit beside
the code that enforced each one; here they sit together so a reader
can see, in one screen, what stops a walk from running forever. The
modules that enforce them import from here.
"""

# ---- the route (playbook.py / route.py) ----

# Counts MOVES (compiled nodes) — waypoints ride free, bounded by the
# pack's own MAX_PAGES. (Inputs are capped by the macro grammar's
# MAX_INPUTS — the same `inputs:` section.)
MAX_NODES = 20

# Recovery: the walk-wide ceiling on recovery actions (what stops a
# splash ad on every cold launch from relaunching forever), and a
# page's own default `limit:` under it.
MAX_RECOVER_ACTIONS = 6
DEFAULT_RECOVER_LIMIT = 2

# An ask's patience (`wait:`): the in-session reply poll cadence (the
# engine's `wait` tool caps a single sleep at 60 s) and how many silent
# rounds before the session suspends for the next wake.
DEFAULT_ASK_WAIT_SECONDS = 45
DEFAULT_ASK_ROUNDS = 3
MIN_ASK_WAIT_SECONDS = 5
MAX_ASK_WAIT_SECONDS = 60
MAX_ASK_ROUNDS = 10

# An agent step: `limit.calls` bounds the LLM calls an episode may spend
# (each action is one call), `limit.scrolls` the scrolling subset — the
# classic degenerate loop gets its own tighter cap.
MAX_AGENT_CALLS = 30
DEFAULT_AGENT_CALLS = 12
DEFAULT_AGENT_SCROLLS = 6
# The prompt is the author's whole brief; the cap only guards against a
# pasted document (a real brief runs a few thousand characters).
MAX_PROMPT_LEN = 8000
MAX_RETURNS = 6

# ---- the pack (pages.py) ----

MAX_PAGES = 30
MAX_ANCHORS = 12
MAX_FORBID = 8
MAX_ANCHOR_LEN = 80
MAX_LANDMARKS = 12

# ---- model calls (micro.py) ----

# Prompt-size guard: a listing rarely exceeds ~40 rows; more candidates
# than this add tokens without adding real choices.
MAX_CANDIDATES = 40

# ---- the boot (overture.py) — fixed, not authorable ----

# `unlock_phone` races the passcode keypad and its own doctrine says to
# retry once or twice; the open macro is deterministic, so a second miss
# means the world is not what the pack describes. Together these ARE
# the boot's turn budget: one opening peek, then only unlocks, opens,
# and history scrolls — never more than 1 + 2 + 2 + 2 turns plus the
# one quit brief.
UNLOCK_TRIES = 2
OPEN_TRIES = 2
# How many times parse_task's `scroll_up` escape may scroll the thread
# for older messages before the cautious read (no full request in view
# → no activation) stands.
HISTORY_SCROLLS = 2

# ---- the drivers (rehearsal.py / replay.py) ----

# A rehearsal is a person watching, so the bound is "long enough for a
# real walk" rather than the engine's session budget. A gate polling
# for a reply is the long pole.
REHEARSE_MAX_TURNS = 60
# A replay's hard stop, whatever the screens say: a walk that keeps
# recovering must not spin past what a person would read.
MAX_REPLAY_TURNS = 200
