"""The model-call vocabulary the parser and the walk share, code-owned.

A playbook never defines a prompt shape or an answer space; it names
steps, and every model call behind a step reads its vocabulary from
here — so the parser's allowlist and the runner's answer space can
never disagree. `ESCALATE` is every call's exit: the model hands the
walk over rather than guessing.
"""

ESCALATE = "escalate"

# The `agent` step's episode vocabulary. `AGENT_DONE` and `ESCALATE`
# are every episode's exits; each grantable tool adds its verbs. One
# map, so the parser's allowlist (`AGENT_TOOLS`), the runner's per-turn
# answer space, and the candidate builder's reserved words
# (`ACT_VERBS`) cannot drift.
AGENT_DONE = "done"
ACT_SCROLL_DOWN = "scroll_down"  # see content further down (the swipe goes up)
ACT_SCROLL_UP = "scroll_up"  # back toward the top (the swipe goes down)
ACT_BACK = "go_back"  # the OS back edge-swipe (an episode `back` tool)
AGENT_TOOL_VERBS: dict[str, tuple[str, ...]] = {
    "tap": (),
    "scroll": (ACT_SCROLL_DOWN, ACT_SCROLL_UP),
    "back": (ACT_BACK,),
}
AGENT_TOOLS = tuple(AGENT_TOOL_VERBS)
ACT_VERBS = tuple(v for verbs in AGENT_TOOL_VERBS.values() for v in verbs)
# How each granted tool reads in the episode's answer legend — `{verbs}`
# is the tool's verbs, quoted. Kept beside the verbs so renaming one can
# never make the legend lie.
AGENT_TOOL_LEGEND: dict[str, str] = {
    "tap": "a row of the NEWEST screen block, copied EXACTLY as quoted (it will be tapped)",
    "scroll": "{verbs}",
    "back": "{verbs}",
}
