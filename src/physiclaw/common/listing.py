"""Element-listing grammar — the shared row shape between core and agent.

`core.vision.util.format_elements` composes the listing (header +
`format_row` per element); `agent.engine.compact._stub_body` parses rows
back out with the regexes when stubbing superseded views. Defining both
sides here means the shape exists once — a formatter change and its
parser change are the same edit.

Three doc surfaces quote `LISTING_HEADER` verbatim for a model to read:
the doctrine (`agent/context/PHYSICLAW.md` § Element listing), the
`peek` tool docstring (`core/server/tools.py`), and the claude-engine
doctrine (`agent/claude/CLAUDE.md`). Those copies stay literal because
each ships on a delivery path that renders no `{{token}}`s today
(`core.server.mcp` reads the doctrine raw; tool docstrings go out as-is)
— so `tests/common/test_listing.py` pins all three against this
constant instead.

Dependency-free on purpose: core composes, engine parses, and neither
should drag the other's imports (cv2 on one side, provider stack on the
other) into a string format.
"""

import re

# Column header, first line of every listing.
LISTING_HEADER = 'id [kind] "label" [left,top,right,bottom] conf'

# The closed element-kind vocabulary. `format_row` rejects anything
# else, so adding a kind forces this tuple (and a matching full-shape
# regex below) to grow in the same edit — the composer can never emit
# a row shape the parsers silently miss.
KINDS = ("icon", "text")


def format_row(id: int, kind: str, label: str, bbox, conf: float) -> str:
    """One element as a listing row: `id [kind] "label" [l,t,r,b] conf`.

    `bbox` is the 0-1 `[left, top, right, bottom]` list; coordinates
    render at 3 decimals, confidence at 2 — precision the row regexes
    below don't care about, but tests pin so agent-visible bytes stay
    stable.
    """
    if kind not in KINDS:
        raise ValueError(f"format_row: unknown kind {kind!r} — expected one of {KINDS}")
    coords = ",".join(f"{v:.3f}" for v in bbox)
    return f'{id} [{kind}] "{label}" [{coords}] {conf:.2f}'


# Prefix matcher — "is this line an element row?" (either kind). No
# production parser uses it (`_stub_body` needs the full-shape regexes
# for idempotency); it exists for test strategies that must generate
# non-row lines, and it derives from KINDS so it can't drift.
ROW_RE = re.compile(rf"^\d+ \[({'|'.join(KINDS)})\] ")

# Full-shape matchers, one per kind in KINDS. `TEXT_ROW_RE` captures the
# label (group 1); the greedy `.*` plus a `]`-free bbox class lets a
# label carry quotes and brackets (`He said "hi" [ok]`) yet still peel
# off the trailing bbox + confidence. `ICON_ROW_RE` requires the
# empty-label icon shape.
TEXT_ROW_RE = re.compile(r'^\d+ \[text\] "(.*)" \[[^\]]*\] [0-9.]+\s*$')
ICON_ROW_RE = re.compile(r'^\d+ \[icon\] "" \[[^\]]*\] [0-9.]+\s*$')
