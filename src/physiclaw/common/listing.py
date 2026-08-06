"""Element-listing grammar — the shared element shape between core and agent.

One canonical shape, `Element`, with both codecs beside it: text via
`format_elements` / `decode_elements` (the listing an agent reads), JSON
via `Element.to_dict` / `from_dict` (`elements_to_json`, snapshot files).
Core composes through `format_elements` (`orchestration.perception`);
`agent.engine.compact._stub_body` parses rows back out with the regexes
when stubbing superseded views. Defining every side here means the shape
exists once — a formatter change and its parser change are the same
edit.

Three doc surfaces quote `LISTING_HEADER` verbatim for a model to read:
the doctrine (`agent/context/PHYSICLAW.md` § Element listing), the
`peek` tool docstring (`core/server/tools.py`), and the claude-engine
doctrine (`agent/claude/CLAUDE.md`). Those copies stay literal rather
than becoming a `{{token}}`: only the doctrine path renders tokens
(`common.doctrine`), while tool docstrings and CLAUDE.md ship as-is —
so `tests/common/test_listing.py` pins all three against this constant
instead.

Dependency-free on purpose: core composes, engine parses, and neither
should drag the other's imports (cv2 on one side, provider stack on the
other) into a string format.
"""

import math
import re
from collections.abc import Iterable
from dataclasses import dataclass

from physiclaw.common.bbox import Bbox

# Column header, first line of every listing.
LISTING_HEADER = 'id [kind] "label" [left,top,right,bottom] conf'

# The closed element-kind vocabulary. `Element` rejects anything else,
# so adding a kind forces this tuple (and a matching full-shape regex
# below) to grow in the same edit — the composer can never emit a row
# shape the parsers silently miss.
KINDS = ("icon", "text")


def format_row(id: int, kind: str, label: str, bbox, conf: float) -> str:
    """One element as a listing row: `id [kind] "label" [l,t,r,b] conf`.

    `bbox` is the 0-1 `[left, top, right, bottom]` list; coordinates
    render at 3 decimals, confidence at 2 — agent-visible bytes tests
    pin. Convenience over `Element(...).row()`, so loose arguments pass
    the same validation as every other composer path — one home for the
    grammar's rules, no back door.
    """
    return Element(id=id, kind=kind, label=label, bbox=tuple(bbox), conf=conf).row()


# Prefix matcher — "is this line an element row?" (either kind).
# `Screen.read` gates on it so a row-SHAPED line that fails `parse_row`
# is dropped rather than kept as matchable prose; test strategies use it
# to generate non-row lines. Derives from KINDS so it can't drift.
ROW_RE = re.compile(rf"^\d+ \[({'|'.join(KINDS)})\] ")

# Full-shape matchers, one per kind in KINDS. `TEXT_ROW_RE` captures the
# label (group 1); the greedy `.*` plus a `]`-free bbox class lets a
# label carry quotes and brackets (`He said "hi" [ok]`) yet still peel
# off the trailing bbox + confidence. `ICON_ROW_RE` requires the
# empty-label icon shape.
TEXT_ROW_RE = re.compile(r'^\d+ \[text\] "(.*)" \[[^\]]*\] [0-9.]+\s*$')
ICON_ROW_RE = re.compile(r'^\d+ \[icon\] "" \[[^\]]*\] [0-9.]+\s*$')

# All five fields, for `parse_row` — the one row parser production code
# uses (region-scoped macro guards read rows through it via
# `Screen.read`). Derives from KINDS like the matchers above, so a new
# kind can't be silently unparseable.
_DECODE_ROW_RE = re.compile(
    rf'^(\d+) \[({"|".join(KINDS)})\] "(.*)" \[([^\]]*)\] ([0-9.]+)\s*$'
)


@dataclass(frozen=True)
class Element:
    """One detected screen element (icon detection or OCR) in its
    canonical shape — the same shape on every surface: `to_dict` /
    `from_dict` are the JSON side, `row` / `parse_row` the text side.

    Canonical by construction: bbox rounds to 3 decimals and conf to 2,
    exactly the precision the text side renders, so text ↔ Element ↔
    dict conversions are identities rather than approximations — the
    round-trip is testable as one. The constraints the row grammar
    implies are enforced here instead of trusted: `kind` is closed over
    KINDS, a label is single-line (rows are lines), and an icon's label
    is empty — a labelled icon row matches neither full-shape regex, so
    it would silently survive `compact._stub_body` as prose."""

    id: int
    kind: str
    label: str
    bbox: Bbox
    conf: float

    def __post_init__(self) -> None:
        # The label constraints below close the character-level grammar;
        # these two close the NUMERIC side: a negative id fails the row
        # regexes' `^\d+`, and a negative or non-finite conf fails
        # `[0-9.]+` — either would compose a row no parser matches.
        if not isinstance(self.id, int) or isinstance(self.id, bool) or self.id < 0:
            raise ValueError(
                f"Element: id must be a non-negative integer (got {self.id!r}) — "
                "the row grammar renders it as bare digits"
            )
        if self.kind not in KINDS:
            raise ValueError(
                f"Element: unknown kind {self.kind!r} — expected one of {KINDS}"
            )
        # `splitlines`-clean, not merely \n-free: a listing is consumed
        # line-wise, and splitlines also breaks on \x0b/\x0c/\x1c-\x1e/
        # \x85/\u2028/\u2029 — any of them would tear the row in two.
        if "".join(self.label.splitlines()) != self.label:
            raise ValueError(
                f"Element: label must be single-line (rows are lines): {self.label!r}"
            )
        if self.kind == "icon" and self.label:
            raise ValueError(
                f"Element: an icon's label must be empty (got {self.label!r}) — "
                'labelled elements are kind "text"'
            )
        if len(self.bbox) != 4:
            raise ValueError(
                "Element: bbox must be [left, top, right, bottom] "
                f"(got {list(self.bbox)!r})"
            )
        bbox = tuple(round(float(v), 3) for v in self.bbox)
        conf = round(float(self.conf), 2)
        if not all(math.isfinite(v) for v in bbox):
            raise ValueError(
                f"Element: bbox coords must be finite (got {list(self.bbox)!r})"
            )
        if not (math.isfinite(conf) and conf >= 0):
            raise ValueError(
                f"Element: conf must be finite and ≥ 0 (got {self.conf!r})"
            )
        # Frozen, so canonicalization writes through object.__setattr__.
        object.__setattr__(self, "bbox", bbox)
        object.__setattr__(self, "conf", conf)

    @classmethod
    def from_dict(cls, d: dict) -> "Element":
        """The `elements_to_json` shape in. `label` may be absent or None
        (icons) — both read as empty, as the composer always has."""
        return cls(
            id=d["id"],
            kind=d["kind"],
            label=d.get("label") or "",
            bbox=tuple(d["bbox"]),
            conf=d["conf"],
        )

    def to_dict(self) -> dict:
        """The JSON shape out — `UIElement.to_dict` delegates here, so
        the producer's dicts can't drift from this one."""
        return {
            "id": self.id,
            "kind": self.kind,
            "label": self.label,
            "bbox": list(self.bbox),
            "conf": self.conf,
        }

    def row(self) -> str:
        """This element as a listing row — the fields are validated and
        rounded already, so this is pure rendering."""
        coords = ",".join(f"{v:.3f}" for v in self.bbox)
        return f'{self.id} [{self.kind}] "{self.label}" [{coords}] {self.conf:.2f}'


def parse_row(line: str) -> Element | None:
    """One listing line → Element, or None when the line is not a
    well-formed row (the header, prose, a malformed bbox, a labelled
    icon). Lenient on purpose: consumers read screen text that
    legitimately mixes rows with non-listing lines, and only they know
    what the rest means. `decode_elements` is the strict form."""
    m = _DECODE_ROW_RE.match(line)
    if m is None:
        return None
    try:
        # Unpacking raises ValueError on a wrong coord count, exactly
        # like a non-numeric coord or an `Element` constraint would.
        left, top, right, bottom = (float(v) for v in m.group(4).split(","))
        return Element(
            id=int(m.group(1)),
            kind=m.group(2),
            label=m.group(3),
            bbox=(left, top, right, bottom),
            conf=float(m.group(5)),
        )
    except ValueError:
        return None


def decode_elements(text: str) -> list[Element]:
    """A whole listing back to elements — the strict inverse of
    `format_elements`. The first line must be the header and every
    following non-blank line must parse as a row; anything else raises
    ValueError naming the line. For screen text that mixes rows with
    prose, use `parse_row` per line instead."""
    lines = text.splitlines()
    if not lines or lines[0].rstrip() != LISTING_HEADER:
        raise ValueError(
            f"decode_elements: first line must be the listing header "
            f"{LISTING_HEADER!r}"
        )
    out: list[Element] = []
    for n, line in enumerate(lines[1:], start=2):
        if not line.strip():
            continue  # a trailing blank line is not a grammar violation
        el = parse_row(line)
        if el is None:
            raise ValueError(
                f"decode_elements: line {n} is not an element row: {line!r}"
            )
        out.append(el)
    return out


def format_elements(items: Iterable[Element]) -> str:
    """Human/agent-friendly element list — the header plus one row per
    element. Lives beside its inverse (`decode_elements`) so the two
    change in the same edit."""
    return "\n".join([LISTING_HEADER, *(e.row() for e in items)])
