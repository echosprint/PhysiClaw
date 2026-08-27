"""The buying-list ledger — desired state, and how the cart is read.

A `kind: list` playbook input's VALUE is a JSON array of {query, qty}
(fields = `calls.LEDGER_FIELDS`). `parse_ledger` is the one value gate:
activation and the CLI rehearsal call it strictly (raise / reject), the
walk calls it fail-open (a broken value degrades to a hand-over at the loop). The
walk's items then track pending → picked; the cart itself stays the
in-cart truth — RECONCILE re-reads it rather than trusting a flag.

The two screen readers are the reconciler's deterministic senses:
`assign_rows` maps picked items to cart rows (exclusive, exact-first),
and `row_qty` reads the quantity numeral between a row's flanking
stepper icons. Pure functions over a Screen — testable without a walk.
"""

import json
import re
from dataclasses import asdict, dataclass

from physiclaw.agent.conductor.calls import LEDGER_FIELDS
from physiclaw.agent.conductor.match import label_matches, normalize
from physiclaw.agent.conductor.playbook import Playbook, PlaybookError
from physiclaw.common.bbox import center_of
from physiclaw.common.listing import Element, Screen

# Items/qty cap what parse_task or the CLI may hand a walk.
MAX_LEDGER_ITEMS = 8
MAX_ITEM_QTY = 9

# Cart-row quantity: the numeral between the row's stepper icons,
# optionally prefixed by a multiplier mark ("2", "x2", "×2").
_QTY_RE = re.compile(r"^[x×✕]?\s*(\d{1,2})$")


@dataclass
class LedgerItem:
    """One buying-list entry — desired state. `status` walks
    pending → picked (the loop chose a row; `label` is what the cart
    should show) → the cart itself is the in-cart truth (RECONCILE
    re-reads it rather than trusting a flag). `qty` 0 = remove (a
    revision took it off the list; the reconciler steps it to zero)."""

    query: str
    qty: int
    status: str = "pending"  # "pending" | "picked"
    label: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "LedgerItem":
        return cls(
            query=str(data.get("query") or ""),
            qty=int(data.get("qty") or 0),
            status=("picked" if data.get("status") == "picked" else "pending"),
            label=(str(data["label"]) if data.get("label") else None),
        )


def parse_ledger(text: str, *, allow_zero: bool = False) -> list[LedgerItem]:
    """The `kind: list` input's value contract: a JSON array of
    {query, qty}. Raises PlaybookError naming the defect — arm fails
    early, activation and revision fall back (log + hand over/None).
    `allow_zero` is the revision form (qty 0 = remove)."""
    try:
        data = json.loads(text)
    except Exception as e:
        raise PlaybookError(f"ledger is not valid JSON: {e}") from e
    if not isinstance(data, list) or not data:
        raise PlaybookError("ledger must be a non-empty JSON array")
    if len(data) > MAX_LEDGER_ITEMS:
        raise PlaybookError(f"ledger has {len(data)} items > max {MAX_LEDGER_ITEMS}")
    floor = 0 if allow_zero else 1
    out: list[LedgerItem] = []
    seen: set[str] = set()
    for entry in data:
        if not isinstance(entry, dict) or set(entry) - set(LEDGER_FIELDS):
            raise PlaybookError("each ledger item must be a {query, qty} object")
        query = entry.get("query")
        if not isinstance(query, str) or not query.strip():
            raise PlaybookError("ledger item `query` must be a non-empty string")
        qty = entry.get("qty")
        if (
            isinstance(qty, bool)
            or not isinstance(qty, int)
            or not (floor <= qty <= MAX_ITEM_QTY)
        ):
            raise PlaybookError(
                f"ledger item `qty` must be {floor}–{MAX_ITEM_QTY} (got {qty!r})"
            )
        norm = normalize(query)
        if norm in seen:
            # Duplicates would collapse silently at the revision merge
            # and fight over one cart row at reconcile.
            raise PlaybookError(f"ledger item {query.strip()!r} appears twice")
        seen.add(norm)
        out.append(LedgerItem(query=query.strip(), qty=qty))
    return out


def check_ledger_value(spec: Playbook, values: dict[str, str]) -> None:
    """The resolved list input parses as a ledger — enforced at both seams
    a value enters through: the overture's activation, and the CLI
    rehearsal."""
    inp = spec.ledger_input
    if inp is not None:
        try:
            parse_ledger(values[inp.name])
        except PlaybookError as e:
            raise PlaybookError(f"input {inp.name!r}: {e}") from e


def assign_rows(screen: Screen, items: list[LedgerItem]) -> list[Element | None]:
    """Each PICKED item's cart row, assigned EXCLUSIVELY — exact
    normalized equality claims first, then the shared tiered fuzz
    (`match.label_matches`: the pick label came off one OCR reading and
    the cart row is another, possibly truncated) over the unclaimed
    rest. Exclusivity matters: with overlapping labels ("coke" /
    "coke zero") a first-match scan lets the shorter item claim the
    other's row and the reconciler taps the wrong stepper. Non-picked
    items get None (they are the loop's, not the reconciler's)."""
    rows = [r for r in screen.rows if r.kind == "text" and r.label.strip()]
    wants = [
        normalize(it.label or it.query) if it.status == "picked" else "" for it in items
    ]
    # Wanted items claim before qty-0 (removed) ones: after a revision
    # both can name near-identical products, and a removed item claiming
    # the kept item's row would step a wanted quantity to zero.
    order = sorted(range(len(items)), key=lambda i: (items[i].qty <= 0, i))
    out: list[Element | None] = [None] * len(items)
    claimed: set[int] = set()
    for i in order:  # pass 1: exact
        want = wants[i]
        if not want:
            continue
        for j, row in enumerate(rows):
            if j not in claimed and normalize(row.label) == want:
                out[i] = row
                claimed.add(j)
                break
    for i in order:  # pass 2: fuzzy over the unclaimed
        want = wants[i]
        if not want or out[i] is not None:
            continue
        for j, row in enumerate(rows):
            if j not in claimed and label_matches(want, normalize(row.label), ()):
                out[i] = row
                claimed.add(j)
                break
    return out


def row_qty(screen: Screen, row: Element) -> tuple[int, Element, Element] | None:
    """(qty, minus, plus) for a cart row, or None when unreadable.
    The steppers are the icons IMMEDIATELY flanking the quantity
    numeral — nearest on each side, never the band's extremes: a cart
    row carries other icons too (the selection checkbox sits far left),
    and a mis-attributed minus would TAP one. A row that reads
    otherwise hands over rather than guessing at a money-adjacent
    tap."""
    top, bot = row.bbox[1] - 0.01, row.bbox[3] + 0.01

    def in_band(el: Element) -> bool:
        c = center_of(el.bbox)
        return c is not None and top <= c[1] <= bot

    icons = sorted(
        (el for el in screen.rows if el.kind == "icon" and in_band(el)),
        key=lambda el: el.bbox[0],
    )
    for el in screen.rows:
        if el.kind != "text" or not in_band(el):
            continue
        m = _QTY_RE.match(el.label.strip())
        if not m:
            continue
        left = [ic for ic in icons if ic.bbox[2] <= el.bbox[0]]
        right = [ic for ic in icons if ic.bbox[0] >= el.bbox[2]]
        if not left or not right:
            continue
        return int(m.group(1)), left[-1], right[0]
    return None


def describe(items: "list[LedgerItem] | None") -> str:
    """The list, terse, for handover reasons and journal lines."""
    return ", ".join(f"{it.query}×{it.qty}({it.status})" for it in items or [])
