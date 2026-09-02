"""Money runs in code — the payment doctrine's predicates, one home.

Two pure functions, deliberately free of walk state so the money
rules can be read (and audited) without the state machine around them.
The walk supplies the numbers and acts on the answers:

  - `declared_total` — the amount beside the label the ask's `total:`
    names: the number the user is quoted and consents to.
  - `amounts` — every amount the screen shows; the ask records this
    set as what the user SAW, the fire-time check reads the sheet the
    same way.
  - `fire_block` — the two fire-time predicates, run against the
    CURRENT screen after the human's consent: staleness (the sheet
    must still show the consented total) and the bound (no amount
    above the total may have APPEARED since the ask — what the user
    saw is the limit; a struck-through original price they saw is
    not a change). None = pay; else the bare reason — the walk
    prefixes the move it was guarding and hands over.

Consent itself — quoting, binding, consuming — stays with the gate
(`step_ask.py`, `gate.Gate`): consent is a conversation, these are
arithmetic.
"""

import math

from physiclaw.common.bbox import center_of
from physiclaw.common.listing import Screen, label_hit
from physiclaw.conductor.match import PRICE_RE

# How far from the total's label row its amount may sit when OCR splits
# them into two rows (a footer's "合计" and its "¥59.9" side by side).
TOTAL_RADIUS = 0.15


def amount(text: str) -> float:
    """One `PRICE_RE` group as a number — thousands separators stripped."""
    return float(text.replace(",", ""))


def amounts(screen: Screen) -> list[float]:
    """Every ¥/￥ amount visible on the screen — `match.PRICE_RE`, the
    one spelling of a currency amount, run over raw row labels; never a
    model."""
    out: list[float] = []
    for row in screen.rows:
        out.extend(amount(m) for m in PRICE_RE.findall(row.label))
    return out


def _is_label_row(label: str, readings: tuple[str, ...]) -> tuple[bool, bool]:
    """(hits, exact): whether a row reads as the total label at all, and
    whether it is that label and nothing else — "合计" or "合计: ¥59.9",
    never "商品合计 ¥79" (a subtotal wearing the same characters)."""
    if not any(label_hit(r, label) for r in readings):
        return False, False
    bare = PRICE_RE.sub("", label)
    bare = "".join(ch for ch in bare if ch.isalnum())
    return True, any("".join(ch for ch in r if ch.isalnum()) == bare for r in readings)


def declared_total(screen: Screen, readings: tuple[str, ...]) -> float | None:
    """The amount beside the declared total label: on the label's own
    row when that row carries one, else the nearest price-bearing row
    within `TOTAL_RADIUS`. A row that is the label and nothing else
    beats a row that merely contains it (the payable 合计 over a 商品合计
    subtotal), and among equals the LOWEST on screen wins — the payable
    total is the footer. None when no such row reads."""
    priced = [
        (pc, amount(m[0]))
        for r in screen.rows
        if (m := PRICE_RE.findall(r.label)) and (pc := center_of(r.bbox)) is not None
    ]
    ranked = []
    for row in screen.rows:
        if row.kind != "text":
            continue
        hits, exact = _is_label_row(row.label, readings)
        if not hits:
            continue
        c = center_of(row.bbox)
        assert c is not None  # Element bboxes are valid by construction
        ranked.append((not exact, -c[1], row, c))
    for _, _, row, c in sorted(ranked, key=lambda t: (t[0], t[1])):
        own = PRICE_RE.findall(row.label)
        if own:
            return amount(own[0])
        nearest = min(((math.dist(c, pc), amt) for pc, amt in priced), default=None)
        if nearest is not None and nearest[0] <= TOTAL_RADIUS:
            return nearest[1]
    return None


def fire_block(
    *, consented: float | None, seen: tuple[float, ...], screen: Screen
) -> str | None:
    """The two fire-time predicates. `seen` is every amount on the sheet
    when the ask quoted it. None = pay; else the bare reason to block —
    the walk prefixes the move it was guarding."""
    if consented is None:
        return "reached without a confirmed total"
    amts = amounts(screen)
    if not any(abs(a - consented) < 0.01 for a in amts):
        return (
            f"sheet changed after consent: confirmed ¥{consented:g}, "
            f"now sees {amts or 'no amounts'}"
        )
    over = [
        a
        for a in amts
        if a > consented + 0.005 and not any(abs(a - s) < 0.01 for s in seen)
    ]
    if over:
        return (
            f"amount(s) {over} above the consented total ¥{consented:g} "
            "appeared after the ask"
        )
    return None
