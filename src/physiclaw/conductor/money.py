"""Money runs in code — the payment doctrine's predicates, one home.

Two pure functions, deliberately free of walk state so the money
rules can be read (and audited) without the state machine around them.
The walk supplies the numbers and acts on the answers:

  - `declared_total` — the amount beside the label the ask's `total:`
    names: the number the user is quoted and consents to.
  - `amounts` — every amount the screen shows; the fire-time check
    reads the sheet this way.
  - `fire_block` — the two fire-time predicates, run against the
    CURRENT screen after the human's consent: staleness (the sheet
    must still show the consented total) and the bound (nothing on it
    may exceed that total — what the user saw IS the limit). None =
    pay; else the bare reason — the walk prefixes the move it was
    guarding and hands over.

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


def amounts(screen: Screen) -> list[float]:
    """Every ¥/￥ amount visible on the screen — `match.PRICE_RE`, the
    one spelling of a currency amount, run over raw row labels; never a
    model."""
    out: list[float] = []
    for row in screen.rows:
        out.extend(float(m) for m in PRICE_RE.findall(row.label))
    return out


def declared_total(screen: Screen, readings: tuple[str, ...]) -> float | None:
    """The amount beside the declared total label: on the label's own
    row when that row carries one, else the nearest price-bearing row
    within `TOTAL_RADIUS`. The first label row that yields an amount
    wins (top to bottom); None when no such row reads."""
    priced = [
        (pc, float(m[0]))
        for r in screen.rows
        if (m := PRICE_RE.findall(r.label)) and (pc := center_of(r.bbox)) is not None
    ]
    for row in screen.rows:
        if row.kind != "text" or not any(label_hit(r, row.label) for r in readings):
            continue
        own = PRICE_RE.findall(row.label)
        if own:
            return float(own[0])
        c = center_of(row.bbox)
        assert c is not None  # Element bboxes are valid by construction
        nearest = min(((math.dist(c, pc), amt) for pc, amt in priced), default=None)
        if nearest is not None and nearest[0] <= TOTAL_RADIUS:
            return nearest[1]
    return None


def fire_block(*, consented: float | None, screen: Screen) -> str | None:
    """The two fire-time predicates. None = pay; else the bare reason to
    block — the walk prefixes the move it was guarding."""
    if consented is None:
        return "reached without a confirmed total"
    amts = amounts(screen)
    if not any(abs(a - consented) < 0.01 for a in amts):
        return (
            f"sheet changed after consent: confirmed ¥{consented:g}, "
            f"now sees {amts or 'no amounts'}"
        )
    over = [a for a in amts if a > consented + 0.005]
    if over:
        return f"amount(s) {over} exceed the consented total ¥{consented:g}"
    return None
