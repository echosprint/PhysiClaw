"""Money runs in code — the payment doctrine's predicates, one home.

Three pure functions, deliberately free of walk state so the money
rules can be read (and audited) without the state machine around them.
The walk (`program.py`) supplies the numbers and acts on the answers:

  - `amounts` — what the screen says. The ask quotes `max(amounts)`;
    the fire-time check re-reads the same way.
  - `mandate_cap` — what the playbook allows: a literal, or an input
    the user's value fills. None means "could not resolve", and the
    gate refuses to ask rather than asking with an unenforceable rule.
  - `fire_block` — the two fire-time predicates, run against the
    CURRENT screen after the human's consent: staleness (the sheet
    must still show the consented total) and the bound (nothing may
    exceed max(cap, consented)). None = pay; else the bare reason —
    the walk prefixes the leg it was guarding, logs it, and hands over.

Consent itself — quoting, binding, consuming — stays with the gate in
`program.py`: consent is a conversation, these are arithmetic.
"""

from physiclaw.common.listing import Screen
from physiclaw.conductor.match import PRICE_RE
from physiclaw.conductor.playbook import Mandate


def amounts(screen: Screen) -> list[float]:
    """Every ¥/￥ amount visible on the screen — `match.PRICE_RE`, the
    one spelling of a currency amount, run over raw row labels; never a
    model."""
    out: list[float] = []
    for row in screen.rows:
        out.extend(float(m) for m in PRICE_RE.findall(row.label))
    return out


def mandate_cap(mandate: Mandate | None, values: dict[str, str]) -> float | None:
    if mandate is None:
        return None
    if isinstance(mandate.max_amount, float):
        return mandate.max_amount
    try:
        return float(values[mandate.max_amount.name])
    except (KeyError, ValueError):
        return None


def fire_block(
    *,
    consented: float | None,
    cap: float | None,
    screen: Screen,
) -> str | None:
    """The two fire-time predicates. None = pay; else the bare reason to
    block — the walk prefixes the leg it was guarding."""
    if consented is None:
        return "reached without a confirmed total"
    amts = amounts(screen)
    if not any(abs(a - consented) < 0.01 for a in amts):
        return (
            f"sheet changed after consent: confirmed ¥{consented:g}, "
            f"now sees {amts or 'no amounts'}"
        )
    bound = max(cap or 0.0, consented)
    over = [a for a in amts if a > bound + 0.005]
    if over:
        return f"amount(s) {over} exceed the consented bound ¥{bound:g}"
    return None
