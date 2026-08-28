"""The cart reconciler's step planner — one divergence, one decision.

Pure policy over the reconciler's senses (`ledger.assign_rows`,
`ledger.row_qty`): given the cart screen and the desired list, name the
ONE next thing to do. Every rule lives here — the re-shop bound, the
action budget, what convergence means — while the walk (`program.py`)
owns the state: it increments the counters this module consults, resets
a re-shopped item to pending, and routes the cursor.

  - Converged (returned as None) = every picked item reads at its
    wanted quantity. Cart rows the ledger does not claim are LEFT
    ALONE (they may be the user's own — the conductor never destroys
    what it cannot attribute to itself).
  - A picked item missing from the cart re-enters the shopping loop —
    once. A second miss means the pick's label will never read as a
    cart row, and re-shopping again just duplicates real adds.
  - A quantity off by any amount is one stepper tap; the tap's own
    result screen is the re-read (there is no other verification).
  - Every divergence costs at least one tap-and-reread action, so a
    converging cart finishes well under `MAX_ACTIONS` — hitting it
    means the cart is NOT converging.
"""

from dataclasses import dataclass
from typing import Mapping

from physiclaw.common.listing import Element, Screen
from physiclaw.conductor import ledger

MAX_ACTIONS = 16


@dataclass(frozen=True)
class Tap:
    """One stepper press: the element to tap, and the journal note."""

    el: Element
    note: str


@dataclass(frozen=True)
class Reshop:
    """A picked item is missing — send THIS item back through the loop."""

    item_idx: int


@dataclass(frozen=True)
class Blocked:
    """Not convergeable — the walk hands over with this reason."""

    reason: str


# None = converged, the package's idle answer (`fire_block` None = pay,
# `advance` None = the LLM speaks).
Step = Tap | Reshop | Blocked | None


def plan(
    screen: Screen,
    items: list[ledger.LedgerItem],
    reshops: Mapping[int, int],
    actions: int,
) -> Step:
    """The next reconciliation step for this reading of the cart.
    Read-only: `reshops` (per-item re-shop counts) and `actions` (this
    node's action total) are the walk's counters, consulted here against
    the bounds and incremented by the walk when it acts."""
    if actions > MAX_ACTIONS:
        return Blocked(
            f"cart not converging after {MAX_ACTIONS} actions — "
            f"list: {ledger.describe(items)}"
        )
    assigned = ledger.assign_rows(screen, items)
    for idx, item in enumerate(items):
        if item.status != "picked":
            continue  # pending items are the loop's, reached via re-shop
        row = assigned[idx]
        if row is None:
            if item.qty <= 0:
                continue  # removed and absent — converged
            if reshops.get(idx, 0) >= 1:
                return Blocked(
                    f"{item.query!r} still missing after a re-shop — "
                    f"list: {ledger.describe(items)}"
                )
            return Reshop(idx)
        found = ledger.row_qty(screen, row)
        if found is None:
            return Blocked(
                f"no readable qty/steppers beside {row.label!r} — "
                f"list: {ledger.describe(items)}"
            )
        have, minus, plus = found
        if have < item.qty:
            return Tap(plus, f"{item.query} {have}→{item.qty}")
        if have > item.qty:
            return Tap(minus, f"{item.query} {have}→{item.qty}")
    return None
