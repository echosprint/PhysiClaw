"""Declared recovery — one deviation, one next action, what the
playbook says.

Pure policy over the walk's senses, the `plan` shape: given a check
that needed page P and did not get it, name the ONE next action, or
Exhausted and the walk hands over. The walk (`program.py`) owns the
state: it increments the counters this module consults, synthesizes
the action, and re-checks on the action's own result screen (the
action's result view IS the re-read).

Two built-ins precede the page's own hand, because they are about the
phone, not the app:

  locked (the cover or the passcode keypad) → ``unlock_phone``. Read
      before anything else: taps do not reach a sleeping phone, so
      every other action is inert against it (the overture's lesson).
  settle → one FREE re-peek before any recovery action is spent: a
      landing judged off a gesture's immediate fused view may still be
      animating — the flakiness rule every mature harness encodes.

Then the page's DECLARED ``recover:`` hand, once per engagement; after
it runs the walk re-locates on the route and walks again (a
``force_quit`` hand re-runs the ``start`` — its whole point). A page
declaring none is exhausted immediately: what you declare is what runs,
and nothing hidden taps around. Everything is bounded by one walk-wide
budget and the counters only ever go up (the rebinding-attack lesson:
recovery must not be farmable into a loop). Handover stays the floor.
"""

from dataclasses import dataclass, field

from physiclaw.common.bbox import Bbox, center_of
from physiclaw.common.listing import Screen, nearest_labeled_row
from physiclaw.conductor.match import Verdict, reads_as_locked
from physiclaw.conductor.pages import Landmark
from physiclaw.conductor.playbook import RecoverHand
from physiclaw.macros.steps import HEAL_RADIUS

# Fixed, not authorable — the GATE_MAX_CHECKS discipline. The global
# budget bounds the whole walk's recovery spend (hands and unlocks);
# the unlock cap stops a dead phone from eating it.
GLOBAL_BUDGET = 6
UNLOCK_TRIES = 2

# Rung names — the tries-counter keys `plan` reads and the walk writes
# (also the tail of each `recover-<rung>` pending kind). Named so a
# typo on either side of the module boundary fails a pin, not a budget.
RUNG_SETTLE = "settle"
RUNG_UNLOCK = "unlock"
RUNG_HAND = "hand"

# Resume modes — how the walk continues once the target page is
# restored. Validated at State construction: a typo'd mode must fail
# loudly, never silently resume as "enter".
MODE_ENTER = "enter"  # re-check the enter, then run the move
MODE_VERIFY = "verify"  # the move already ran — verify is satisfied
MODES = (MODE_ENTER, MODE_VERIFY)


@dataclass(frozen=True)
class Settle:
    """One free re-peek before any recovery action is spent."""


@dataclass(frozen=True)
class Unlock:
    """The phone reads locked — `unlock_phone` before anything else."""


@dataclass(frozen=True)
class Hand:
    """The page's declared `recover:` hand, run once per engagement.
    The walk owns its interpretation (tool / landmark tap / macro);
    this planner only decides WHEN it is spent."""

    hand: RecoverHand


@dataclass(frozen=True)
class Exhausted:
    """Nothing left to try — the walk hands over with `reason`."""

    reason: str


Step = Settle | Unlock | Hand | Exhausted


@dataclass
class State:
    """One recovery in flight: the page the frozen cursor requires, how
    the walk resumes once it is restored, and the counters `plan`
    consults. Owned by the walk; cleared only on resume or handover.
    `reason` is the ORIGINAL check failure — the exhausted handover
    reports what actually went wrong, not the last action's miss."""

    target: str  # full `app.page` id the interrupted check needs
    mode: str  # one of MODES — the resume rule
    reason: str = ""
    tries: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.mode not in MODES:
            raise ValueError(f"unknown recovery mode {self.mode!r}")

    def note(self) -> str:
        """The recovery actions used, terse — for journal lines and the
        brief. The settle re-peek is verification hygiene, not a
        recovery action, so it is not reported as one."""
        return ", ".join(
            f"{k}×{v}" for k, v in self.tries.items() if v and k != RUNG_SETTLE
        )


def plan(
    verdict: Verdict,
    screen: Screen,
    tries: dict,
    actions: int,
    hand: RecoverHand | None,
) -> Step:
    """The next recovery action for this reading of the screen. Read-only
    over the walk's counters: `tries` is this engagement's per-rung
    count, `actions` the walk's lifetime recovery-action count (the
    budget — what stops a splash ad on every cold launch from relaunching
    forever), `hand` the target page's declared recovery, or None."""
    if actions >= GLOBAL_BUDGET:
        return Exhausted(f"recovery budget ({GLOBAL_BUDGET} actions) spent")
    if reads_as_locked(screen):
        if tries.get(RUNG_UNLOCK, 0) >= UNLOCK_TRIES:
            return Exhausted(f"phone still locked after {UNLOCK_TRIES} unlock attempts")
        return Unlock()
    if not tries.get(RUNG_SETTLE, 0):
        return Settle()
    if hand is None:
        return Exhausted(
            f"screen reads as {verdict.kind} "
            f"({verdict.page_id or 'no known page'}) and its page declares "
            "no recover"
        )
    if tries.get(RUNG_HAND, 0):
        return Exhausted("the declared recover hand already ran")
    return Hand(hand)


def locate_landmark(landmark: Landmark, screen: Screen) -> "tuple[Bbox, str]":
    """Where a declared landmark IS right now: the text row matching one
    of its label readings nearest the declared spot (`nearest_labeled_row`
    — the one search the macro heal rides too, within the one
    HEAL_RADIUS), else the declared bbox. Returns the bbox to tap and a
    short note for the journal."""
    if not screen.readable:
        return landmark.bbox, ""
    declared = center_of(landmark.bbox)
    assert declared is not None  # Landmark bboxes are parse-validated
    best = nearest_labeled_row(screen.rows, landmark.label, declared)
    if best is None or best[0] > HEAL_RADIUS:
        return landmark.bbox, ""
    return best[1].bbox, f" (located {best[1].label.strip()!r} on screen)"
