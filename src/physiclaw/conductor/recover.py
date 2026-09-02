"""Declared recovery — a page's `recover:` hand, and nothing else.

Pure policy over the walk's counter, the `plan` shape: given a check
that needed page P and did not get it, name the ONE next action, or
Exhausted and the walk hands over. What the playbook declares is what
runs: a page with a `recover:` hand runs it (after it the walk re-checks
on the hand's own result view and, still off, walks the route again
from its first unsettled node — a `force_quit` hand re-runs `start`,
its whole point); a page declaring none hands over on the spot.
Nothing taps, unlocks, or waits in the background — a phone that may
lock mid-walk gets an `unlock_phone` hand declared by its author.

Everything is bounded by one walk-wide budget that only ever goes up
(the rebinding-attack lesson: recovery must not be farmable into a
loop). Handover stays the floor.
"""

from dataclasses import dataclass

from physiclaw.common.bbox import Bbox, center_of
from physiclaw.common.listing import Screen, nearest_labeled_row
from physiclaw.conductor.pages import Landmark
from physiclaw.conductor.playbook import RecoverHand
from physiclaw.macros.steps import HEAL_RADIUS

# Fixed, not authorable: the walk-wide budget of recovery actions —
# what stops a splash ad on every cold launch from relaunching forever.
GLOBAL_BUDGET = 6

# Resume modes — how the walk continues once the target page is
# restored. Validated at State construction: a typo'd mode must fail
# loudly, never silently resume as "enter".
MODE_ENTER = "enter"  # re-check the enter, then run the move
MODE_VERIFY = "verify"  # the move already ran — verify is satisfied
MODES = (MODE_ENTER, MODE_VERIFY)


@dataclass(frozen=True)
class Hand:
    """The page's declared `recover:` hand. The walk owns its
    interpretation (tool / landmark tap / macro)."""

    hand: RecoverHand


@dataclass(frozen=True)
class Exhausted:
    """Nothing declared is left to try — the walk hands over with `reason`."""

    reason: str


Step = Hand | Exhausted


@dataclass(frozen=True)
class State:
    """One recovery in flight: the page the frozen cursor requires, how
    the walk resumes once it is restored, and the ORIGINAL check failure
    (the exhausted handover reports what actually went wrong, not the
    hand's miss). Owned by the walk; cleared on the hand's landing."""

    target: str  # full `app.page` id the interrupted check needs
    mode: str  # one of MODES — the resume rule
    reason: str = ""

    def __post_init__(self) -> None:
        if self.mode not in MODES:
            raise ValueError(f"unknown recovery mode {self.mode!r}")


def plan(actions: int, hand: RecoverHand | None) -> Step:
    """The next recovery action: the declared hand, within the walk's
    lifetime budget (`actions`, the recovery actions spent so far)."""
    if actions >= GLOBAL_BUDGET:
        return Exhausted(f"recovery budget ({GLOBAL_BUDGET} actions) spent")
    if hand is None:
        return Exhausted("its page declares no recover")
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
