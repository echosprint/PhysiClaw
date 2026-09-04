"""Declared recovery — a page's `recover:` hand, and nothing else.

Pure policy over the walk's counters: given a check that needed page P
and did not get it, `plan` names the ONE next action — the hand the
page declared for the reading (`covered`: P under a sheet or popup;
`locked`: the lock screen, read by shape; `elsewhere`: any other
screen; the flat form declares one hand for all three) — or Exhausted,
and the walk hands over. After a hand the walk re-checks on its own
result view and, still off, walks the route again from its first
unsettled node (a `force_quit` hand re-runs `start`). Nothing taps,
unlocks, or waits in the background.

Two bounds, both visible in the playbook: the page's own `tries:` and
the walk-wide ceiling `MAX_RECOVER_ACTIONS`.
"""

from dataclasses import dataclass
from enum import StrEnum

from physiclaw.common.bbox import Bbox, center_of
from physiclaw.common.listing import Screen, nearest_labeled_row
from physiclaw.conductor.spec.limits import MAX_RECOVER_ACTIONS
from physiclaw.conductor.spec.model import READING_ELSEWHERE, RecoverHand, Recovery
from physiclaw.conductor.spec.pages import Landmark
from physiclaw.macros.steps import HEAL_RADIUS


class Mode(StrEnum):
    """How the walk continues once the target page is restored."""

    ENTER = "enter"  # re-check the enter, then run the move
    VERIFY = "verify"  # the move already ran — verify is satisfied


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
    mode: Mode  # the resume rule
    reason: str = ""

    def __post_init__(self) -> None:
        # A typo'd mode must fail loudly, never silently resume as enter.
        if not isinstance(self.mode, Mode):
            raise ValueError(f"unknown recovery mode {self.mode!r}")


def plan(
    actions: int,
    recovery: Recovery | None,
    page_actions: int = 0,
    *,
    reading: str = READING_ELSEWHERE,
) -> Step:
    """The next recovery action: the hand the page declares for this
    `reading` (one of `playbook.RECOVER_READINGS`), within the page's
    own `tries` (`page_actions`, its actions so far) and the walk's
    lifetime ceiling (`actions`)."""
    if actions >= MAX_RECOVER_ACTIONS:
        return Exhausted(f"recovery budget ({MAX_RECOVER_ACTIONS} actions) spent")
    if recovery is None:
        return Exhausted("its page declares no recover")
    if page_actions >= recovery.tries:
        return Exhausted(f"its page's recover tries ({recovery.tries}) spent")
    hand = recovery.hand_for(reading)
    if hand is None:
        return Exhausted(f"its page declares no `{reading}` recover hand")
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
