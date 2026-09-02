"""Tests for `physiclaw.conductor.recover` — declared recovery's policy:
the page's declared hand, and the budget that bounds it."""

from __future__ import annotations

import pytest
from conductor_fakes import make_screen

from physiclaw.conductor import recover
from physiclaw.conductor.pages import Landmark
from physiclaw.conductor.playbook import MAX_RECOVER_ACTIONS, RecoverHand, Recovery

HAND = RecoverHand(tool="go_back")
RECOVERY = Recovery(occluded=HAND, elsewhere=HAND, limit=2)


def test_the_declared_hand_is_the_plan() -> None:
    step = recover.plan(0, RECOVERY)

    assert isinstance(step, recover.Hand) and step.hand is HAND


def test_page_without_a_hand_is_exhausted_at_once() -> None:
    # Nothing recovers in the background: no re-peek, no unlock, no tap.
    step = recover.plan(0, None)

    assert isinstance(step, recover.Exhausted) and "declares no recover" in step.reason


def test_global_budget_wins_over_the_hand() -> None:
    step = recover.plan(MAX_RECOVER_ACTIONS, RECOVERY)

    assert isinstance(step, recover.Exhausted) and "budget" in step.reason


def test_page_limit_is_spent_before_the_walk_budget() -> None:
    step = recover.plan(1, RECOVERY, page_actions=RECOVERY.limit)

    assert isinstance(step, recover.Exhausted) and "recover limit (2)" in step.reason


def test_keyed_hands_follow_the_reading() -> None:
    # A page declaring only an `occluded` hand hands over on any other
    # screen — and vice versa: what is declared is what runs.
    dismiss = RecoverHand(tool="tap", landmark="dismiss")
    only_occluded = Recovery(occluded=dismiss)

    step = recover.plan(0, only_occluded, reading="occluded")
    assert isinstance(step, recover.Hand) and step.hand is dismiss
    step = recover.plan(0, only_occluded, reading="elsewhere")
    assert isinstance(step, recover.Exhausted) and "`elsewhere`" in step.reason


def test_state_rejects_an_unknown_mode() -> None:
    with pytest.raises(ValueError, match="unknown recovery mode"):
        recover.State(target="demo.home", mode="sideways")


def test_locate_landmark_follows_the_label_within_radius() -> None:
    landmark = Landmark(label=("返回",), bbox=(0.02, 0.05, 0.1, 0.1))
    screen = make_screen(("返回", 0.1, 0.12))  # drifted a little

    bbox, note = recover.locate_landmark(landmark, screen)

    assert note and bbox != landmark.bbox
    far = Landmark(label=("返回",), bbox=(0.8, 0.8, 0.9, 0.9))
    bbox2, note2 = recover.locate_landmark(far, screen)
    assert bbox2 == far.bbox and note2 == ""  # off-radius → declared
