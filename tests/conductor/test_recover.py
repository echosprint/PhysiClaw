"""Tests for `physiclaw.conductor.recover` — declared recovery's policy:
the built-in unlock and settle, the page's one declared hand, and the
budgets that bound them."""

from __future__ import annotations

from conductor_fakes import make_screen

from physiclaw.conductor import recover
from physiclaw.conductor.match import Verdict
from physiclaw.conductor.pages import Landmark
from physiclaw.conductor.playbook import RecoverHand


def _v(kind: str = "unknown", page: str | None = None) -> Verdict:
    return Verdict(kind, page, 0.5, 0.0, 0.0, "d")


LOCKED = make_screen(("Enter Passcode", 0.5, 0.5))
PLAIN = make_screen(("Nothing known", 0.5, 0.5))
HAND = RecoverHand(tool="go_back")


def test_locked_screen_unlocks_before_anything_else() -> None:
    step = recover.plan(_v(), LOCKED, {}, 0, HAND)

    assert isinstance(step, recover.Unlock)


def test_unlock_budget_spent_exhausts() -> None:
    step = recover.plan(_v(), LOCKED, {recover.RUNG_UNLOCK: 2}, 2, HAND)

    assert isinstance(step, recover.Exhausted) and "locked" in step.reason


def test_first_reading_settles_for_free() -> None:
    step = recover.plan(_v(), PLAIN, {}, 0, HAND)

    assert isinstance(step, recover.Settle)


def test_settled_screen_runs_the_declared_hand_once() -> None:
    tries = {recover.RUNG_SETTLE: 1}

    first = recover.plan(_v(), PLAIN, tries, 0, HAND)
    assert isinstance(first, recover.Hand) and first.hand is HAND

    tries[recover.RUNG_HAND] = 1
    again = recover.plan(_v(), PLAIN, tries, 1, HAND)
    assert isinstance(again, recover.Exhausted) and "already ran" in again.reason


def test_page_without_a_hand_is_exhausted_after_the_settle() -> None:
    step = recover.plan(
        _v("wrong", "demo.home"), PLAIN, {recover.RUNG_SETTLE: 1}, 0, None
    )

    assert isinstance(step, recover.Exhausted)
    assert "declares no recover" in step.reason and "demo.home" in step.reason


def test_global_budget_wins_over_every_rung() -> None:
    step = recover.plan(_v(), LOCKED, {}, recover.GLOBAL_BUDGET, HAND)

    assert isinstance(step, recover.Exhausted) and "budget" in step.reason


def test_state_note_names_the_actions_used() -> None:
    st = recover.State(target="demo.home", mode=recover.MODE_ENTER)
    st.tries = {recover.RUNG_SETTLE: 1, recover.RUNG_UNLOCK: 1, recover.RUNG_HAND: 1}

    assert st.note() == "unlock×1, hand×1"  # the settle is not an action


def test_state_rejects_an_unknown_mode() -> None:
    import pytest

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
