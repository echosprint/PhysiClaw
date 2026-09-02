"""Tests for `physiclaw.conductor.rescue` — the ladder policy: rung
selection order, per-rung and global budgets, the dismissal vocabulary,
and the deny-list that keeps money-shaped rows untappable."""

from __future__ import annotations

import pytest
from conductor_fakes import make_screen

from physiclaw.common.listing import Element, Screen, format_elements
from physiclaw.conductor import rescue
from physiclaw.conductor.match import Verdict

BAND = (0.3, 0.7)


def _v(kind: str = "unknown", page: str | None = None, band=None) -> Verdict:
    return Verdict(kind, page, 0.5, 0.0, 0.0, "d", overlay_band=band)


POPUP = make_screen(
    ("new user gift pack", 0.5, 0.44),
    ("立即领取", 0.5, 0.48),
    ("以后再说", 0.5, 0.56),
)

DENIED_ONLY = make_screen(
    ("立即领取", 0.5, 0.48),
    ("去支付 ¥9.9", 0.5, 0.56),
)

LOCKED = make_screen(("Enter Passcode", 0.5, 0.5))

PLAIN = make_screen(("Nothing known", 0.5, 0.5))


# ---------- plan: rung selection and budgets ----------


def test_plan_locked_screen_unlocks_before_any_other_rung() -> None:
    step = rescue.plan(_v("occluded", "demo.home", BAND), LOCKED, {}, 0)

    assert isinstance(step, rescue.Unlock)


def test_plan_unlock_budget_spent_exhausts() -> None:
    step = rescue.plan(_v(), LOCKED, {"unlock": rescue.UNLOCK_TRIES}, 2)

    assert isinstance(step, rescue.Exhausted)
    assert "still locked" in step.reason


def test_plan_occluded_with_a_safe_word_dismisses() -> None:
    step = rescue.plan(_v("occluded", "demo.home", BAND), POPUP, {}, 0)

    assert isinstance(step, rescue.Dismiss)
    assert "以后再说" in step.note


def test_plan_occluded_without_a_safe_hit_falls_through_to_back() -> None:
    step = rescue.plan(_v("occluded", "demo.home", BAND), DENIED_ONLY, {"settle": 1}, 0)

    assert isinstance(step, rescue.Back)


def test_plan_dismiss_budget_spent_falls_through_to_back() -> None:
    step = rescue.plan(
        _v("occluded", "demo.home", BAND),
        POPUP,
        {"dismiss": rescue.DISMISS_TRIES, "settle": 1},
        2,
    )

    assert isinstance(step, rescue.Back)


def test_plan_unknown_screen_settles_once_then_goes_back() -> None:
    first = rescue.plan(_v(), PLAIN, {}, 0)
    assert isinstance(first, rescue.Settle)  # free re-peek before any action

    step = rescue.plan(_v(), PLAIN, {"settle": 1}, 0)

    assert isinstance(step, rescue.Back)


def test_plan_back_budget_spent_exhausts() -> None:
    step = rescue.plan(_v(), PLAIN, {"back": rescue.BACK_TRIES}, 2)

    assert isinstance(step, rescue.Exhausted)
    assert "back attempts" in step.reason


def test_plan_back_budget_with_the_reset_available_resets() -> None:
    step = rescue.plan(_v(), PLAIN, {"back": rescue.BACK_TRIES}, 2, can_reset=True)

    assert isinstance(step, rescue.Reset)


def test_plan_global_budget_wins_over_every_rung() -> None:
    step = rescue.plan(
        _v("occluded", "demo.home", BAND), POPUP, {}, rescue.GLOBAL_BUDGET
    )

    assert isinstance(step, rescue.Exhausted)
    assert "budget" in step.reason


# ---------- find_dismiss: vocabulary, deny-list, band scoping ----------


def test_find_dismiss_prefers_vocabulary_priority_order() -> None:
    screen = make_screen(("取消", 0.5, 0.4), ("关闭", 0.5, 0.6))

    step = rescue.find_dismiss(screen, BAND)

    assert step is not None and "关闭" in step.note  # earlier in the vocab


def test_find_dismiss_never_taps_money_shaped_rows() -> None:
    assert rescue.find_dismiss(DENIED_ONLY, BAND) is None


@pytest.mark.parametrize(
    "label",
    ["确认支付", "buy now", "订阅并领取", "¥12.9 起", "agree and continue"],
)
def test_deny_list_refuses_each_money_shape(label: str) -> None:
    screen = make_screen((label, 0.5, 0.5))

    assert rescue.find_dismiss(screen, BAND) is None


def test_find_dismiss_ignores_rows_outside_the_band() -> None:
    screen = make_screen(("以后再说", 0.5, 0.9))  # below the band

    assert rescue.find_dismiss(screen, BAND) is None


def test_find_dismiss_accepts_the_close_glyph() -> None:
    screen = make_screen(("×", 0.9, 0.35))

    step = rescue.find_dismiss(screen, BAND)

    assert step is not None and "glyph" in step.note


def test_find_dismiss_falls_back_to_the_top_right_icon() -> None:
    els = [
        Element(id=0, kind="icon", label="", bbox=(0.85, 0.32, 0.92, 0.36), conf=0.8),
        Element(
            id=1,
            kind="text",
            label="grand opening sale",
            bbox=(0.3, 0.45, 0.7, 0.5),
            conf=0.9,
        ),
    ]
    screen = Screen.read(format_elements(els))

    step = rescue.find_dismiss(screen, BAND)

    assert step is not None and "top-right icon" in step.note


def test_find_dismiss_ignores_low_left_icons() -> None:
    els = [
        Element(id=0, kind="icon", label="", bbox=(0.1, 0.6, 0.2, 0.65), conf=0.8),
    ]
    screen = Screen.read(format_elements(els))

    assert rescue.find_dismiss(screen, BAND) is None


def test_state_note_names_the_rungs_used() -> None:
    st = rescue.State(target="demo.home", mode=rescue.MODE_ENTER)
    st.tries.update({"dismiss": 1, "unlock": 0, "back": 2})

    assert st.note() == "dismiss×1, back×2"


# ---------- the micro tier (clear_overlay) ----------


def test_plan_without_a_vocab_hit_asks_the_micro_tier_once() -> None:
    step = rescue.plan(_v("occluded", "demo.home", BAND), DENIED_ONLY, {"settle": 1}, 0)
    assert isinstance(step, rescue.Back)  # every band row denied — nothing to ask

    neutral = make_screen(("limited offer", 0.5, 0.44), ("check details", 0.5, 0.52))
    step = rescue.plan(_v("occluded", "demo.home", BAND), neutral, {}, 0)

    assert isinstance(step, rescue.AskDismiss)
    assert {c.key for c in step.request.candidates} == {
        "limited offer",
        "check details",
    }


def test_plan_micro_tier_is_spent_after_one_try() -> None:
    neutral = make_screen(("limited offer", 0.5, 0.44), ("check details", 0.5, 0.52))

    step = rescue.plan(
        _v("occluded", "demo.home", BAND), neutral, {"micro": 1, "settle": 1}, 1
    )

    assert isinstance(step, rescue.Back)


def test_overlay_request_filters_denied_and_out_of_band_rows() -> None:
    screen = make_screen(
        ("check details", 0.5, 0.44),
        ("去支付", 0.5, 0.52),  # denied
        ("out of band", 0.5, 0.9),
    )

    req = rescue.overlay_request(screen, BAND)

    assert [c.key for c in req.candidates] == ["check details"]


def test_find_dismiss_uses_learned_labels_after_the_vocab() -> None:
    screen = make_screen(("开心收下", 0.5, 0.5))
    assert rescue.find_dismiss(screen, BAND) is None

    step = rescue.find_dismiss(screen, BAND, learned=("开心收下",))

    assert step is not None and "开心收下" in step.note


def test_learn_dismiss_round_trips_dedupes_and_caps() -> None:
    rescue.learn_dismiss("demo", "check details")
    rescue.learn_dismiss("demo", "check details")  # dedupe

    assert rescue.load_dismiss("demo") == ("check details",)


def test_learn_dismiss_refuses_denied_labels() -> None:
    rescue.learn_dismiss("demo", "去支付")

    assert rescue.load_dismiss("demo") == ()


def test_learn_dismiss_drops_oldest_past_the_cap() -> None:
    labels = [f"label {i}" for i in range(rescue.MAX_LEARNED + 2)]
    for label in labels:
        rescue.learn_dismiss("demo", label)

    learned = rescue.load_dismiss("demo")

    assert len(learned) == rescue.MAX_LEARNED
    assert learned[-1] == labels[-1] and labels[0] not in learned


# ---------- declared app chrome (`landmarks:`) ----------


def _landmarks(**kw):
    from physiclaw.conductor.pages import Landmark

    return {k: Landmark(label=(v[0],), bbox=v[1]) for k, v in kw.items()}


def test_scrim_dismiss_fires_outside_the_band_when_nothing_safer_exists() -> None:
    # No vocab word, no glyph, no icon in the band — the declared scrim
    # area (outside the overlay) is tapped before the micro tier is paid.
    screen = make_screen(("mystery offer", 0.5, 0.6), ("act fast", 0.5, 0.7))
    landmarks = _landmarks(dismiss=("scrim above the sheet", (0.35, 0.1, 0.65, 0.2)))

    step = rescue.plan(
        _v("occluded", page="demo.home", band=(0.5, 0.9)),
        screen,
        {},
        0,
        landmarks=landmarks,
    )

    assert isinstance(step, rescue.Dismiss)
    assert "outside the modal" in step.note
    assert step.bbox == (0.35, 0.1, 0.65, 0.2)


def test_scrim_dismiss_stands_down_inside_the_band() -> None:
    # A full-height modal leaves no scrim: the declared spot falls inside
    # the overlay band, so tapping it would hit modal content — skipped.
    screen = make_screen(("mystery offer", 0.5, 0.6))
    landmarks = _landmarks(dismiss=("scrim", (0.35, 0.1, 0.65, 0.2)))

    step = rescue.plan(
        _v("occluded", page="demo.home", band=(0.05, 0.9)),
        screen,
        {},
        0,
        landmarks=landmarks,
    )

    assert not isinstance(step, rescue.Dismiss)


def test_vocab_dismissal_still_beats_the_scrim() -> None:
    # Explicit close affordances first — the field's hierarchy.
    screen = make_screen(("以后再说", 0.5, 0.6))
    landmarks = _landmarks(dismiss=("scrim", (0.35, 0.1, 0.65, 0.2)))

    step = rescue.plan(
        _v("occluded", page="demo.home", band=(0.5, 0.9)),
        screen,
        {},
        0,
        landmarks=landmarks,
    )

    assert isinstance(step, rescue.Dismiss)
    assert "以后再说" in step.note


def test_back_rung_carries_the_declared_back_control() -> None:
    screen = make_screen(("Nothing known", 0.5, 0.5))
    landmarks = _landmarks(back=("back chevron", (0.02, 0.05, 0.1, 0.1)))

    step = rescue.plan(
        _v("unknown"), screen, {rescue.RUNG_SETTLE: 1}, 1, landmarks=landmarks
    )

    assert isinstance(step, rescue.Back)
    assert step.landmark is not None and step.landmark.bbox == (0.02, 0.05, 0.1, 0.1)


def test_locate_landmark_follows_the_label_within_radius() -> None:
    from physiclaw.conductor.pages import Landmark

    landmark = Landmark(label=("返回",), bbox=(0.02, 0.05, 0.1, 0.1))
    screen = make_screen(("返回", 0.1, 0.12))  # drifted a little

    bbox, note = rescue.locate_landmark(landmark, screen)

    assert note and bbox != landmark.bbox
    landmark_far = Landmark(label=("返回",), bbox=(0.8, 0.8, 0.9, 0.9))
    bbox2, note2 = rescue.locate_landmark(landmark_far, screen)
    assert bbox2 == landmark_far.bbox and note2 == ""  # off-radius → declared
