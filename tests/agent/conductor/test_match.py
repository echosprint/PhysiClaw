"""Tests for `physiclaw.agent.conductor.match` — normalization, fuzzy
tiers, scoring, and the open-set decision."""

from __future__ import annotations

from conductor_fakes import make_screen

from physiclaw.agent.conductor import match as m
from physiclaw.agent.conductor.pages import (
    AnchorDecl,
    LearnedAnchor,
    LearnedPage,
    PageDecl,
    PagePrint,
)
from physiclaw.common.listing import Screen


def _learned(text: str, cx: float, cy: float, *, weight=1.0, variants=()):
    return LearnedAnchor(
        text=text,
        cx=cx,
        cy=cy,
        pos_tol=0.02,
        freq=1.0,
        weight=weight,
        variants=variants,
    )


def _print(
    *,
    anchors,
    learned_anchors=None,
    forbid=(),
    scrollable=False,
    threshold=0.6,
    name="page",
) -> PagePrint:
    decl = PageDecl(
        name=name,
        anchors=tuple(anchors),
        forbid=tuple(forbid),
        scrollable=scrollable,
    )
    learned = None
    if learned_anchors is not None:
        learned = LearnedPage(
            anchors={a.text: a for a in learned_anchors},
            threshold=threshold,
            observations=6,
        )
    return PagePrint(app="app", decl=decl, learned=learned)


# ---------- normalization + fuzzy tiers ----------


def test_normalize_folds_width_case_space_and_volatile_spans() -> None:
    assert m.normalize("ｉＰｈｏｎｅ　12  Pro") == "iphone<NUM>pro"
    assert m.normalize("¥12.50") == "<PRICE>"
    assert m.normalize("20:07") == "<TIME>"
    assert m.normalize("综 合") == "综合"


def test_bigram_dice_tolerates_one_substitution_in_long_strings() -> None:
    assert m.bigram_dice("搜索历史记录", "搜素历史记录") >= 0.5


def test_short_cjk_anchor_matches_single_char_confusion() -> None:
    assert m.label_matches("综合", "综台", ()) is True
    assert m.label_matches("综合", "点击综台按钮", ()) is True
    assert m.label_matches("综合", "完全无关", ()) is False


def test_edit_ratio_bounds() -> None:
    assert m.edit_ratio("abc", "abc") == 0.0
    assert m.edit_ratio("abcd", "abce") == 0.25
    assert m.edit_ratio("", "abc") == 1.0


def test_label_matches_single_char_is_whole_label() -> None:
    assert m.label_matches("x", "x", ()) is True
    assert m.label_matches("x", "box", ()) is False


def test_label_matches_variant_short_circuits() -> None:
    assert m.label_matches("综合", "完全不同", ("完全不同",)) is True


# ---------- scoring ----------


def test_score_full_match_with_geometry() -> None:
    pp = _print(
        anchors=[AnchorDecl("综合"), AnchorDecl("销量")],
        learned_anchors=[_learned("综合", 0.2, 0.1), _learned("销量", 0.4, 0.1)],
    )
    screen = make_screen(("综合", 0.2, 0.1), ("销量", 0.4, 0.1))

    s = m.score_page(pp, screen)

    assert s.score == 1.0
    assert not s.missing


def test_score_rejects_geometry_drift_beyond_tolerance() -> None:
    pp = _print(
        anchors=[AnchorDecl("综合")],
        learned_anchors=[_learned("综合", 0.2, 0.1)],
    )
    screen = make_screen(("综合", 0.2, 0.5))  # right text, wrong place

    s = m.score_page(pp, screen)

    assert s.score == 0.0
    assert s.missing == ("综合",)


def test_score_scrollable_page_votes_shared_dy() -> None:
    pp = _print(
        anchors=[AnchorDecl("份量"), AnchorDecl("商家")],
        learned_anchors=[_learned("份量", 0.2, 0.30), _learned("商家", 0.2, 0.50)],
        scrollable=True,
    )
    # Both anchors shifted up by the same 0.2 — one scroll offset.
    screen = make_screen(("份量", 0.2, 0.10), ("商家", 0.2, 0.30))

    s = m.score_page(pp, screen)

    assert s.score == 1.0
    assert abs(s.dy + 0.2) < 0.03


def test_score_forbid_vetoes() -> None:
    pp = _print(anchors=[AnchorDecl("综合")], forbid=["直播中"])
    screen = make_screen(("综合", 0.2, 0.1), ("直播中", 0.5, 0.5))

    s = m.score_page(pp, screen)

    assert s.forbidden is True
    assert s.score == 0.0


def test_score_region_hint_rejects_out_of_band_row() -> None:
    pp = _print(anchors=[AnchorDecl("搜索", region="top")])
    screen = make_screen(("搜索", 0.5, 0.9))  # bottom of screen

    s = m.score_page(pp, screen)

    assert s.score == 0.0


# ---------- open-set decision ----------


def test_match_screen_accepts_above_threshold_with_margin() -> None:
    good = _print(
        name="results",
        anchors=[AnchorDecl("综合"), AnchorDecl("销量")],
        learned_anchors=[_learned("综合", 0.2, 0.1), _learned("销量", 0.4, 0.1)],
    )
    other = _print(name="cart", anchors=[AnchorDecl("结算")])
    screen = make_screen(("综合", 0.2, 0.1), ("销量", 0.4, 0.1))

    v = m.match_screen(screen, [good, other])

    assert v.kind == "match"
    assert v.page_id == "app.results"


def test_match_screen_unknown_when_below_threshold() -> None:
    pp = _print(
        anchors=[AnchorDecl("综合"), AnchorDecl("销量"), AnchorDecl("筛选")],
        learned_anchors=[
            _learned("综合", 0.2, 0.1),
            _learned("销量", 0.4, 0.1),
            _learned("筛选", 0.6, 0.1),
        ],
        threshold=0.8,
    )
    screen = make_screen(("综合", 0.2, 0.1))  # 1 of 3

    v = m.match_screen(screen, [pp])

    assert v.kind == "unknown"


def test_match_screen_unknown_without_margin_over_runner_up() -> None:
    a = _print(name="a", anchors=[AnchorDecl("共享"), AnchorDecl("独有甲")])
    b = _print(name="b", anchors=[AnchorDecl("共享"), AnchorDecl("独有乙")])
    screen = make_screen(("共享", 0.5, 0.1))  # only the shared anchor

    v = m.match_screen(screen, [a, b])

    assert v.kind == "unknown"


def test_match_screen_unreadable_is_unknown() -> None:
    pp = _print(anchors=[AnchorDecl("综合")])

    v = m.match_screen(Screen.read(""), [pp])

    assert v.kind == "unknown"


def test_match_screen_occluded_when_missing_anchors_share_a_band() -> None:
    pp = _print(
        name="thread",
        anchors=[AnchorDecl("妈妈"), AnchorDecl("发送"), AnchorDecl("语音")],
        learned_anchors=[
            _learned("妈妈", 0.5, 0.05),
            _learned("发送", 0.9, 0.85),  # bottom band — under the keyboard
            _learned("语音", 0.1, 0.90),
        ],
        threshold=0.9,
    )
    # Top anchor visible; bottom band shows keyboard keys instead.
    screen = make_screen(
        ("妈妈", 0.5, 0.05),
        ("qwerty", 0.5, 0.80),
        ("asdfgh", 0.5, 0.88),
        ("zxcvbn", 0.5, 0.95),
    )

    v = m.match_screen(screen, [pp])

    assert v.kind == "occluded"
    assert v.page_id == "app.thread"
