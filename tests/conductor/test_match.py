"""Tests for `physiclaw.conductor.match` — normalization, fuzzy
tiers, scoring, and the open-set decision."""

from __future__ import annotations

from conductor_fakes import make_screen

from physiclaw.common.listing import LISTING_HEADER, Screen
from physiclaw.conductor import match as m
from physiclaw.conductor.pages import (
    AnchorDecl,
    LearnedAnchor,
    LearnedPage,
    PageDecl,
    PagePrint,
)


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


# ---------- anchor alternates ----------


def test_alternate_readings_match_either_way() -> None:
    pp = _print(anchors=[AnchorDecl("Search", alts=("搜索",))])

    for label in ("Search", "搜索"):
        assert m.score_page(pp, make_screen((label, 0.5, 0.1))).score == 1.0


def test_alternates_count_as_one_anchor_where_two_anchors_would_halve() -> None:
    """The regression alternates exist to prevent: a mixed-locale device
    shows ONE of the two spellings, so declaring them as separate anchors
    scores 0.5 — under the declaration-only threshold, on the right page."""
    screen = make_screen(("Search", 0.5, 0.1), ("other", 0.5, 0.5))

    together = _print(anchors=[AnchorDecl("Search", alts=("搜索",))])
    apart = _print(anchors=[AnchorDecl("Search"), AnchorDecl("搜索")])

    assert m.score_page(together, screen).score == 1.0
    assert m.score_page(apart, screen).score == 0.5


def test_alternate_reading_still_gets_the_fuzzy_tiers() -> None:
    # 综台 is the classic one-character OCR confusion for 综合; an alternate
    # must be held to the same tiers as a lone anchor, not exact-matched.
    pp = _print(anchors=[AnchorDecl("Sort", alts=("综合",))])

    assert m.score_page(pp, make_screen(("综台", 0.5, 0.1))).score == 1.0


def test_alternates_report_the_canonical_reading() -> None:
    # hits/missing name the canonical spelling whichever alternate landed,
    # so learned geometry and reports key off one string.
    pp = _print(anchors=[AnchorDecl("Search", alts=("搜索",))])

    hit = m.score_page(pp, make_screen(("搜索", 0.5, 0.1)))
    miss = m.score_page(pp, make_screen(("nothing", 0.5, 0.1)))

    assert [h.anchor for h in hit.hits] == ["Search"]
    assert miss.missing == ("Search",)


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


def test_overlay_band_is_padded_around_the_missing_anchors() -> None:
    # Only the padding (0.85/0.90 ± OVERLAY_PAD) makes three unexpected
    # rows fall inside the band — without it the verdict is unknown.
    pp = _print(
        name="thread",
        anchors=[AnchorDecl("妈妈"), AnchorDecl("发送"), AnchorDecl("语音")],
        learned_anchors=[
            _learned("妈妈", 0.5, 0.05),
            _learned("发送", 0.9, 0.85),
            _learned("语音", 0.1, 0.90),
        ],
        threshold=0.9,
    )
    screen = make_screen(
        ("妈妈", 0.5, 0.05),
        ("qwerty", 0.5, 0.80),
        ("asdfgh", 0.5, 0.88),
        ("zxcvbn", 0.5, 0.95),
    )

    v = m.match_screen(screen, [pp])

    assert v.kind == "occluded" and v.page_id == "app.thread"


# ---------- the cover, which has no page ----------

# Captured live off the rig (`physiclaw mcp -H`, iPhone locked), because
# the whole point of `reads_as_cover` is that the cover's real shape is
# not what a synthetic fixture would guess. Note the clock's width: the
# hero clock spans most of the screen, the status bar's spans a tenth.
COVER_RESTING = f"""{LISTING_HEADER}
0 [text] "Thu Aug 27" [0.341,0.079,0.591,0.110] 0.94
1 [text] "21:45" [0.121,0.095,0.828,0.248] 0.98"""

# Same phone, woken by a tap and fully lit — no dark warning on this one,
# which is why the camera's brightness verdict cannot be the signal.
COVER_WOKEN = f"""{LISTING_HEADER}
0 [icon] "" [0.282,0.000,0.332,0.025] 0.56
1 [text] "Thu Aug 27" [0.335,0.065,0.596,0.099] 0.94
2 [text] "21:46" [0.093,0.086,0.846,0.239] 0.99"""

# The cover with a notification stack — the frame a row-count rule got
# wrong: as many rows as an app screen, but still unmistakably the cover.
COVER_WITH_NOTIFICATION = f"""{LISTING_HEADER}
0 [text] "Thu Aug 27" [0.335,0.065,0.597,0.099] 0.94
1 [text] "21:35" [0.101,0.084,0.842,0.239] 0.97
2 [icon] "" [0.412,0.109,0.473,0.160] 0.33
3 [text] "1m ago" [0.818,0.791,0.915,0.807] 0.93
4 [text] "WeChat" [0.231,0.804,0.364,0.826] 0.99
5 [text] "Notification" [0.233,0.822,0.413,0.845] 0.99"""

# An ordinary unlocked screen. Row 0 normalizes to `<TIME>` exactly like
# the cover's clock, so the token alone can never be the discriminator.
UNLOCKED_APP = f"""{LISTING_HEADER}
0 [text] "19:31" [0.111,0.004,0.219,0.027] 0.98
1 [text] "Camera" [0.743,0.152,0.847,0.169] 0.99
2 [text] "Photos" [0.532,0.156,0.628,0.172] 0.98
3 [text] "Calendar" [0.302,0.158,0.425,0.177] 0.99"""


def test_reads_as_cover_on_every_captured_cover_state() -> None:
    for name, text in (
        ("resting", COVER_RESTING),
        ("woken", COVER_WOKEN),
        ("with a notification stack", COVER_WITH_NOTIFICATION),
    ):
        assert m.reads_as_cover(Screen.read(text)), name


def test_match_screen_reads_the_lock_screen_first_by_shape() -> None:
    # No candidate describes the cover (a real one prints no anchorable
    # text), yet every walk must tell it from "unknown": the matcher
    # answers the OS page itself, so a page's `locked:` hand can fire.
    from physiclaw.conductor.pages import LOCKED_ID

    pp = PagePrint(
        app="app", decl=PageDecl(name="home", anchors=(AnchorDecl(text="Files"),))
    )
    for text in (COVER_RESTING, COVER_WOKEN, COVER_WITH_NOTIFICATION):
        v = m.match_screen(Screen.read(text), [pp])
        assert v.matches(LOCKED_ID), text
    keypad = make_screen(("Enter Passcode", 0.5, 0.5))
    assert m.match_screen(keypad, [pp]).matches(LOCKED_ID)
    assert not m.match_screen(Screen.read(UNLOCKED_APP), [pp]).matches(LOCKED_ID)
    assert not m.match_screen(Screen.read(UNLOCKED_APP), []).matches(LOCKED_ID)


def test_reads_as_cover_rejects_an_ordinary_app_screen() -> None:
    # The status-bar clock is a `<TIME>` row like the cover's; its WIDTH
    # (0.11 vs 0.75) is what separates them.
    assert not m.reads_as_cover(Screen.read(UNLOCKED_APP))


def test_reads_as_cover_needs_a_clock_not_just_a_sparse_screen() -> None:
    assert not m.reads_as_cover(make_screen(("Loading", 0.5, 0.4)))


def test_reads_as_cover_needs_a_bare_clock() -> None:
    # A timestamp EMBEDDED in a longer label (a chat bubble's "20:53
    # delivered") does not normalize to the token alone — and a wide
    # bubble would otherwise clear the width floor.
    row = '0 [text] "20:53 delivered" [0.100,0.400,0.900,0.430] 0.97'
    assert not m.reads_as_cover(Screen.read(f"{LISTING_HEADER}\n{row}"))


def test_reads_as_cover_is_false_on_a_failed_read() -> None:
    # An empty listing is a failed camera read, not a cover — the caller
    # must not spend an unlock on it.
    assert not m.reads_as_cover(Screen.read(""))


def test_reads_as_cover_width_floor_sits_between_the_two_clocks() -> None:
    # Pin the gap the floor lives in: narrower than any hero clock
    # measured (0.686) and wider than any status bar (0.122).
    def clock(width: float) -> Screen:
        row = f'0 [text] "21:45" [0.100,0.100,{0.100 + width:.3f},0.250] 0.98'
        return Screen.read(f"{LISTING_HEADER}\n{row}")

    assert m.reads_as_cover(clock(0.686))  # narrowest hero clock seen
    assert not m.reads_as_cover(clock(0.122))  # widest status bar seen


def test_region_pinned_anchor_ignores_the_scroll_offset() -> None:
    # Chrome does not scroll with the content: the tab-bar anchor is
    # checked where it was learned, while the content anchors vote the
    # offset and move with it.
    pp = _print(
        anchors=[
            AnchorDecl("搜索", region="top"),
            AnchorDecl("综合"),
            AnchorDecl("销量"),
        ],
        learned_anchors=[
            _learned("搜索", 0.5, 0.05),
            _learned("综合", 0.5, 0.5),
            _learned("销量", 0.5, 0.6),
        ],
        scrollable=True,
        threshold=0.9,
    )
    screen = make_screen(("搜索", 0.5, 0.05), ("综合", 0.5, 0.47), ("销量", 0.5, 0.57))

    v = m.match_screen(screen, [pp])

    assert v.kind == "match" and abs(v.dy + 0.03) < 1e-9


def test_forbid_reads_row_labels_not_the_result_prose() -> None:
    # A macro result carries its step log beside the listing; a forbid
    # term in that prose (a step named after it) must not veto the page.
    pp = _print(anchors=[AnchorDecl("综合")], forbid=["popup"])
    text = make_screen(("综合", 0.5, 0.1)).text + "\nmacro demo/x: tool popup ran"

    v = m.match_screen(Screen.read(text), [pp])

    assert v.kind == "match"
