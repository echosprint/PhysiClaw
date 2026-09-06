"""Tests for `physiclaw.conductor.drive.capture` — geometry mining,
app-level calibration, and anchor proposal."""

from __future__ import annotations

from conductor_fakes import make_screen

from physiclaw.conductor.drive import capture
from physiclaw.conductor.spec.pages import AnchorDecl, PageDecl

DECL = PageDecl(
    name="results",
    anchors=(AnchorDecl("综合"), AnchorDecl("销量"), AnchorDecl("罕见")),
)


def _observations(n: int = 5):
    # 综合/销量 stable everywhere; 罕见 appears once (below MIN_FREQ);
    # one observation carries an OCR misread of 综合.
    obs = []
    for i in range(n):
        label = "综台" if i == 3 else "综合"
        rows = [(label, 0.2, 0.101 + 0.001 * i), ("销量", 0.4, 0.1, 0.8)]
        if i == 0:
            rows.append(("罕见", 0.6, 0.5))
        obs.append(make_screen(*rows))
    return obs


# ---------- mine_anchors ----------


def test_mine_anchors_positions_variants_and_rare_drop() -> None:
    anchors, warnings = capture.mine_anchors(DECL, _observations())

    a = anchors["综合"]
    assert abs(a.cx - 0.2) < 0.01 and abs(a.cy - 0.103) < 0.01
    assert a.pos_tol >= capture.TOL_FLOOR
    assert "综台" in a.variants
    assert "罕见" not in anchors  # 1/5 < MIN_FREQ
    assert any("罕见" in w for w in warnings)


# ---------- capture_app ----------


def test_capture_app_reports_how_the_whole_rule_fares() -> None:
    obs = _observations()
    negatives = [make_screen(("结算", 0.5, 0.9))]

    learned, reports, warnings = capture.capture_app(
        "taobao", {"results": DECL}, {"results": obs}, negatives
    )

    lp = learned["results"]
    assert lp.observations == 5
    (report,) = reports
    assert report.page == "taobao.results"
    # 罕见 stayed declaration-only (1/5), so the whole rule — every
    # declared anchor — reads only the observation that printed it.
    assert report.genuine_pass == 0.2
    assert report.impostor_pass == 0 and report.separable
    assert any("罕见" in w for w in warnings)


def test_capture_report_flags_a_page_negatives_also_read() -> None:
    decl = PageDecl(name="results", anchors=(AnchorDecl("综合"),))
    obs = [make_screen(("综合", 0.2, 0.1)) for _ in range(3)]
    negatives = [make_screen(("综合", 0.2, 0.1), ("结算", 0.5, 0.9))]

    _, reports, _ = capture.capture_app(
        "taobao", {"results": decl}, {"results": obs}, negatives
    )

    assert reports[0].genuine_pass == 1.0
    assert reports[0].impostor_pass == 1 and not reports[0].separable


def test_capture_app_skips_pages_without_observations() -> None:
    learned, reports, warnings = capture.capture_app(
        "taobao", {"results": DECL}, {}, []
    )

    assert learned == {} and reports == []
    assert any("no observations" in w for w in warnings)


def test_capture_report_impostor_none_without_negatives() -> None:
    _, reports, _ = capture.capture_app(
        "taobao", {"results": DECL}, {"results": _observations()}, []
    )

    assert reports[0].impostor_pass is None
    assert reports[0].separable


# ---------- propose_anchors ----------


def test_propose_anchors_keeps_chrome_shaped_labels() -> None:
    screen = make_screen(
        ("综合", 0.2, 0.1),
        ("¥12.50", 0.4, 0.1),  # no letters — content
        ("20:07", 0.6, 0.1),  # no letters — content
        ("综合", 0.8, 0.1),  # duplicate
        ("这是一条超过十二个字符长度的很长内容", 0.5, 0.5),  # too long
    )

    assert capture.propose_anchors(screen) == ["综合"]


def test_mining_reads_a_two_char_confusion_only_at_the_vouched_spot() -> None:
    # 综台 at the spot 综合 was read exactly is mined as a variant; the same
    # confusion elsewhere on the screen is not the anchor.
    decl = PageDecl(name="results", anchors=(AnchorDecl("综合"),))
    obs = [make_screen(("综合", 0.2, 0.1)) for _ in range(3)]
    obs.append(make_screen(("综台", 0.2, 0.1)))
    obs.append(make_screen(("综台", 0.8, 0.9)))  # far away: not vouched for

    anchors, _ = capture.mine_anchors(decl, obs)

    a = anchors["综合"]
    assert a.variants == ("综台",)
    assert a.freq == 0.8  # 4 of 5 observations read it
