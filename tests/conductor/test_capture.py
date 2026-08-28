"""Tests for `physiclaw.conductor.capture` — geometry mining,
app-level calibration, and anchor proposal."""

from __future__ import annotations

from conductor_fakes import make_screen

from physiclaw.conductor import capture
from physiclaw.conductor.match import normalize
from physiclaw.conductor.pages import AnchorDecl, PageDecl

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


def test_mine_anchors_weight_divides_by_document_frequency() -> None:
    df = {normalize("综合"): 3, normalize("销量"): 1}

    anchors, _ = capture.mine_anchors(DECL, _observations(), app_df=df)

    assert anchors["综合"].weight < anchors["销量"].weight


# ---------- capture_app ----------


def test_capture_app_calibrates_threshold_and_reports_separation() -> None:
    obs = _observations()
    negatives = [make_screen(("结算", 0.5, 0.9))]

    learned, reports, warnings = capture.capture_app(
        "taobao", {"results": DECL}, {"results": obs}, negatives
    )

    lp = learned["results"]
    assert capture.THRESHOLD_LO <= lp.threshold <= capture.THRESHOLD_HI
    assert lp.observations == 5
    (report,) = reports
    assert report.page == "taobao.results"
    assert report.genuine_min >= lp.threshold  # safety factor below min
    assert report.separable
    assert any("罕见" in w for w in warnings)


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

    assert reports[0].impostor_max is None
    assert reports[0].separable


def test_app_document_frequency_counts_pages_per_anchor() -> None:
    decls = {
        "a": PageDecl(name="a", anchors=(AnchorDecl("共享"), AnchorDecl("甲"))),
        "b": PageDecl(name="b", anchors=(AnchorDecl("共享"),)),
    }

    df = capture.app_document_frequency(decls)

    assert df[normalize("共享")] == 2
    assert df[normalize("甲")] == 1


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
