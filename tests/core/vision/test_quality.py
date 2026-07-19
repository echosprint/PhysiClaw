"""Tests for `physiclaw.core.vision.quality` — AF/AE failure detection."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from physiclaw.core.vision import quality
from physiclaw.core.vision.quality import (
    BLUR_NOTE,
    DARK_MEDIAN_LUMA,
    DARK_NOTE,
    DARK_P99_LUMA,
    NORMALIZED_WIDTH,
    QualityMonitor,
    QualityReport,
    assess,
    laplacian_variance,
)


def _checkerboard(
    h: int = 200, w: int = 100, lo: int = 0, hi: int = 120, cell: int = 2
) -> np.ndarray:
    """Sharp synthetic frame — a fine checkerboard has huge Laplacian variance."""
    yy, xx = np.indices((h, w))
    gray = np.where(((yy // cell) + (xx // cell)) % 2 == 0, hi, lo).astype(np.uint8)
    return np.stack([gray] * 3, axis=-1)


def _flat(h: int = 200, w: int = 100, value: int = 128) -> np.ndarray:
    return np.full((h, w, 3), value, dtype=np.uint8)


# ---------- laplacian_variance ----------


def test_laplacian_variance_for_uniform_image_is_zero() -> None:
    img = _flat()

    assert laplacian_variance(img) == 0.0


def test_laplacian_variance_for_image_with_strong_edges_is_higher() -> None:
    assert laplacian_variance(_checkerboard()) > 100.0


# ---------- assess ----------


def test_sharp_normal_frame_is_ok() -> None:
    r = assess(_checkerboard())
    assert not r.blurry
    assert not r.blown


def test_flat_frame_is_blurry() -> None:
    r = assess(_flat())
    assert r.sharpness == 0.0
    assert r.blurry


def test_white_patches_on_dark_screen_are_blown() -> None:
    # Blown-out dock/icons on a dark home screen: sharp texture, dark
    # median, but a big clipped-white region.
    frame = _checkerboard()
    frame[:50, :, :] = 255  # top 25% burned to white
    r = assess(frame)
    assert r.clip_pct == pytest.approx(0.25, abs=0.01)
    assert r.median_luma < quality.BLOWN_MEDIAN_LUMA
    assert r.blown


def test_legit_white_page_is_not_blown() -> None:
    # A well-exposed white page (chat, order form) clips its background
    # legitimately — median is white, so the two-factor rule passes it.
    frame = _flat(value=255)
    frame[80:120, :, :] = _checkerboard(40, 100)[:, :, :]  # dark text band
    r = assess(frame)
    assert r.clip_pct > quality.BLOWN_CLIP_PCT
    assert not r.blown


def test_washed_white_ui_below_strict_clip_is_blown() -> None:
    # The 2026-07 miss: a white UI (WeChat chat) overexposed so the
    # background sits at ~240 — in the detail-destroyed highlight band
    # (>=225) but UNDER the strict 250 clip, and featureless. The old
    # median>=250 / clip>=250 clauses all slipped it; the highlight-
    # fraction clause catches it.
    frame = _flat(value=240)
    r = assess(frame)
    assert r.clip_pct == 0.0  # nothing at >=250, so the clip clauses are blind
    assert r.highlight_pct >= quality.WASHED_HIGHLIGHT_PCT
    assert r.sharpness < quality.BLUR_THRESHOLD
    assert r.blown


def test_hot_but_readable_white_page_is_not_blown() -> None:
    # Same highlight band, but a readable page: its text carries edges,
    # so sharpness clears the blur floor and the washed clause holds off —
    # the sharpness guard, not a luma cutoff, is what distinguishes them.
    frame = _flat(value=240)
    frame[80:120, :, :] = _checkerboard(40, 100)[:, :, :]  # dark text band
    r = assess(frame)
    assert r.highlight_pct >= quality.WASHED_HIGHLIGHT_PCT
    assert r.sharpness >= quality.BLUR_THRESHOLD
    assert not r.blown


def _burned_icon_grid(background: int) -> np.ndarray:
    """3x4 grid of solid clipped squares — burned app icons — on the
    given background. Icon area fraction ~2% (inside the blob band)."""
    frame = np.full((400, 200, 3), background, dtype=np.uint8)
    for row in range(3):
        for col in range(4):
            y, x = 40 + row * 100, 10 + col * 48
            frame[y : y + 40, x : x + 40] = 255
    return frame


def test_icon_grid_white_out_blown_despite_white_median() -> None:
    # The failure the histogram rule cannot see (2026-07 sessions):
    # burned icons on a bright page — median >= 200 reads as a legit
    # white page, but the blob signature catches the grid.
    r = assess(_burned_icon_grid(background=210))
    assert r.median_luma >= quality.BLOWN_MEDIAN_LUMA  # histogram rule blind
    assert r.white_blobs >= quality.BLOWN_BLOB_COUNT
    assert r.blown


def test_icon_grid_white_out_blown_on_dark_background_too() -> None:
    r = assess(_burned_icon_grid(background=30))
    assert r.white_blobs >= quality.BLOWN_BLOB_COUNT
    assert r.blown


def test_icon_blob_filter_rejects_bars_and_page_sized_regions() -> None:
    # A clipped long bar (wrong aspect) and a clipped half-page (above
    # the area band) are not icon blobs — and with a white median the
    # frame stays un-blown, mirroring a legit bright page.
    frame = np.full((400, 200, 3), 30, dtype=np.uint8)
    frame[50:70, 10:190] = 255  # bar: aspect 9
    frame[200:400, :] = 255  # half page: area fraction 0.5
    r = assess(frame)
    assert r.white_blobs == 0
    assert not r.blown


def test_legit_white_page_has_no_icon_blobs() -> None:
    frame = _flat(value=255)
    frame[80:120, :, :] = _checkerboard(40, 100)[:, :, :]
    assert assess(frame).white_blobs < quality.BLOWN_BLOB_COUNT


def test_assess_matches_report_fields() -> None:
    frame = _flat(value=30)
    r = assess(frame)
    assert r.median_luma == 30.0
    assert r.clip_pct == 0.0


def test_sharpness_score_survives_a_higher_resolution_camera() -> None:
    # The same physical screen captured at 2× sensor resolution (features
    # twice as many pixels wide) must score in the same regime — that's
    # what normalizing to NORMALIZED_WIDTH is for. Without it the fixed
    # BLUR_THRESHOLD calibrated on one rig wouldn't transfer to another.
    base = _checkerboard(960, NORMALIZED_WIDTH, cell=2)
    double = _checkerboard(1920, NORMALIZED_WIDTH * 2, cell=4)
    r_base, r_double = assess(base), assess(double)
    assert not r_base.blurry and not r_double.blurry
    assert r_double.sharpness == pytest.approx(r_base.sharpness, rel=0.25)


def test_blurry_high_resolution_frame_still_flagged() -> None:
    big = _checkerboard(1920, NORMALIZED_WIDTH * 2, cell=4)
    blurred = cv2.GaussianBlur(big, (31, 31), 10)
    assert assess(blurred).blurry


def test_narrow_frame_is_not_upscaled_into_false_blur() -> None:
    # Crops below NORMALIZED_WIDTH are scored as-is: interpolating them up
    # would smooth genuinely sharp pixels into a false blur verdict.
    r = assess(_checkerboard(200, 100, cell=2))
    assert not r.blurry


def test_exposure_stats_stable_under_normalization() -> None:
    # The blown-highlights verdict must not shift with camera resolution.
    frame = _checkerboard(1920, NORMALIZED_WIDTH * 2, cell=4)
    frame[:480, :, :] = 255  # top 25% burned to white
    r = assess(frame)
    assert r.clip_pct == pytest.approx(0.25, abs=0.02)
    assert r.blown


# ---------- QualityMonitor ----------


def _bad() -> QualityReport:
    return QualityReport(sharpness=10.0, clip_pct=0.0, median_luma=100.0)


def _blown() -> QualityReport:
    return QualityReport(sharpness=500.0, clip_pct=0.3, median_luma=100.0)


def _good() -> QualityReport:
    return QualityReport(sharpness=500.0, clip_pct=0.0, median_luma=100.0)


def test_good_view_returns_none() -> None:
    assert QualityMonitor().observe(_good()) is None


def test_blurry_view_warns_with_blur_note() -> None:
    line = QualityMonitor().observe(_bad())
    assert line.startswith("⚠ camera:")
    assert quality.BLUR_NOTE in line
    assert quality.UNRELIABLE_NOTE in line


def test_blown_view_warns_with_clip_percentage() -> None:
    line = QualityMonitor().observe(_blown())
    assert quality.BLOWN_NOTE in line
    assert "30%" in line


def test_both_issues_join_in_one_line() -> None:
    r = QualityReport(sharpness=10.0, clip_pct=0.3, median_luma=100.0)
    line = QualityMonitor().observe(r)
    assert quality.BLUR_NOTE in line
    assert quality.BLOWN_NOTE in line


def test_transient_bad_view_does_not_escalate() -> None:
    m = QualityMonitor()
    line = m.observe(_bad())
    assert "report it to your user" not in line


def test_persistent_bad_views_escalate_to_user_reminder() -> None:
    m = QualityMonitor()
    lines = [m.observe(_bad()) for _ in range(quality.PERSIST_AFTER)]
    assert "report it to your user" not in lines[-2]
    assert "report it to your user" in lines[-1]
    assert f"{quality.PERSIST_AFTER} in a row" in lines[-1]


def test_good_view_resets_the_streak() -> None:
    m = QualityMonitor()
    for _ in range(quality.PERSIST_AFTER):
        m.observe(_bad())
    assert m.observe(_good()) is None
    # Streak restarted — the next bad view warns softly again.
    assert "report it to your user" not in m.observe(_bad())


def test_streak_property_mirrors_consecutive_bad_views() -> None:
    # Read by the orchestration layer's re-tune policy.
    m = QualityMonitor()
    assert m.streak == 0
    m.observe(_blown())
    m.observe(_blown())
    assert m.streak == 2
    m.observe(_good())
    assert m.streak == 0


# ---------- dark: the two-axis underexposure predicate ----------


def test_dark_requires_missing_highlights_not_just_low_median() -> None:
    # Correctly-exposed dark-mode UI (black background, white text)
    # meters a low median but carries near-white text — it is NOT
    # underexposed; calling it dark would false-alarm the rig warning
    # streak and fire tunes on every dark-themed app.
    dark_ui = QualityReport(
        sharpness=400.0, clip_pct=0.01, median_luma=7.0, p99_luma=250.0
    )
    crushed = QualityReport(
        sharpness=20.0, clip_pct=0.0, median_luma=7.0, p99_luma=40.0
    )

    assert dark_ui.dark is False
    assert crushed.dark is True


def test_assess_measures_highlights_for_the_dark_axis() -> None:
    # Black frame with a solid bright block (dark-mode text stand-in):
    # p99 must clear the highlight floor so `dark` stays False.
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    frame[:20, :20] = 230

    report = assess(frame)

    assert report.median_luma < DARK_MEDIAN_LUMA
    assert report.p99_luma >= DARK_P99_LUMA
    assert report.dark is False


def test_saturated_frame_is_blown_despite_white_median() -> None:
    # A manual value held from a far dimmer scene nukes the whole frame:
    # median >= 250 evades the two-factor rule and a page-sized clipped
    # region is not icon-shaped — the saturation clause must catch it,
    # or nothing ever corrects the exposure.
    nuked = QualityReport(
        sharpness=1.0, clip_pct=0.99, median_luma=254.0, p99_luma=255.0
    )

    assert nuked.blown is True


def test_monitor_reports_dark_instead_of_blur() -> None:
    # A crushed frame always meters blurry too — naming the sharpness
    # would steer the agent at the camera when the problem is light.
    monitor = QualityMonitor()
    crushed = QualityReport(sharpness=20.0, clip_pct=0.0, median_luma=7.0)

    line = monitor.observe(crushed)

    assert line is not None and DARK_NOTE in line
    assert BLUR_NOTE not in line


def test_uniform_saturated_frame_is_blown() -> None:
    # The morning white-out: a manual value held from a dim night scene
    # nukes the lit screen featureless — white median, heavy clip, no
    # edges. Median >= 250 evades the two-factor rule and a page-sized
    # region is not icon-shaped; the saturation clause must catch it.
    frame = np.full((200, 100, 3), 255, dtype=np.uint8)

    r = assess(frame)

    assert r.blown


def test_glint_does_not_defeat_the_dark_axis() -> None:
    # A ~1% specular glint at 255 would pin a raw p99 and mechanically
    # disarm `dark` on a genuinely crushed frame — the highlight axis
    # excludes clipped pixels.
    frame = np.full((200, 100, 3), 5, dtype=np.uint8)
    frame[:4, :50] = 255  # glint: 1% of the crop

    r = assess(frame)

    assert r.dark
