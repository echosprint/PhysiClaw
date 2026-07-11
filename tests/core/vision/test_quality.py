"""Tests for `physiclaw.core.vision.quality` — AF/AE failure detection."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from physiclaw.core.vision import quality
from physiclaw.core.vision.quality import (
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
