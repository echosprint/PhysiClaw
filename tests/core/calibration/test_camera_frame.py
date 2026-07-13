"""Tests for `physiclaw.core.calibration.camera_frame` — rotation pick.

`_pick_rotation_from_markers` branches on the relative positions of the
blue UP / red RIGHT blob centroids; each branch (and both missing-marker
errors) is driven by scripted blob positions. `calibrate_camera_frame`
is checked for its dead-camera error and the diagnostic dict shape.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import cv2
import numpy as np
import pytest

from physiclaw.core.calibration import camera_frame as camera_frame_mod
from physiclaw.core.calibration.camera_frame import (
    _pick_rotation_from_markers,
    calibrate_camera_frame,
)

# ---------- _pick_rotation_from_markers ----------


def test_pick_rotation_from_markers_no_rotation(mocker) -> None:
    """UP above RIGHT, mostly vertical separation → 0° / no rotation."""
    blob_calls = iter(
        [
            (50.0, 100.0),  # blue UP at top
            (50.0, 200.0),  # red RIGHT at bottom
            None,  # no wrapped red
        ]
    )
    mocker.patch(
        "physiclaw.core.calibration.camera_frame.find_largest_hsv_blob",
        side_effect=lambda *a, **kw: next(blob_calls),
    )

    code, label = _pick_rotation_from_markers(np.zeros((400, 400, 3), np.uint8))

    assert code == -1
    assert "0°" in label


def test_pick_rotation_from_markers_90_clockwise(mocker) -> None:
    """UP at left, RIGHT at right, mostly horizontal → 90°."""
    blob_calls = iter(
        [
            (100.0, 50.0),  # blue UP at left
            (200.0, 50.0),  # red RIGHT at right
            None,
        ]
    )
    mocker.patch(
        "physiclaw.core.calibration.camera_frame.find_largest_hsv_blob",
        side_effect=lambda *a, **kw: next(blob_calls),
    )

    code, label = _pick_rotation_from_markers(np.zeros((400, 400, 3), np.uint8))

    assert code == cv2.ROTATE_90_CLOCKWISE


def test_pick_rotation_from_markers_180(mocker) -> None:
    """UP below RIGHT, mostly vertical → 180°."""
    blob_calls = iter(
        [
            (50.0, 200.0),  # blue UP at bottom
            (50.0, 100.0),  # red RIGHT at top
            None,
        ]
    )
    mocker.patch(
        "physiclaw.core.calibration.camera_frame.find_largest_hsv_blob",
        side_effect=lambda *a, **kw: next(blob_calls),
    )

    code, _ = _pick_rotation_from_markers(np.zeros((400, 400, 3), np.uint8))

    assert code == cv2.ROTATE_180


def test_pick_rotation_from_markers_90_counterclockwise(mocker) -> None:
    """Default fallthrough — UP right of RIGHT."""
    blob_calls = iter(
        [
            (200.0, 50.0),  # blue UP at right
            (100.0, 50.0),  # red RIGHT at left
            None,
        ]
    )
    mocker.patch(
        "physiclaw.core.calibration.camera_frame.find_largest_hsv_blob",
        side_effect=lambda *a, **kw: next(blob_calls),
    )

    code, _ = _pick_rotation_from_markers(np.zeros((400, 400, 3), np.uint8))

    assert code == cv2.ROTATE_90_COUNTERCLOCKWISE


def test_pick_rotation_from_markers_raises_when_red_missing(mocker) -> None:
    """Red is one search over both hue ends (via red_ranges); if it finds
    nothing, the step fails with a clear message."""
    blob_calls = iter(
        [
            (50.0, 100.0),  # blue UP
            None,  # red: nothing at either hue end
        ]
    )
    mocker.patch(
        "physiclaw.core.calibration.camera_frame.find_largest_hsv_blob",
        side_effect=lambda *a, **kw: next(blob_calls),
    )

    with pytest.raises(RuntimeError, match=r"^RIGHT \(red\) marker not found$"):
        _pick_rotation_from_markers(np.zeros((400, 400, 3), np.uint8))


def test_pick_rotation_from_markers_raises_when_marker_missing(mocker) -> None:
    """First _find_marker call returns None → RuntimeError."""
    mocker.patch(
        "physiclaw.core.calibration.camera_frame.find_largest_hsv_blob",
        return_value=None,
    )

    with pytest.raises(RuntimeError, match="UP \\(blue\\) marker not found"):
        _pick_rotation_from_markers(np.zeros((400, 400, 3), np.uint8))


# ---------- calibrate_camera_frame ----------


def test_calibrate_camera_frame_raises_when_camera_dead(mocker) -> None:
    mocker.patch.object(camera_frame_mod.time, "sleep")
    cam = MagicMock()
    cam.raw_frame.return_value = None
    cal = MagicMock()

    with pytest.raises(RuntimeError, match="Camera read failed"):
        calibrate_camera_frame(cam, cal)


def test_calibrate_camera_frame_returns_diagnostic_dict(mocker) -> None:
    mocker.patch.object(camera_frame_mod.time, "sleep")
    cam = MagicMock()
    cam.raw_frame.return_value = np.zeros((480, 640, 3), np.uint8)
    cal = MagicMock()
    mocker.patch.object(
        camera_frame_mod,
        "check_phone_in_frame",
        return_value={
            "ok": True,
            "issues": [],
            "coverage": 0.9,
            "aspect_ratio": 16 / 9,
            "image_size": (480, 640),
            "phone_region": (0, 0, 100, 100),
        },
    )
    mocker.patch.object(
        camera_frame_mod,
        "_pick_rotation_from_markers",
        return_value=(cv2.ROTATE_90_CLOCKWISE, "90° clockwise"),
    )

    out = calibrate_camera_frame(cam, cal)

    assert out["rotation"] == cv2.ROTATE_90_CLOCKWISE
    assert out["rotation_name"] == "90° clockwise"
    assert out["setup_ok"] is True
    assert out["coverage"] == 0.9
    cal.set_phase.assert_called_once_with("markers")


# ---------- _pick_rotation_from_markers — synthetic frames ----------
#
# Real HSV blob detection on rendered frames (no mocks) — complements
# the mock-driven branch tests above; includes the high-hue-red pin.


def _draw_marker(
    frame: np.ndarray, cx: int, cy: int, color_bgr: tuple, size: int = 30
) -> None:
    """Draw a saturated swatch at (cx, cy). Size 30 → area 900 ≥ min_area=500."""
    s = size // 2
    frame[cy - s : cy + s, cx - s : cx + s] = color_bgr


def _frame_with_up_and_right(
    up_xy: tuple[int, int], right_xy: tuple[int, int], shape=(600, 800, 3)
) -> np.ndarray:
    img = np.zeros(shape, dtype=np.uint8)
    _draw_marker(img, *up_xy, color_bgr=(255, 100, 0))  # blue (BGR)
    _draw_marker(img, *right_xy, color_bgr=(0, 50, 255))  # red (BGR)
    return img


def test_pick_rotation_no_rotation_when_up_is_above_right_and_aligned_x() -> None:
    # UP at (400, 100), RIGHT at (500, 300) — up_y < right_y and the
    # horizontal gap (100) is less than the vertical gap (200).
    img = _frame_with_up_and_right((400, 100), (500, 300))

    code, label = _pick_rotation_from_markers(img)

    assert code == -1
    assert label == "0° — no rotation needed"


def test_pick_rotation_90_clockwise_when_up_is_left_of_right() -> None:
    # UP at (100, 300), RIGHT at (300, 320) — up_x < right_x and
    # vertical gap (20) < horizontal gap (200).
    img = _frame_with_up_and_right((100, 300), (300, 320))

    code, label = _pick_rotation_from_markers(img)

    assert code == cv2.ROTATE_90_CLOCKWISE
    assert label == "90° clockwise"


def test_pick_rotation_180_when_up_is_below_right() -> None:
    # UP at (400, 500), RIGHT at (300, 100) — up_y > right_y, horizontal
    # gap (100) < vertical gap (400).
    img = _frame_with_up_and_right((400, 500), (300, 100))

    code, label = _pick_rotation_from_markers(img)

    assert code == cv2.ROTATE_180
    assert label == "180°"


def test_pick_rotation_90_counterclockwise_for_remaining_orientation() -> None:
    # UP at (700, 300), RIGHT at (300, 290) — falls through to the
    # default branch (none of the three earlier predicates match).
    img = _frame_with_up_and_right((700, 300), (300, 290))

    code, label = _pick_rotation_from_markers(img)

    assert code == cv2.ROTATE_90_COUNTERCLOCKWISE
    assert label == "90° counter-clockwise"


def test_pick_rotation_raises_when_blue_up_marker_missing() -> None:
    img = np.zeros((600, 800, 3), dtype=np.uint8)
    _draw_marker(img, 400, 300, (0, 50, 255))  # only red

    with pytest.raises(RuntimeError, match=r"^UP \(blue\) marker not found$"):
        _pick_rotation_from_markers(img)


def test_pick_rotation_raises_when_red_right_marker_missing() -> None:
    img = np.zeros((600, 800, 3), dtype=np.uint8)
    _draw_marker(img, 400, 100, (255, 100, 0))  # only blue

    with pytest.raises(RuntimeError, match=r"^RIGHT \(red\) marker not found$"):
        _pick_rotation_from_markers(img)


def test_pick_rotation_detects_red_at_high_hue_end() -> None:
    # Regression: the camera commonly renders the on-screen red near the
    # high hue end (H≈175), which wraps past 180 — almost nothing lands in
    # the low [0,10] range. Detection must check BOTH ends; the old code
    # checked the low range first and raised before the high-range fallback,
    # reporting a clearly-visible red marker as "not found".
    red_hi = tuple(
        int(c)
        for c in cv2.cvtColor(np.uint8([[[175, 200, 200]]]), cv2.COLOR_HSV2BGR)[0, 0]
    )
    img = np.zeros((600, 800, 3), dtype=np.uint8)
    _draw_marker(img, 400, 100, (255, 100, 0))  # blue UP
    _draw_marker(img, 500, 300, red_hi)  # red RIGHT, high-hue end

    code, label = _pick_rotation_from_markers(img)

    # up above right, |Δx| < |Δy| → no rotation (same geometry as the
    # low-hue no-rotation case, proving the high-hue red was detected).
    assert code == -1
    assert label == "0° — no rotation needed"
