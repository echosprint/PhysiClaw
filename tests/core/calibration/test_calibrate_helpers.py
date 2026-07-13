"""Tests for the three pure helpers in `core/calibration/calibrate.py`.

The orchestration functions in this module touch hardware (StylusArm,
Camera, AssistiveTouch) and live in the integration tier; the three
helpers covered here are coordinate math and image analysis only:

  - `grid_positions(cal)` — generator over the 5×3 grid
  - `_tilt_from_affine(pct_to_grbl)` — arm/phone axis mismatch ratio
  - `_pick_rotation_from_markers(frame)` — camera rotation code from
    blue UP + red RIGHT markers in the frame

Synthetic BGR frames drive the marker tests; the rotation thresholds
in the source compare blob centroid positions, so we render markers
at coordinates that unambiguously satisfy each branch.
"""

from __future__ import annotations

from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from physiclaw.core.calibration.calibrate import (
    _pick_rotation_from_markers,
    _tilt_from_affine,
    grid_positions,
)

# ---------- grid_positions ----------


def test_grid_positions_yields_col_then_row_with_inner_col_outer_row() -> None:
    cal = SimpleNamespace(
        GRID_ROWS_PCT=[0.1, 0.5, 0.9],
        GRID_COLS_PCT=[0.2, 0.5, 0.8],
    )

    out = list(grid_positions(cal))

    # Outer row=0.1: cols sweep first
    assert out[0] == (0.2, 0.1)
    assert out[1] == (0.5, 0.1)
    assert out[2] == (0.8, 0.1)
    # Then row=0.5
    assert out[3] == (0.2, 0.5)


def test_grid_positions_yields_full_cartesian_product() -> None:
    cal = SimpleNamespace(
        GRID_ROWS_PCT=[0.1, 0.5, 0.9],
        GRID_COLS_PCT=[0.2, 0.5, 0.8],
    )

    assert len(list(grid_positions(cal))) == 9


def test_grid_positions_empty_when_either_axis_is_empty() -> None:
    cal = SimpleNamespace(GRID_ROWS_PCT=[], GRID_COLS_PCT=[0.5])

    assert list(grid_positions(cal)) == []


# ---------- _tilt_from_affine ----------


def test_tilt_from_affine_zero_for_aligned_diagonal_matrix() -> None:
    # arm-x maps onto screen-x only — no off-axis component → tilt 0.
    pct_to_grbl = np.array([[100.0, 0.0, 0.0], [0.0, 200.0, 0.0]])

    assert _tilt_from_affine(pct_to_grbl) == 0.0


def test_tilt_from_affine_one_for_perfectly_diagonal_arm_axis() -> None:
    # arm-x rotated 45° relative to screen — equal projection on both
    # screen axes → tilt 1.
    pct_to_grbl = np.array([[1.0, -1.0, 0.0], [1.0, 1.0, 0.0]])

    assert _tilt_from_affine(pct_to_grbl) == pytest.approx(1.0)


def test_tilt_from_affine_one_for_singular_matrix() -> None:
    # rank-1 — np.linalg.inv raises LinAlgError; helper returns 1.0.
    pct_to_grbl = np.array([[1.0, 1.0, 0.0], [2.0, 2.0, 0.0]])

    assert _tilt_from_affine(pct_to_grbl) == 1.0


def test_tilt_from_affine_intermediate_value_for_partial_tilt() -> None:
    # Rotation of arm-x by ~26.6° → tan(26.6°) ≈ 0.5. The minor/major
    # ratio of the inverted basis comes out close to that.
    angle = np.radians(26.565)
    R = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    pct_to_grbl = np.zeros((2, 3))
    pct_to_grbl[:, :2] = R

    tilt = _tilt_from_affine(pct_to_grbl)

    # 26.565° → 0.5 ratio. Allow ±0.05 for floating-point.
    assert tilt == pytest.approx(0.5, abs=0.05)


def test_tilt_from_affine_returns_one_when_major_axis_near_zero(
    mocker,
) -> None:
    # Edge case: A_inv computes successfully but the arm-x basis vector
    # is essentially zero. We synthesize this by patching np.linalg.inv
    # to return a degenerate row.
    fake_inv = np.array([[1e-9, 1.0], [1e-9, 1.0]])
    mocker.patch("numpy.linalg.inv", return_value=fake_inv)

    assert _tilt_from_affine(np.eye(2, 3)) == 1.0


# ---------- _pick_rotation_from_markers ----------


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
