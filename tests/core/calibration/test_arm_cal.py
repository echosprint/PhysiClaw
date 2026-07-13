"""Tests for `physiclaw.core.calibration.arm_cal` — probe triangle,
grid fit, viewport-shift conversion, and the tilt diagnostic
(`_tilt_from_affine`). `calibrate_arm` is driven by patching
`arm_cal._tap_and_read` — the reference the loop actually looks up;
the shared tap helpers themselves are covered in test__common.py.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from physiclaw.core.bridge.calib import CalibrationState
from physiclaw.core.calibration import arm_cal as arm_cal_mod
from physiclaw.core.calibration.arm_cal import _tilt_from_affine
from physiclaw.core.calibration.transforms import ViewportShift

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


# ---------- calibrate_arm ----------


def _make_cal(*, viewport_shift=None) -> MagicMock:
    """A CalibrationState mock with the grid constants surfaced."""
    cal = MagicMock()
    cal.GRID_COLS_PCT = CalibrationState.GRID_COLS_PCT
    cal.GRID_ROWS_PCT = CalibrationState.GRID_ROWS_PCT
    cal.viewport_shift = viewport_shift
    return cal


def test_calibrate_arm_succeeds(mocker) -> None:
    mocker.patch.object(arm_cal_mod.time, "sleep")
    arm = MagicMock()
    cal = _make_cal()

    # 3 probes + 15 grid taps = 18 touches. Use a regular grid for clean
    # affine fit. Probes at screen (0.5, 0.5), (0.6, 0.5), (0.5, 0.6),
    # then 15 grid points scaled the same way.
    probe_touches = [
        {"x": 0.5, "y": 0.5},
        {"x": 0.6, "y": 0.5},
        {"x": 0.5, "y": 0.6},
    ]
    grid_touches = [
        {"x": col, "y": row} for row in cal.GRID_ROWS_PCT for col in cal.GRID_COLS_PCT
    ]

    # Mock _tap_and_read directly to bypass the inner flush_touches loop.
    tap_returns = iter(probe_touches + grid_touches)
    mocker.patch.object(
        arm_cal_mod, "_tap_and_read", side_effect=lambda *a, **kw: next(tap_returns)
    )

    pct_to_grbl, tilt, gtouches = arm_cal_mod.calibrate_arm(arm, cal)

    assert pct_to_grbl.shape == (2, 3)
    assert isinstance(tilt, float)
    assert len(gtouches) == 15
    arm.set_origin.assert_called_once()


def test_calibrate_arm_raises_on_probe_center_miss(mocker) -> None:
    mocker.patch.object(arm_cal_mod.time, "sleep")
    arm = MagicMock()
    cal = _make_cal()
    mocker.patch.object(arm_cal_mod, "_tap_and_read", side_effect=[None])

    with pytest.raises(RuntimeError, match="no touch at center"):
        arm_cal_mod.calibrate_arm(arm, cal)


def test_calibrate_arm_raises_on_probe_x_miss(mocker) -> None:
    mocker.patch.object(arm_cal_mod.time, "sleep")
    arm = MagicMock()
    cal = _make_cal()
    mocker.patch.object(
        arm_cal_mod,
        "_tap_and_read",
        side_effect=[{"x": 0.5, "y": 0.5}, None],
    )

    with pytest.raises(RuntimeError, match="no touch at \\+10mm X"):
        arm_cal_mod.calibrate_arm(arm, cal)


def test_calibrate_arm_raises_on_probe_y_miss(mocker) -> None:
    mocker.patch.object(arm_cal_mod.time, "sleep")
    arm = MagicMock()
    cal = _make_cal()
    mocker.patch.object(
        arm_cal_mod,
        "_tap_and_read",
        side_effect=[
            {"x": 0.5, "y": 0.5},
            {"x": 0.6, "y": 0.5},
            None,
        ],
    )

    with pytest.raises(RuntimeError, match="no touch at \\+10mm Y"):
        arm_cal_mod.calibrate_arm(arm, cal)


def test_calibrate_arm_raises_when_too_few_grid_hits(mocker) -> None:
    mocker.patch.object(arm_cal_mod.time, "sleep")
    arm = MagicMock()
    cal = _make_cal()
    # 3 probes succeed, all 15 grid taps miss (None) — only 3 valid pairs.
    probe = [
        {"x": 0.5, "y": 0.5},
        {"x": 0.6, "y": 0.5},
        {"x": 0.5, "y": 0.6},
    ]
    grid_misses = [None] * 15
    mocker.patch.object(
        arm_cal_mod,
        "_tap_and_read",
        side_effect=probe + grid_misses,
    )

    with pytest.raises(RuntimeError, match="only 3 valid taps"):
        arm_cal_mod.calibrate_arm(arm, cal)


def test_calibrate_arm_uses_viewport_pct_when_shift_set(mocker) -> None:
    mocker.patch.object(arm_cal_mod.time, "sleep")
    arm = MagicMock()
    vshift = ViewportShift(
        offset_x=0,
        offset_y=0,
        dpr=3.0,
        screenshot_width=1170,
        screenshot_height=2532,
    )
    cal = _make_cal(viewport_shift=vshift)
    cal.viewport_pct_to_screenshot_pct.side_effect = lambda c, r: (c, r)

    probe = [
        {"x": 0.5, "y": 0.5},
        {"x": 0.6, "y": 0.5},
        {"x": 0.5, "y": 0.6},
    ]
    grid = [
        {"x": col, "y": row} for row in cal.GRID_ROWS_PCT for col in cal.GRID_COLS_PCT
    ]
    mocker.patch.object(arm_cal_mod, "_tap_and_read", side_effect=probe + grid)

    arm_cal_mod.calibrate_arm(arm, cal)

    # viewport_pct_to_screenshot_pct was used for grid prediction.
    assert cal.viewport_pct_to_screenshot_pct.call_count == 15
