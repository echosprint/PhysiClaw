"""Tests for `physiclaw.core.calibration.validate` — full-chain checks.

`validate_calibration` runs with identity-ish affines and a mocked
`_detect_orange_dot`, exercising the pass path, the undetected-dot and
off-screen-back-projection fallbacks (which must degrade to the known
position, never steer the arm off the panel), the tap-retry cycle, and
the no-touch failure record. `trace_screen_edge` is pinned to its 9
edge visits.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np

from physiclaw.core.calibration import validate as validate_mod
from tests.core.calibration.conftest import make_cal as _make_cal


def _identity_pct_to_grbl() -> np.ndarray:
    return np.array(
        [
            [10.0, 0.0, 0.0],
            [0.0, 20.0, 0.0],
        ]
    )


def _identity_pct_to_cam() -> np.ndarray:
    return np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
    )


# ---------- validate_calibration ----------


def test_validate_calibration_records_passed_when_touches_match(mocker) -> None:
    mocker.patch.object(validate_mod.time, "sleep")
    mocker.patch.object(validate_mod.random, "random", return_value=0.5)
    arm = MagicMock()
    cam = MagicMock()
    cam.raw_frame.return_value = np.zeros((100, 100, 3), np.uint8)
    cal = _make_cal()

    # detect_orange_dot returns the dot at the same pct (camera 0-1 = screen 0-1).
    mocker.patch.object(
        validate_mod,
        "_detect_orange_dot",
        side_effect=lambda f, **kw: (50, 50),  # at center of 100×100 → 0.5 pct
    )
    # Touches always come back at the expected position.
    cal.flush_touches.side_effect = [[], [{"x": 0.5, "y": 0.5}]] * 5

    results = validate_mod.validate_calibration(
        arm,
        cam,
        cal,
        rotation=-1,
        pct_to_grbl=_identity_pct_to_grbl(),
        pct_to_cam=_identity_pct_to_cam(),
        cam_size=(100, 100),
        num_tests=2,
    )

    assert len(results) == 2
    assert all(r["passed"] for r in results)


def test_validate_calibration_falls_back_when_dot_undetected(mocker) -> None:
    mocker.patch.object(validate_mod.time, "sleep")
    mocker.patch.object(validate_mod.random, "random", return_value=0.5)
    arm = MagicMock()
    cam = MagicMock()
    cam.raw_frame.return_value = np.zeros((100, 100, 3), np.uint8)
    cal = _make_cal()
    mocker.patch.object(validate_mod, "_detect_orange_dot", return_value=None)
    cal.flush_touches.side_effect = [[], [{"x": 0.5, "y": 0.5}]] * 1

    results = validate_mod.validate_calibration(
        arm,
        cam,
        cal,
        rotation=-1,
        pct_to_grbl=_identity_pct_to_grbl(),
        pct_to_cam=_identity_pct_to_cam(),
        cam_size=(100, 100),
        num_tests=1,
    )

    assert len(results) == 1
    # Fallback used expected position → should pass.
    assert results[0]["passed"] is True


def test_validate_calibration_records_failure_on_no_touch(mocker) -> None:
    mocker.patch.object(validate_mod.time, "sleep")
    mocker.patch.object(validate_mod.random, "random", return_value=0.5)
    arm = MagicMock()
    cam = MagicMock()
    cam.raw_frame.return_value = np.zeros((100, 100, 3), np.uint8)
    cal = _make_cal()
    mocker.patch.object(
        validate_mod,
        "_detect_orange_dot",
        return_value=(50, 50),
    )
    # All flushes return empty → tap retries 4 times, each fails.
    cal.flush_touches.return_value = []

    results = validate_mod.validate_calibration(
        arm,
        cam,
        cal,
        rotation=-1,
        pct_to_grbl=_identity_pct_to_grbl(),
        pct_to_cam=_identity_pct_to_cam(),
        cam_size=(100, 100),
        num_tests=1,
    )

    assert len(results) == 1
    assert results[0]["passed"] is False
    assert results[0]["error"] == 999.0


def test_validate_calibration_refires_on_miss(mocker) -> None:
    mocker.patch.object(validate_mod.time, "sleep")
    mocker.patch.object(validate_mod.random, "random", return_value=0.5)
    arm = MagicMock()
    cam = MagicMock()
    cam.raw_frame.return_value = np.zeros((100, 100, 3), np.uint8)
    cal = _make_cal()
    mocker.patch.object(validate_mod, "_detect_orange_dot", return_value=(50, 50))

    # First two flushes (clear + miss) then clear + hit on retry.
    cal.flush_touches.side_effect = [
        [],
        [],  # attempt 0: clear + miss
        [],
        [{"x": 0.5, "y": 0.5}],  # attempt 1: clear + hit
    ]

    results = validate_mod.validate_calibration(
        arm,
        cam,
        cal,
        rotation=-1,
        pct_to_grbl=_identity_pct_to_grbl(),
        pct_to_cam=_identity_pct_to_cam(),
        cam_size=(100, 100),
        num_tests=1,
    )

    assert results[0]["passed"] is True


def test_validate_calibration_rejects_off_screen_backprojection(mocker, caplog) -> None:
    # A detected blob that back-projects past the panel edge must not steer the
    # arm off-screen — the guard falls back to the known position instead.
    mocker.patch.object(validate_mod.time, "sleep")
    mocker.patch.object(validate_mod.random, "random", return_value=0.5)
    arm = MagicMock()
    cam = MagicMock()
    cam.raw_frame.return_value = np.zeros((100, 100, 3), np.uint8)
    cal = _make_cal()
    # (200, 200) on a 100×100 frame → camera 0-1 (2, 2) → back-projects off-screen.
    mocker.patch.object(validate_mod, "_detect_orange_dot", return_value=(200, 200))
    cal.flush_touches.side_effect = [[], [{"x": 0.5, "y": 0.5}]]

    with caplog.at_level("WARNING"):
        results = validate_mod.validate_calibration(
            arm,
            cam,
            cal,
            rotation=-1,
            pct_to_grbl=_identity_pct_to_grbl(),
            pct_to_cam=_identity_pct_to_cam(),
            cam_size=(100, 100),
            num_tests=1,
        )

    assert results[0]["passed"] is True  # fell back to the known position
    assert any("off-screen" in r.message for r in caplog.records)


# ---------- trace_screen_edge ----------


def test_trace_screen_edge_visits_all_check_points(mocker) -> None:
    mocker.patch.object(validate_mod.time, "sleep")
    arm = MagicMock()
    transforms = MagicMock()
    transforms.pct_to_grbl_mm.side_effect = lambda x, y: (x * 100, y * 100)

    validate_mod.trace_screen_edge(arm, transforms)

    # 9 check points + 2 return_to_origin → 9 fast moves + 2 origin returns.
    assert arm.rapid_to.call_count == 9
    assert arm.return_to_origin.call_count == 2
