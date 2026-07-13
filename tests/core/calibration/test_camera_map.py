"""Tests for `physiclaw.core.calibration.camera_map` — Mapping B.

`compute_camera_mapping` is driven with synthetic frames: red dots
rendered at the canonical grid positions (optionally pre-rotated, with
off-screen strays, or with a displaced dot) exercise the happy path,
the rotation application, the RGBM corner fence, and both fit-gate
failure modes — the real detection and fitting code runs throughout.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import cv2
import numpy as np
import pytest

from physiclaw.core.bridge.calib import CalibrationState
from physiclaw.core.calibration import camera_map as camera_map_mod
from physiclaw.core.calibration.transforms import ViewportShift


def _make_cal(*, viewport_shift=None) -> MagicMock:
    """A CalibrationState mock with the grid constants surfaced."""
    cal = MagicMock()
    cal.GRID_COLS_PCT = CalibrationState.GRID_COLS_PCT
    cal.GRID_ROWS_PCT = CalibrationState.GRID_ROWS_PCT
    cal.viewport_shift = viewport_shift
    return cal


def _grid_dot_image(rows=5, cols=3, w=600, h=900) -> np.ndarray:
    """Build a frame with red dots at grid_positions × frame size."""
    img = np.zeros((h, w, 3), dtype=np.uint8)
    for row in CalibrationState.GRID_ROWS_PCT:
        for col in CalibrationState.GRID_COLS_PCT:
            cv2.circle(
                img,
                (int(col * w), int(row * h)),
                radius=10,
                color=(0, 0, 255),
                thickness=-1,
            )
    return img


def test_compute_camera_mapping_succeeds(mocker) -> None:
    mocker.patch.object(camera_map_mod.time, "sleep")
    cam = MagicMock()
    cam.raw_frame.return_value = _grid_dot_image()
    cal = _make_cal()

    pct_to_cam, cam_size = camera_map_mod.compute_camera_mapping(cam, cal, rotation=-1)

    assert pct_to_cam.shape == (2, 3)
    assert cam_size == (600, 900)


def test_compute_camera_mapping_camera_dead(mocker) -> None:
    mocker.patch.object(camera_map_mod.time, "sleep")
    cam = MagicMock()
    cam.raw_frame.return_value = None
    cal = _make_cal()

    with pytest.raises(RuntimeError, match="camera read failed"):
        camera_map_mod.compute_camera_mapping(cam, cal, rotation=-1)


def test_compute_camera_mapping_retries_then_fails(mocker) -> None:
    mocker.patch.object(camera_map_mod.time, "sleep")
    cam = MagicMock()
    blank = np.zeros((600, 900, 3), dtype=np.uint8)
    cam.raw_frame.return_value = blank
    cal = _make_cal()

    with pytest.raises(RuntimeError, match=r"last saw 0 dots inside the screen"):
        camera_map_mod.compute_camera_mapping(cam, cal, rotation=-1)


def test_compute_camera_mapping_applies_rotation(mocker) -> None:
    mocker.patch.object(camera_map_mod.time, "sleep")
    cam = MagicMock()
    # The camera returns a frame that is canonical only AFTER the configured
    # rotation is applied — so pre-rotate the canonical grid the opposite way.
    # This is the real scenario: rotation is picked to upright the grid, and
    # only then does sort_dots_to_grid (and the fit gate) line up.
    canonical = _grid_dot_image()
    cam.raw_frame.return_value = cv2.rotate(canonical, cv2.ROTATE_90_COUNTERCLOCKWISE)
    cal = _make_cal()

    rotate_spy = mocker.spy(camera_map_mod.cv2, "rotate")

    pct_to_cam, _ = camera_map_mod.compute_camera_mapping(
        cam, cal, rotation=cv2.ROTATE_90_CLOCKWISE
    )

    assert rotate_spy.called
    assert pct_to_cam.shape == (2, 3)


def _draw_corner_cluster(frame: np.ndarray, cx: int, cy: int, d: int = 20) -> None:
    """Draw a 2×2 RGBM corner cluster centered at (cx, cy)."""
    cv2.circle(frame, (cx - d, cy - d), 8, (0, 0, 255), -1)  # R
    cv2.circle(frame, (cx + d, cy - d), 8, (0, 255, 0), -1)  # G
    cv2.circle(frame, (cx + d, cy + d), 8, (255, 0, 0), -1)  # B
    cv2.circle(frame, (cx - d, cy + d), 8, (255, 0, 255), -1)  # M (magenta)


def test_compute_camera_mapping_masks_off_screen_reflection(mocker) -> None:
    # The corner frame bounds the screen; the grid frame carries the 15 dots
    # plus a stray red reflection outside that boundary. The mask drops the
    # reflection so the fit still sees a clean grid and succeeds.
    mocker.patch.object(camera_map_mod.time, "sleep")
    w, h = 600, 900
    corners_frame = np.zeros((h, w, 3), dtype=np.uint8)
    for cx, cy in [(60, 90), (540, 90), (540, 810), (60, 810)]:
        _draw_corner_cluster(corners_frame, cx, cy)
    grid_frame = _grid_dot_image()
    cv2.circle(grid_frame, (10, 450), 10, (0, 0, 255), -1)  # off-screen stray

    cam = MagicMock()
    cam.raw_frame.side_effect = [corners_frame] + [grid_frame] * 5
    cal = _make_cal()

    pct_to_cam, cam_size = camera_map_mod.compute_camera_mapping(cam, cal, rotation=-1)

    assert pct_to_cam.shape == (2, 3)
    assert cam_size == (600, 900)


def test_compute_camera_mapping_rejects_mis_corresponded_grid(mocker) -> None:
    # 15 dots are present but one is displaced into another row, so the fit can
    # never line up. The quality gate must reject it rather than return a
    # silently-wrong mapping.
    mocker.patch.object(camera_map_mod.time, "sleep")
    w, h = 600, 900
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    positions = [
        (int(col * w), int(row * h))
        for row in CalibrationState.GRID_ROWS_PCT
        for col in CalibrationState.GRID_COLS_PCT
    ]
    positions[0] = (300, 850)  # displaced far from its grid slot
    for x, y in positions:
        cv2.circle(frame, (x, y), 10, (0, 0, 255), -1)
    cam = MagicMock()
    cam.raw_frame.return_value = frame
    cal = _make_cal()

    with pytest.raises(RuntimeError, match=r"clean 15-dot grid passing the fit gate"):
        camera_map_mod.compute_camera_mapping(cam, cal, rotation=-1)


def test_compute_camera_mapping_uses_viewport_shift(mocker) -> None:
    mocker.patch.object(camera_map_mod.time, "sleep")
    cam = MagicMock()
    cam.raw_frame.return_value = _grid_dot_image()
    vshift = ViewportShift(
        offset_x=0,
        offset_y=0,
        dpr=3.0,
        screenshot_width=1170,
        screenshot_height=2532,
    )
    cal = _make_cal(viewport_shift=vshift)
    cal.viewport_pct_to_screenshot_pct.side_effect = lambda c, r: (c, r)

    camera_map_mod.compute_camera_mapping(cam, cal, rotation=-1)

    assert cal.viewport_pct_to_screenshot_pct.call_count == 15
