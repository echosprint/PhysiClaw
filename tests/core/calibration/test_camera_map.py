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
from tests.core.calibration.conftest import make_cal as _make_cal


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
    cam.focus_lockable = False
    cam.raw_frame.return_value = _grid_dot_image()
    cal = _make_cal()

    pct_to_cam, cam_size, _ = camera_map_mod.compute_camera_mapping(
        cam, cal, rotation=-1
    )

    assert pct_to_cam.shape == (2, 3)
    assert cam_size == (600, 900)


def test_compute_camera_mapping_camera_dead(mocker) -> None:
    mocker.patch.object(camera_map_mod.time, "sleep")
    cam = MagicMock()
    cam.focus_lockable = False
    cam.raw_frame.return_value = None
    cal = _make_cal()

    with pytest.raises(RuntimeError, match="camera read failed"):
        camera_map_mod.compute_camera_mapping(cam, cal, rotation=-1)


def test_compute_camera_mapping_retries_then_fails(mocker) -> None:
    mocker.patch.object(camera_map_mod.time, "sleep")
    cam = MagicMock()
    cam.focus_lockable = False
    blank = np.zeros((600, 900, 3), dtype=np.uint8)
    cam.raw_frame.return_value = blank
    cal = _make_cal()

    with pytest.raises(RuntimeError, match=r"last saw 0 dots inside the screen"):
        camera_map_mod.compute_camera_mapping(cam, cal, rotation=-1)


def test_compute_camera_mapping_applies_rotation(mocker) -> None:
    mocker.patch.object(camera_map_mod.time, "sleep")
    cam = MagicMock()
    cam.focus_lockable = False
    # The camera returns a frame that is canonical only AFTER the configured
    # rotation is applied — so pre-rotate the canonical grid the opposite way.
    # This is the real scenario: rotation is picked to upright the grid, and
    # only then does sort_dots_to_grid (and the fit gate) line up.
    canonical = _grid_dot_image()
    cam.raw_frame.return_value = cv2.rotate(canonical, cv2.ROTATE_90_COUNTERCLOCKWISE)
    cal = _make_cal()

    rotate_spy = mocker.spy(camera_map_mod.cv2, "rotate")

    pct_to_cam, _, _ = camera_map_mod.compute_camera_mapping(
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
    cam.focus_lockable = False
    cam.raw_frame.side_effect = [corners_frame] + [grid_frame] * 5
    cal = _make_cal()

    pct_to_cam, cam_size, _ = camera_map_mod.compute_camera_mapping(
        cam, cal, rotation=-1
    )

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
    cam.focus_lockable = False
    cam.raw_frame.return_value = frame
    cal = _make_cal()

    with pytest.raises(RuntimeError, match=r"clean 15-dot grid passing the fit gate"):
        camera_map_mod.compute_camera_mapping(cam, cal, rotation=-1)


def test_compute_camera_mapping_uses_viewport_shift(mocker) -> None:
    mocker.patch.object(camera_map_mod.time, "sleep")
    cam = MagicMock()
    cam.focus_lockable = False
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


def test_compute_camera_mapping_requires_viewport_shift(mocker) -> None:
    """Same precondition as the arm fit — Mapping B must land in the
    canonical screenshot 0-1 space, never raw viewport 0-1."""
    mocker.patch.object(camera_map_mod.time, "sleep")
    cal = _make_cal(viewport_shift=None)

    with pytest.raises(RuntimeError, match="Viewport shift not measured"):
        camera_map_mod.compute_camera_mapping(MagicMock(), cal, rotation=-1)


# ---------- focus pin ----------


def _sharp_frame(w=600, h=900) -> np.ndarray:
    """High-variance noise — meters far above BLUR_THRESHOLD."""
    rng = np.random.default_rng(7)
    return rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)


def _lockable_cam(frame: np.ndarray | None) -> MagicMock:
    cam = MagicMock()
    cam.focus_lockable = True
    cam.wait_frames.return_value = True
    cam.raw_frame.return_value = frame
    cam.lock_focus.return_value = True
    return cam


def test_pin_focus_returns_none_when_not_lockable() -> None:
    cam = MagicMock()
    cam.focus_lockable = False

    assert camera_map_mod._pin_focus(cam, -1, None) is None
    cam.lock_focus.assert_not_called()


def test_pin_focus_releases_a_stale_pin_before_relocking() -> None:
    # A re-run (mapping retried, or a setup re-run over a warm-started
    # session) arrives with the lens still pinned at the OLD position —
    # frozen, AF can't re-converge. The pin must start from live AF.
    cam = _lockable_cam(_sharp_frame())
    cam.read_focus.return_value = 123.0
    order: list[str] = []
    cam.unlock_focus.side_effect = lambda: order.append("unlock")
    cam.lock_focus.side_effect = lambda: order.append("lock") or True

    camera_map_mod._pin_focus(cam, -1, None)

    assert order[0] == "unlock"
    assert "lock" in order


def test_pin_focus_locks_reads_and_reapplies() -> None:
    # The verified position is read back AND re-applied through
    # apply_focus, so the camera remembers the exact value across a
    # reopen instead of re-freezing wherever the lens sits.
    cam = _lockable_cam(_sharp_frame())
    cam.read_focus.return_value = 123.0

    value = camera_map_mod._pin_focus(cam, -1, None)

    assert value == 123.0
    cam.apply_focus.assert_called_once_with(123.0)


def test_pin_focus_returns_none_when_scene_never_sharp() -> None:
    # A blank frame can't converge AF — the lock defers and nothing is
    # persisted (the bundle's cam_focus stays None; AF stays live).
    cam = _lockable_cam(np.zeros((900, 600, 3), dtype=np.uint8))

    value = camera_map_mod._pin_focus(cam, -1, None)

    assert value is None
    cam.read_focus.assert_not_called()


def test_pin_focus_returns_none_when_position_unreadable() -> None:
    # The freeze verified but the driver reports no position — nothing
    # a reconnect could replay, so the lens goes back to live AF and
    # every session behaves the same.
    cam = _lockable_cam(_sharp_frame())
    cam.read_focus.return_value = None

    value = camera_map_mod._pin_focus(cam, -1, None)

    assert value is None
    cam.apply_focus.assert_not_called()
    cam.unlock_focus.assert_called()


def test_pin_focus_returns_none_when_apply_refused() -> None:
    # The driver froze the lens but rejects absolute writes — a
    # persisted value would fail to re-apply on every future connect,
    # so nothing is persisted and the lens goes back to AF.
    cam = _lockable_cam(_sharp_frame())
    cam.read_focus.return_value = 123.0
    cam.apply_focus.return_value = False

    value = camera_map_mod._pin_focus(cam, -1, None)

    assert value is None
    cam.unlock_focus.assert_called()


def test_pin_focus_meters_only_the_screen_region() -> None:
    # Sharp content inside the corner-fenced screen, dark desk outside —
    # the meter must score the fence's bounding rect, not the full frame.
    frame = np.zeros((900, 600, 3), dtype=np.uint8)
    frame[100:800, 50:550] = _sharp_frame(w=500, h=700)
    poly = np.array([[50, 100], [550, 100], [550, 800], [50, 800]])

    region = camera_map_mod._focus_region(frame, poly)

    assert region.shape[:2] == (701, 501)  # boundingRect is inclusive


def test_focus_region_full_frame_without_fence() -> None:
    frame = np.zeros((900, 600, 3), dtype=np.uint8)

    assert camera_map_mod._focus_region(frame, None) is frame


def test_compute_camera_mapping_returns_pinned_focus(mocker) -> None:
    mocker.patch.object(camera_map_mod.time, "sleep")
    cam = MagicMock()
    cam.focus_lockable = False
    cam.raw_frame.return_value = _grid_dot_image()
    cal = _make_cal()
    mocker.patch.object(camera_map_mod, "_pin_focus", return_value=42.0)

    _, _, cam_focus = camera_map_mod.compute_camera_mapping(cam, cal, rotation=-1)

    assert cam_focus == 42.0
