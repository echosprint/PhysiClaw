"""Tests for `physiclaw.core.hardware.iphone` — AssistiveTouch driver."""

from __future__ import annotations

import logging
from unittest.mock import MagicMock

import numpy as np
import pytest

from physiclaw.core.calibration.transforms import ViewportShift
from physiclaw.core.hardware.iphone import AssistiveTouch

# ---------- Helpers ----------


def _shift(
    *,
    dpr: float = 3.0,
    offset_x: float = 0.0,
    offset_y: float = 0.0,
    w: int = 1170,
    h: int = 2532,
) -> ViewportShift:
    return ViewportShift(
        offset_x=offset_x,
        offset_y=offset_y,
        dpr=dpr,
        screenshot_width=w,
        screenshot_height=h,
    )


def _identity_pct_to_grbl() -> np.ndarray:
    return np.eye(3)


# ---------- ready / construction ----------


def test_init_ready_false_until_compute() -> None:
    at = AssistiveTouch()

    assert at.ready is False
    assert at.at_screen is None
    assert at.at_radius_screen is None


def test_ready_true_after_compute() -> None:
    at = AssistiveTouch()
    at.compute_at_screen_pos(_shift())

    assert at.ready is True


# ---------- compute_at_screen_pos ----------


def test_compute_at_screen_pos_returns_and_stores() -> None:
    at = AssistiveTouch()
    s = _shift(dpr=3.0, w=1170, h=2532, offset_x=0, offset_y=0)

    out = at.compute_at_screen_pos(s)

    # AT_CSS_X=39, dpr=3 → 117 / 1170 = 0.1
    # AT_CSS_Y=260, dpr=3 → 780 / 2532 ≈ 0.3081
    assert out == at.at_screen
    assert out[0] == pytest.approx(117 / 1170)
    assert out[1] == pytest.approx(780 / 2532)


def test_compute_at_screen_pos_radius_axes() -> None:
    at = AssistiveTouch()
    s = _shift(dpr=3.0, w=1170, h=2532)

    at.compute_at_screen_pos(s)

    rx, ry = at.at_radius_screen
    assert rx == pytest.approx(28 * 3.0 / 1170)
    assert ry == pytest.approx(28 * 3.0 / 2532)


def test_compute_at_screen_pos_logs(caplog: pytest.LogCaptureFixture) -> None:
    at = AssistiveTouch()

    with caplog.at_level(logging.INFO, logger="physiclaw.core.hardware.iphone"):
        at.compute_at_screen_pos(_shift())

    assert any("AT screen position" in r.getMessage() for r in caplog.records)


# ---------- overlaps_at ----------


def test_overlaps_at_returns_false_when_unset() -> None:
    at = AssistiveTouch()

    assert at.overlaps_at(0.5, 0.5) is False


def test_overlaps_at_center_inside_ellipse() -> None:
    at = AssistiveTouch()
    at.compute_at_screen_pos(_shift())
    ax, ay = at.at_screen

    assert at.overlaps_at(ax, ay) is True


def test_overlaps_at_far_point_outside() -> None:
    at = AssistiveTouch()
    at.compute_at_screen_pos(_shift())

    assert at.overlaps_at(0.9, 0.9) is False


def test_overlaps_at_just_outside_radius() -> None:
    at = AssistiveTouch()
    at.compute_at_screen_pos(_shift())
    ax, ay = at.at_screen
    rx, _ry = at.at_radius_screen

    # Outside the ellipse along x.
    assert at.overlaps_at(ax + rx * 1.01, ay) is False
    # Inside.
    assert at.overlaps_at(ax + rx * 0.5, ay) is True


def test_overlaps_at_returns_false_when_radius_unset() -> None:
    at = AssistiveTouch()
    at.at_screen = (0.1, 0.1)  # Manually set screen but not radius.

    assert at.overlaps_at(0.1, 0.1) is False


# ---------- swipe_crosses_at ----------


def test_swipe_crosses_at_unset_returns_false() -> None:
    at = AssistiveTouch()

    assert at.swipe_crosses_at(0.5, 0.5, "up") is False


def test_swipe_crosses_at_vertical_directions_check_x_overlap() -> None:
    at = AssistiveTouch()
    at.compute_at_screen_pos(_shift())
    ax, _ay = at.at_screen

    # Same column as AT → vertical swipes cross.
    assert at.swipe_crosses_at(ax, 0.5, "up") is True
    assert at.swipe_crosses_at(ax, 0.5, "down") is True
    # Far column.
    assert at.swipe_crosses_at(0.9, 0.5, "up") is False
    assert at.swipe_crosses_at(0.9, 0.5, "down") is False


def test_swipe_crosses_at_horizontal_directions_check_y_overlap() -> None:
    at = AssistiveTouch()
    at.compute_at_screen_pos(_shift())
    _ax, ay = at.at_screen

    assert at.swipe_crosses_at(0.5, ay, "left") is True
    assert at.swipe_crosses_at(0.5, ay, "right") is True
    assert at.swipe_crosses_at(0.5, 0.9, "left") is False
    assert at.swipe_crosses_at(0.5, 0.9, "right") is False


def test_swipe_crosses_at_unknown_direction_returns_false() -> None:
    at = AssistiveTouch()
    at.compute_at_screen_pos(_shift())

    assert at.swipe_crosses_at(0.5, 0.5, "diagonal") is False


def test_swipe_crosses_at_returns_false_when_radius_unset() -> None:
    at = AssistiveTouch()
    at.at_screen = (0.1, 0.1)

    assert at.swipe_crosses_at(0.1, 0.5, "up") is False


# ---------- _move_to_at / tap variants ----------


def test_move_to_at_raises_when_position_unset() -> None:
    at = AssistiveTouch()
    arm = MagicMock()

    with pytest.raises(RuntimeError, match="AT position not set"):
        at._move_to_at(arm, _identity_pct_to_grbl())


def test_move_to_at_calls_fast_move_with_grbl_coordinates() -> None:
    at = AssistiveTouch()
    at.compute_at_screen_pos(_shift())
    arm = MagicMock()
    pct_to_grbl = np.array(
        [
            [10.0, 0.0, 1.0],
            [0.0, 20.0, 2.0],
            [0.0, 0.0, 1.0],
        ]
    )
    sx, sy = at.at_screen
    expected_x = 10.0 * sx + 1.0
    expected_y = 20.0 * sy + 2.0

    at._move_to_at(arm, pct_to_grbl)

    arm._fast_move.assert_called_once()
    args = arm._fast_move.call_args.args
    assert args[0] == pytest.approx(expected_x)
    assert args[1] == pytest.approx(expected_y)
    arm.wait_idle.assert_called_once()


def test_tap_raises_when_position_unset() -> None:
    at = AssistiveTouch()
    arm = MagicMock()

    with pytest.raises(RuntimeError, match="AT position not set"):
        at.tap(arm, _identity_pct_to_grbl())


def test_tap_moves_then_taps(caplog: pytest.LogCaptureFixture) -> None:
    at = AssistiveTouch()
    at.compute_at_screen_pos(_shift())
    arm = MagicMock()

    with caplog.at_level(logging.INFO, logger="physiclaw.core.hardware.iphone"):
        at.tap(arm, _identity_pct_to_grbl())

    arm._fast_move.assert_called_once()
    arm.tap.assert_called_once()
    assert any("single-tap" in r.getMessage() for r in caplog.records)


def test_double_tap_raises_when_position_unset() -> None:
    at = AssistiveTouch()
    arm = MagicMock()

    with pytest.raises(RuntimeError, match="AT position not set"):
        at.double_tap(arm, _identity_pct_to_grbl())


def test_double_tap_calls_arm_double_tap(caplog: pytest.LogCaptureFixture) -> None:
    at = AssistiveTouch()
    at.compute_at_screen_pos(_shift())
    arm = MagicMock()

    with caplog.at_level(logging.INFO, logger="physiclaw.core.hardware.iphone"):
        at.double_tap(arm, _identity_pct_to_grbl())

    arm.double_tap.assert_called_once()
    assert any("double-tap" in r.getMessage() for r in caplog.records)


def test_long_press_raises_when_position_unset() -> None:
    at = AssistiveTouch()
    arm = MagicMock()

    with pytest.raises(RuntimeError, match="AT position not set"):
        at.long_press(arm, _identity_pct_to_grbl())


def test_long_press_calls_arm_long_press(caplog: pytest.LogCaptureFixture) -> None:
    at = AssistiveTouch()
    at.compute_at_screen_pos(_shift())
    arm = MagicMock()

    with caplog.at_level(logging.INFO, logger="physiclaw.core.hardware.iphone"):
        at.long_press(arm, _identity_pct_to_grbl())

    arm.long_press.assert_called_once()
    assert any("long-press" in r.getMessage() for r in caplog.records)


# ---------- take_screenshot ----------


def test_take_screenshot_returns_bytes_on_success(mocker) -> None:
    mocker.patch("physiclaw.core.hardware.iphone.time.sleep")
    at = AssistiveTouch()
    at.compute_at_screen_pos(_shift())
    arm = MagicMock()
    bridge = MagicMock()
    bridge.wait_screenshot.return_value = b"\x89PNG"

    out = at.take_screenshot(arm, bridge, _identity_pct_to_grbl(), timeout=0.5)

    assert out == b"\x89PNG"
    bridge.clear_screenshot.assert_called_once()
    arm.tap.assert_called_once()
    arm.double_tap.assert_called_once()
    bridge.wait_screenshot.assert_called_once_with(timeout=0.5)


def test_take_screenshot_warns_and_returns_none_on_timeout(
    mocker, caplog: pytest.LogCaptureFixture
) -> None:
    mocker.patch("physiclaw.core.hardware.iphone.time.sleep")
    at = AssistiveTouch()
    at.compute_at_screen_pos(_shift())
    arm = MagicMock()
    bridge = MagicMock()
    bridge.wait_screenshot.return_value = None

    with caplog.at_level(logging.WARNING, logger="physiclaw.core.hardware.iphone"):
        out = at.take_screenshot(arm, bridge, _identity_pct_to_grbl())

    assert out is None
    assert any("upload timed out" in r.getMessage() for r in caplog.records)


def test_take_screenshot_sleeps_5s_between_taps(mocker) -> None:
    sleep_spy = mocker.patch("physiclaw.core.hardware.iphone.time.sleep")
    at = AssistiveTouch()
    at.compute_at_screen_pos(_shift())
    arm = MagicMock()
    bridge = MagicMock()
    bridge.wait_screenshot.return_value = b"\x00"

    at.take_screenshot(arm, bridge, _identity_pct_to_grbl())

    sleep_spy.assert_called_once_with(5.0)


def test_take_screenshot_default_timeout_is_10s(mocker) -> None:
    mocker.patch("physiclaw.core.hardware.iphone.time.sleep")
    at = AssistiveTouch()
    at.compute_at_screen_pos(_shift())
    arm = MagicMock()
    bridge = MagicMock()
    bridge.wait_screenshot.return_value = b"\x00"

    at.take_screenshot(arm, bridge, _identity_pct_to_grbl())

    bridge.wait_screenshot.assert_called_once_with(timeout=10.0)
