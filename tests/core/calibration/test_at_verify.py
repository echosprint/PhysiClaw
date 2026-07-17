"""Tests for `physiclaw.core.calibration.at_verify` — AT gesture checks.

`verify_assistive_touch` is exercised over its precondition errors (AT
position / nonce / viewport-shift unset), the screenshot-timeout and
decode-failure records, the full tap + double-tap + long-press success
path, the phone-mode choreography (grid up front, plain bridge page at
the end), and the clipboard-timeout partial failure.
"""

from __future__ import annotations

import logging
from unittest.mock import MagicMock

import cv2
import numpy as np
import pytest

from physiclaw.core.bridge.calib import CalibrationState
from physiclaw.core.calibration import at_verify as at_verify_mod
from physiclaw.core.calibration.transforms import ViewportShift


def _make_cal(*, viewport_shift=None) -> MagicMock:
    """A CalibrationState mock with the grid constants surfaced."""
    cal = MagicMock()
    cal.GRID_COLS_PCT = CalibrationState.GRID_COLS_PCT
    cal.GRID_ROWS_PCT = CalibrationState.GRID_ROWS_PCT
    cal.viewport_shift = viewport_shift
    return cal


def _identity_pct_to_grbl() -> np.ndarray:
    return np.array(
        [
            [10.0, 0.0, 0.0],
            [0.0, 20.0, 0.0],
        ]
    )


def test_verify_assistive_touch_raises_when_at_unset(mocker) -> None:
    arm = MagicMock()
    at = MagicMock()
    at.at_screen = None

    with pytest.raises(RuntimeError, match="AT position not set"):
        at_verify_mod.verify_assistive_touch(
            arm, at, MagicMock(), MagicMock(), _identity_pct_to_grbl(), MagicMock()
        )


def test_verify_assistive_touch_raises_when_no_nonce(mocker) -> None:
    arm = MagicMock()
    at = MagicMock()
    at.at_screen = (0.05, 0.1)
    cal = _make_cal()
    cal._screenshot_nonce = None

    with pytest.raises(RuntimeError, match="No nonce set"):
        at_verify_mod.verify_assistive_touch(
            arm, at, MagicMock(), cal, _identity_pct_to_grbl(), MagicMock()
        )


def test_verify_assistive_touch_returns_failed_dict_on_screenshot_timeout(
    mocker,
) -> None:
    mocker.patch.object(at_verify_mod.time, "sleep")
    arm = MagicMock()
    at = MagicMock()
    at.at_screen = (0.05, 0.1)
    cal = _make_cal()
    cal._screenshot_nonce = [1, 0, 1, 0]
    bridge = MagicMock()
    bridge.wait_screenshot.return_value = None  # timeout
    bridge.connectivity_hint.return_value = None  # phone still polling

    out = at_verify_mod.verify_assistive_touch(
        arm,
        at,
        bridge,
        cal,
        _identity_pct_to_grbl(),
        MagicMock(),
    )

    assert out["passed"] is False
    assert out["screenshot"]["passed"] is False
    assert out["clipboard"]["fetched"] is False


def test_verify_assistive_touch_screenshot_timeout_carries_connectivity_hint(
    mocker, caplog: pytest.LogCaptureFixture
) -> None:
    mocker.patch.object(at_verify_mod.time, "sleep")
    arm = MagicMock()
    at = MagicMock()
    at.at_screen = (0.05, 0.1)
    cal = _make_cal()
    cal._screenshot_nonce = [1, 0, 1, 0]
    bridge = MagicMock()
    bridge.wait_screenshot.return_value = None  # timeout
    bridge.connectivity_hint.return_value = (
        "the phone may be off Wi-Fi or on a different Wi-Fi network than this computer"
    )

    with caplog.at_level(
        logging.WARNING, logger="physiclaw.core.calibration.at_verify"
    ):
        out = at_verify_mod.verify_assistive_touch(
            arm,
            at,
            bridge,
            cal,
            _identity_pct_to_grbl(),
            MagicMock(),
        )

    assert out["passed"] is False
    assert any("may be off Wi-Fi" in r.getMessage() for r in caplog.records)


def test_verify_assistive_touch_returns_failed_dict_on_decode_error(mocker) -> None:
    mocker.patch.object(at_verify_mod.time, "sleep")
    arm = MagicMock()
    at = MagicMock()
    at.at_screen = (0.05, 0.1)
    cal = _make_cal()
    cal._screenshot_nonce = [1, 0, 1, 0]
    bridge = MagicMock()
    bridge.wait_screenshot.return_value = b"not an image"

    out = at_verify_mod.verify_assistive_touch(
        arm,
        at,
        bridge,
        cal,
        _identity_pct_to_grbl(),
        MagicMock(),
    )

    assert out["passed"] is False


def test_verify_assistive_touch_raises_when_viewport_shift_unset(mocker) -> None:
    mocker.patch.object(at_verify_mod.time, "sleep")
    arm = MagicMock()
    at = MagicMock()
    at.at_screen = (0.05, 0.1)
    cal = _make_cal()
    cal._screenshot_nonce = [1, 0, 1, 0]
    cal.viewport_shift = None
    bridge = MagicMock()
    img = np.zeros((400, 400, 3), dtype=np.uint8)
    ok, buf = cv2.imencode(".png", img)
    bridge.wait_screenshot.return_value = buf.tobytes()

    with pytest.raises(RuntimeError, match="viewport_shift not set"):
        at_verify_mod.verify_assistive_touch(
            arm, at, bridge, cal, _identity_pct_to_grbl(), MagicMock()
        )


def test_verify_assistive_touch_full_success_path(mocker) -> None:
    mocker.patch.object(at_verify_mod.time, "sleep")
    mocker.patch.object(
        at_verify_mod.random,
        "randbytes",
        return_value=b"\xab\xcd\xef",
    )
    arm = MagicMock()
    at = MagicMock()
    at.at_screen = (0.05, 0.1)
    vshift = ViewportShift(
        offset_x=0,
        offset_y=0,
        dpr=3.0,
        screenshot_width=1170,
        screenshot_height=2532,
    )
    cal = _make_cal(viewport_shift=vshift)
    cal._screenshot_nonce = [1, 0, 1, 0]
    bridge = MagicMock()
    bridge.wait_clipboard.return_value = True
    img = np.zeros((400, 400, 3), dtype=np.uint8)
    ok, buf = cv2.imencode(".png", img)
    bridge.wait_screenshot.return_value = buf.tobytes()

    mocker.patch.object(at_verify_mod, "verify_nonce", return_value=(True, 4))

    out = at_verify_mod.verify_assistive_touch(
        arm,
        at,
        bridge,
        cal,
        _identity_pct_to_grbl(),
        MagicMock(),
    )

    assert out["passed"] is True
    assert out["screenshot"]["passed"] is True
    assert out["clipboard"]["fetched"] is True
    assert out["clipboard"]["text"].startswith("PhysiClaw-")


def test_verify_assistive_touch_shows_clipboard_confirmation_on_phone(mocker) -> None:
    # On the clipboard sub-step the phone flips to bridge mode and the clip
    # text is queued, so the page shows it then the "copied" confirmation.
    mocker.patch.object(at_verify_mod.time, "sleep")
    mocker.patch.object(at_verify_mod.random, "randbytes", return_value=b"\xab\xcd\xef")
    arm = MagicMock()
    at = MagicMock()
    at.at_screen = (0.05, 0.1)
    vshift = ViewportShift(
        offset_x=0,
        offset_y=0,
        dpr=3.0,
        screenshot_width=1170,
        screenshot_height=2532,
    )
    cal = _make_cal(viewport_shift=vshift)
    cal._screenshot_nonce = [1, 0, 1, 0]
    bridge = MagicMock()
    bridge.wait_clipboard.return_value = True
    img = np.zeros((400, 400, 3), dtype=np.uint8)
    ok, buf = cv2.imencode(".png", img)
    bridge.wait_screenshot.return_value = buf.tobytes()
    mocker.patch.object(at_verify_mod, "verify_nonce", return_value=(True, 4))
    phone = MagicMock()

    at_verify_mod.verify_assistive_touch(
        arm,
        at,
        bridge,
        cal,
        _identity_pct_to_grbl(),
        phone,
    )

    # Establishes its own grid up front (so a re-run doesn't depend on
    # /show), then flips to bridge mode for the confirmation and stays there
    # — success ends on the plain bridge page, not the grid.
    assert phone.set_mode.call_args_list == [
        mocker.call("calibrate", "assistive_touch", nonce_bits=[1, 0, 1, 0]),
        mocker.call("bridge"),
    ]
    bridge.send_text.assert_called_once()
    assert bridge.send_text.call_args[0][0].startswith("PhysiClaw-")
    bridge.clear_text.assert_called_once()


def test_verify_assistive_touch_clipboard_timeout(mocker) -> None:
    mocker.patch.object(at_verify_mod.time, "sleep")
    mocker.patch.object(at_verify_mod.random, "randbytes", return_value=b"\x01\x02\x03")
    arm = MagicMock()
    at = MagicMock()
    at.at_screen = (0.05, 0.1)
    vshift = ViewportShift(
        offset_x=0,
        offset_y=0,
        dpr=3.0,
        screenshot_width=1170,
        screenshot_height=2532,
    )
    cal = _make_cal(viewport_shift=vshift)
    cal._screenshot_nonce = [1, 0, 1, 0]
    bridge = MagicMock()
    bridge.wait_clipboard.return_value = False
    img = np.zeros((400, 400, 3), dtype=np.uint8)
    ok, buf = cv2.imencode(".png", img)
    bridge.wait_screenshot.return_value = buf.tobytes()
    mocker.patch.object(at_verify_mod, "verify_nonce", return_value=(True, 4))

    out = at_verify_mod.verify_assistive_touch(
        arm,
        at,
        bridge,
        cal,
        _identity_pct_to_grbl(),
        MagicMock(),
    )

    # Screenshot passed but clipboard didn't → overall failure.
    assert out["passed"] is False
    assert out["screenshot"]["passed"] is True
    assert out["clipboard"]["fetched"] is False
