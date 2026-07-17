"""Tests for `physiclaw.core.orchestration.camera_pick` — camera
identification and preview algorithms (moved out of the old
`core.hardware.handler`)."""

from __future__ import annotations

import logging
from unittest.mock import MagicMock

import numpy as np
import pytest

from physiclaw.core.orchestration import camera_pick
from physiclaw.core.orchestration.camera_pick import (
    _auto_pick_camera_index,
    _capture_raw,
    camera_preview,
    resolve_auto_index,
)

# ---------- camera_preview ----------


def test_camera_preview_returns_jpeg_bytes(mocker) -> None:
    fake_frame = np.zeros((10, 10, 3), dtype=np.uint8)
    fake_cam = MagicMock()
    fake_cam.snapshot.return_value = fake_frame
    mocker.patch.object(camera_pick, "Camera", return_value=fake_cam)
    encode_spy = mocker.patch.object(camera_pick, "encode_jpeg", return_value=b"JPEG")
    wm_spy = mocker.patch.object(camera_pick, "watermark_index")

    out = camera_preview(0, watermark=False)

    assert out == b"JPEG"
    fake_cam.close.assert_called_once()
    encode_spy.assert_called_once_with(fake_frame, quality=80)
    wm_spy.assert_not_called()


def test_camera_preview_applies_watermark_when_requested(mocker) -> None:
    fake_frame = np.zeros((10, 10, 3), dtype=np.uint8)
    wm_frame = np.ones((10, 10, 3), dtype=np.uint8)
    fake_cam = MagicMock()
    fake_cam.snapshot.return_value = fake_frame
    mocker.patch.object(camera_pick, "Camera", return_value=fake_cam)
    mocker.patch.object(camera_pick, "watermark_index", return_value=wm_frame)
    encode_spy = mocker.patch.object(camera_pick, "encode_jpeg", return_value=b"JPEG")

    out = camera_preview(2, watermark=True)

    assert out == b"JPEG"
    encode_spy.assert_called_once()
    # Watermarked frame is what gets encoded.
    assert encode_spy.call_args.args[0] is wm_frame


def test_camera_preview_raises_when_snapshot_returns_none(mocker) -> None:
    fake_cam = MagicMock()
    fake_cam.snapshot.return_value = None
    mocker.patch.object(camera_pick, "Camera", return_value=fake_cam)

    with pytest.raises(RuntimeError, match="Camera 0 returned no frame"):
        camera_preview(0)

    fake_cam.close.assert_called_once()


# ---------- _capture_raw ----------


def test_capture_raw_returns_frame(mocker) -> None:
    frame = np.ones((4, 4, 3), dtype=np.uint8)
    fake_cam = MagicMock()
    fake_cam.raw_frame.return_value = frame
    mocker.patch.object(camera_pick, "Camera", return_value=fake_cam)

    assert _capture_raw(1) is frame
    fake_cam.close.assert_called_once()


def test_capture_raw_returns_none_and_logs_on_oserror(
    mocker, caplog: pytest.LogCaptureFixture
) -> None:
    fake_cam = MagicMock()
    fake_cam.raw_frame.side_effect = OSError("can't open")
    mocker.patch.object(camera_pick, "Camera", return_value=fake_cam)

    with caplog.at_level(
        logging.WARNING, logger="physiclaw.core.orchestration.camera_pick"
    ):
        out = _capture_raw(3)

    assert out is None
    fake_cam.close.assert_called_once()
    assert any("cam 3: capture failed" in r.getMessage() for r in caplog.records)


def test_capture_raw_returns_none_on_runtime_error(mocker) -> None:
    fake_cam = MagicMock()
    fake_cam.raw_frame.side_effect = RuntimeError("transient")
    mocker.patch.object(camera_pick, "Camera", return_value=fake_cam)

    assert _capture_raw(0) is None
    fake_cam.close.assert_called_once()


def test_capture_raw_returns_none_when_constructor_raises(mocker) -> None:
    """A missing USB index raises from Camera() itself — the auto-pick
    loop must treat it like any other capture failure, not crash."""
    mocker.patch.object(camera_pick, "Camera", side_effect=RuntimeError("no device"))

    assert _capture_raw(5) is None


# ---------- _auto_pick_camera_index ----------


def test_auto_pick_camera_returns_first_match(mocker) -> None:
    frames = [None, np.zeros((4, 4, 3), dtype=np.uint8), None]
    capture_spy = mocker.patch.object(
        camera_pick,
        "_capture_raw",
        side_effect=lambda idx: frames[idx] if idx < len(frames) else None,
    )

    def detect_corners(f):
        # Only the second frame has corners.
        return [(0, 0), (1, 0), (1, 1), (0, 1)] if f is not None else None

    mocker.patch.object(
        camera_pick, "detect_bridge_corners", side_effect=detect_corners
    )

    out = _auto_pick_camera_index()

    assert out == 1
    # Probes 0 (None) and 1 (success) — stops there.
    assert capture_spy.call_count >= 2


def test_auto_pick_camera_returns_none_when_no_match(mocker) -> None:
    mocker.patch.object(camera_pick, "_capture_raw", return_value=None)
    mocker.patch.object(camera_pick, "detect_bridge_corners", return_value=None)

    assert _auto_pick_camera_index() is None


def test_auto_pick_camera_skips_when_corners_not_detected(
    mocker, caplog: pytest.LogCaptureFixture
) -> None:
    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    mocker.patch.object(camera_pick, "_capture_raw", return_value=frame)
    mocker.patch.object(camera_pick, "detect_bridge_corners", return_value=None)

    with caplog.at_level(
        logging.INFO, logger="physiclaw.core.orchestration.camera_pick"
    ):
        out = _auto_pick_camera_index()

    assert out is None
    assert any("corners not detected" in r.getMessage() for r in caplog.records)


# ---------- resolve_auto_index ----------


def test_resolve_auto_index_happy_path(mocker) -> None:
    rig = MagicMock()
    rig.require_bridge.return_value = rig.bridge
    rig.bridge.wait_for_connection.return_value = True
    phone = MagicMock()
    mocker.patch.object(camera_pick, "_auto_pick_camera_index", return_value=4)
    mocker.patch.object(camera_pick.time, "sleep")

    out = resolve_auto_index(rig, phone)

    assert out == 4
    rig.bridge.wait_for_connection.assert_called_once()
    # Sets corners then restores bridge.
    calls = phone.set_mode.call_args_list
    assert calls[0].args == ("calibrate",)
    assert calls[0].kwargs == {"phase": "corners"}
    assert calls[-1].args == ("bridge",)


def test_resolve_auto_index_fails_when_bridge_not_polling(mocker) -> None:
    rig = MagicMock()
    rig.require_bridge.return_value = rig.bridge
    rig.bridge.wait_for_connection.return_value = False
    phone = MagicMock()

    with pytest.raises(RuntimeError, match="not polling"):
        resolve_auto_index(rig, phone)

    # Phone never reaches corners phase.
    phone.set_mode.assert_not_called()


def test_resolve_auto_index_restores_bridge_mode_on_no_match(mocker) -> None:
    rig = MagicMock()
    rig.require_bridge.return_value = rig.bridge
    rig.bridge.wait_for_connection.return_value = True
    phone = MagicMock()
    mocker.patch.object(camera_pick, "_auto_pick_camera_index", return_value=None)
    mocker.patch.object(camera_pick.time, "sleep")

    with pytest.raises(RuntimeError, match="no camera with all four RGBM corners"):
        resolve_auto_index(rig, phone)

    # Phone restored to bridge mode even on failure.
    assert phone.set_mode.call_args_list[-1].args == ("bridge",)
