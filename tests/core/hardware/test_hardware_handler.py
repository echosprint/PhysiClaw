"""Tests for `physiclaw.core.hardware.handler` — hardware setup HTTP routes."""

from __future__ import annotations

import base64
import json
import logging
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

from physiclaw.core.hardware import handler
from physiclaw.core.hardware.handler import (
    _auto_pick_camera_index,
    _capture_raw,
    camera_preview,
    handle_camera_preview,
    handle_connect_arm,
    handle_connect_camera,
    handle_disconnect_camera,
    handle_setup_page,
    handle_status,
)


@pytest.mark.asyncio
async def test_handle_setup_page_serves_wizard(mocker) -> None:
    # Substitution lives in bridge.handler.render_phone_page_html now.
    import physiclaw.core.bridge.handler as bridge_handler

    mocker.patch.object(
        bridge_handler,
        "bridge_base_urls",
        return_value=("http://device.local:8048", "http://10.0.0.5:8048"),
    )
    req = _fake_request()
    req.url = SimpleNamespace(port=8048)

    resp = await handle_setup_page(req)

    assert resp.status_code == 200
    assert resp.headers["cache-control"] == "no-store"
    body = bytes(resp.body)
    assert b"Hardware Setup" in body
    # drives the same API the CLI does
    assert b"/api/calibrate/arm" in body
    # phone bridge URLs are substituted in so the QR renders inline (no iframe)
    assert b"__PHONE_URL__" not in body
    assert b"http://device.local:8048/bridge" in body
    assert b"<iframe" not in body


def _async(value: Any):
    async def _coro():
        return value

    return _coro


def _async_raise(exc: Exception):
    async def _coro():
        raise exc

    return _coro


def _fake_request(
    json_obj: Any = None,
    raise_on_json: bool = False,
    path_params: dict | None = None,
    query_params: dict | None = None,
):
    req = SimpleNamespace()
    if raise_on_json:
        req.json = _async_raise(RuntimeError("bad body"))
    else:
        req.json = _async(json_obj)
    req.path_params = path_params or {}
    req.query_params = query_params or {}
    return req


def _read_json(response) -> dict:
    return json.loads(bytes(response.body).decode())


# ---------- handle_status ----------


@pytest.mark.asyncio
async def test_handle_status_returns_status_dict() -> None:
    physiclaw = SimpleNamespace(status=lambda: {"arm": True, "cam": False})
    req = _fake_request()

    resp = await handle_status(req, physiclaw)

    assert _read_json(resp) == {"arm": True, "cam": False}


# ---------- handle_connect_arm ----------


@pytest.mark.asyncio
async def test_handle_connect_arm_happy_path() -> None:
    physiclaw = MagicMock()

    resp = await handle_connect_arm(_fake_request(), physiclaw)

    body = _read_json(resp)
    assert body["status"] == "ok"
    assert "Arm connected" in body["message"]
    physiclaw.acquire.assert_called_once()
    physiclaw.connect_arm.assert_called_once()
    physiclaw.release.assert_called_once()


@pytest.mark.asyncio
async def test_handle_connect_arm_releases_even_on_failure() -> None:
    physiclaw = MagicMock()
    physiclaw.connect_arm.side_effect = RuntimeError("no port")

    resp = await handle_connect_arm(_fake_request(), physiclaw)

    assert resp.status_code == 500
    body = _read_json(resp)
    assert body["status"] == "error"
    assert "no port" in body["message"]
    physiclaw.release.assert_called_once()


# ---------- camera_preview ----------


def test_camera_preview_returns_jpeg_bytes(mocker) -> None:
    fake_frame = np.zeros((10, 10, 3), dtype=np.uint8)
    fake_cam = MagicMock()
    fake_cam.snapshot.return_value = fake_frame
    mocker.patch.object(handler, "Camera", return_value=fake_cam)
    encode_spy = mocker.patch.object(handler, "encode_jpeg", return_value=b"JPEG")
    wm_spy = mocker.patch.object(handler, "watermark_index")

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
    mocker.patch.object(handler, "Camera", return_value=fake_cam)
    mocker.patch.object(handler, "watermark_index", return_value=wm_frame)
    encode_spy = mocker.patch.object(handler, "encode_jpeg", return_value=b"JPEG")

    out = camera_preview(2, watermark=True)

    assert out == b"JPEG"
    encode_spy.assert_called_once()
    # Watermarked frame is what gets encoded.
    assert encode_spy.call_args.args[0] is wm_frame


def test_camera_preview_raises_when_snapshot_returns_none(mocker) -> None:
    fake_cam = MagicMock()
    fake_cam.snapshot.return_value = None
    mocker.patch.object(handler, "Camera", return_value=fake_cam)

    with pytest.raises(RuntimeError, match="Camera 0 returned no frame"):
        camera_preview(0)

    fake_cam.close.assert_called_once()


# ---------- _capture_raw ----------


def test_capture_raw_returns_frame(mocker) -> None:
    frame = np.ones((4, 4, 3), dtype=np.uint8)
    fake_cam = MagicMock()
    fake_cam.raw_frame.return_value = frame
    mocker.patch.object(handler, "Camera", return_value=fake_cam)

    assert _capture_raw(1) is frame
    fake_cam.close.assert_called_once()


def test_capture_raw_returns_none_and_logs_on_oserror(
    mocker, caplog: pytest.LogCaptureFixture
) -> None:
    fake_cam = MagicMock()
    fake_cam.raw_frame.side_effect = OSError("can't open")
    mocker.patch.object(handler, "Camera", return_value=fake_cam)

    with caplog.at_level(logging.WARNING, logger="physiclaw.core.hardware.handler"):
        out = _capture_raw(3)

    assert out is None
    fake_cam.close.assert_called_once()
    assert any("cam 3: capture failed" in r.getMessage() for r in caplog.records)


def test_capture_raw_returns_none_on_runtime_error(mocker) -> None:
    fake_cam = MagicMock()
    fake_cam.raw_frame.side_effect = RuntimeError("transient")
    mocker.patch.object(handler, "Camera", return_value=fake_cam)

    assert _capture_raw(0) is None
    fake_cam.close.assert_called_once()


def test_capture_raw_returns_none_when_constructor_raises(mocker) -> None:
    """A missing USB index raises from Camera() itself — the auto-pick
    loop must treat it like any other capture failure, not crash."""
    mocker.patch.object(handler, "Camera", side_effect=RuntimeError("no device"))

    assert _capture_raw(5) is None


# ---------- _auto_pick_camera_index ----------


def test_auto_pick_camera_returns_first_match(mocker) -> None:
    frames = [None, np.zeros((4, 4, 3), dtype=np.uint8), None]
    capture_spy = mocker.patch.object(
        handler,
        "_capture_raw",
        side_effect=lambda idx: frames[idx] if idx < len(frames) else None,
    )

    def detect_corners(f):
        # Only the second frame has corners.
        return [(0, 0), (1, 0), (1, 1), (0, 1)] if f is not None else None

    mocker.patch.object(handler, "detect_bridge_corners", side_effect=detect_corners)

    out = _auto_pick_camera_index()

    assert out == 1
    # Probes 0 (None) and 1 (success) — stops there.
    assert capture_spy.call_count >= 2


def test_auto_pick_camera_returns_none_when_no_match(mocker) -> None:
    mocker.patch.object(handler, "_capture_raw", return_value=None)
    mocker.patch.object(handler, "detect_bridge_corners", return_value=None)

    assert _auto_pick_camera_index() is None


def test_auto_pick_camera_skips_when_corners_not_detected(
    mocker, caplog: pytest.LogCaptureFixture
) -> None:
    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    mocker.patch.object(handler, "_capture_raw", return_value=frame)
    mocker.patch.object(handler, "detect_bridge_corners", return_value=None)

    with caplog.at_level(logging.INFO, logger="physiclaw.core.hardware.handler"):
        out = _auto_pick_camera_index()

    assert out is None
    assert any("corners not detected" in r.getMessage() for r in caplog.records)


# ---------- handle_connect_camera ----------


def _fake_physiclaw_cam_index(idx: int = 5) -> SimpleNamespace:
    """Build a fake physiclaw with .cam.index and required methods."""
    physiclaw = MagicMock()
    physiclaw.cam.index = idx
    return physiclaw


@pytest.mark.asyncio
async def test_handle_connect_camera_explicit_index(mocker) -> None:
    physiclaw = _fake_physiclaw_cam_index(idx=2)
    phone = MagicMock()

    resp = await handle_connect_camera(
        _fake_request(json_obj={"index": 2}),
        physiclaw,
        phone,
    )

    body = _read_json(resp)
    assert body["status"] == "ok"
    assert body["index"] == 2
    physiclaw.connect_camera.assert_called_once_with(2)
    # No auto-pick → no phase set.
    phone.set_mode.assert_not_called()


@pytest.mark.asyncio
async def test_handle_connect_camera_auto_pick_happy_path(mocker) -> None:
    physiclaw = _fake_physiclaw_cam_index(idx=4)
    physiclaw._bridge.wait_for_connection.return_value = True
    phone = MagicMock()
    mocker.patch.object(handler, "_auto_pick_camera_index", return_value=4)
    mocker.patch.object(handler.time, "sleep")

    resp = await handle_connect_camera(
        _fake_request(json_obj={"index": "auto"}),
        physiclaw,
        phone,
    )

    body = _read_json(resp)
    assert body["status"] == "ok"
    assert body["index"] == 4
    physiclaw._bridge.wait_for_connection.assert_called_once()
    # Sets corners then restores bridge.
    calls = phone.set_mode.call_args_list
    assert calls[0].args == ("calibrate",)
    assert calls[0].kwargs == {"phase": "corners"}
    assert calls[-1].args == ("bridge",)
    physiclaw.connect_camera.assert_called_once_with(4)


@pytest.mark.asyncio
async def test_handle_connect_camera_auto_pick_treats_missing_body_as_auto(
    mocker,
) -> None:
    physiclaw = _fake_physiclaw_cam_index(idx=1)
    physiclaw._bridge.wait_for_connection.return_value = True
    phone = MagicMock()
    mocker.patch.object(handler, "_auto_pick_camera_index", return_value=1)
    mocker.patch.object(handler.time, "sleep")

    resp = await handle_connect_camera(
        _fake_request(raise_on_json=True),
        physiclaw,
        phone,
    )

    assert _read_json(resp)["status"] == "ok"


@pytest.mark.asyncio
async def test_handle_connect_camera_auto_pick_fails_when_bridge_not_polling(
    mocker,
) -> None:
    physiclaw = _fake_physiclaw_cam_index()
    physiclaw._bridge.wait_for_connection.return_value = False
    phone = MagicMock()

    resp = await handle_connect_camera(
        _fake_request(json_obj={"index": "auto"}),
        physiclaw,
        phone,
    )

    assert resp.status_code == 500
    body = _read_json(resp)
    assert "auto-pick" in body["message"]
    assert "not polling" in body["message"]
    # Phone never reaches corners phase.
    phone.set_mode.assert_not_called()


@pytest.mark.asyncio
async def test_handle_connect_camera_auto_pick_no_match(mocker) -> None:
    physiclaw = _fake_physiclaw_cam_index()
    physiclaw._bridge.wait_for_connection.return_value = True
    phone = MagicMock()
    mocker.patch.object(handler, "_auto_pick_camera_index", return_value=None)
    mocker.patch.object(handler.time, "sleep")

    resp = await handle_connect_camera(
        _fake_request(json_obj={"index": "auto"}),
        physiclaw,
        phone,
    )

    assert resp.status_code == 500
    body = _read_json(resp)
    assert "no camera with all four RGBM corners" in body["message"]
    # Phone restored to bridge mode even on failure.
    assert phone.set_mode.call_args_list[-1].args == ("bridge",)


@pytest.mark.asyncio
async def test_handle_connect_camera_releases_on_connect_failure(mocker) -> None:
    physiclaw = _fake_physiclaw_cam_index()
    physiclaw.connect_camera.side_effect = RuntimeError("usb error")
    phone = MagicMock()

    resp = await handle_connect_camera(
        _fake_request(json_obj={"index": 0}),
        physiclaw,
        phone,
    )

    assert resp.status_code == 500
    physiclaw.release.assert_called_once()


@pytest.mark.asyncio
async def test_handle_connect_camera_stores_index_in_calibration(mocker) -> None:
    physiclaw = _fake_physiclaw_cam_index(idx=2)
    phone = MagicMock()

    await handle_connect_camera(
        _fake_request(json_obj={"index": 2}),
        physiclaw,
        phone,
    )

    # Stored on the calibration namespace as int.
    assert physiclaw.calibration.cam_index == 2


# ---------- handle_disconnect_camera ----------


@pytest.mark.asyncio
async def test_handle_disconnect_camera_releases_when_connected() -> None:
    physiclaw = MagicMock()
    physiclaw.disconnect_camera.return_value = True

    resp = await handle_disconnect_camera(_fake_request(), physiclaw)

    body = _read_json(resp)
    assert body == {"status": "ok", "released": True}
    physiclaw.disconnect_camera.assert_called_once()
    physiclaw.release.assert_called_once()


@pytest.mark.asyncio
async def test_handle_disconnect_camera_idempotent_when_no_camera() -> None:
    physiclaw = MagicMock()
    physiclaw.disconnect_camera.return_value = False

    resp = await handle_disconnect_camera(_fake_request(), physiclaw)

    body = _read_json(resp)
    assert body == {"status": "ok", "released": False}


@pytest.mark.asyncio
async def test_handle_disconnect_camera_releases_lock_on_failure() -> None:
    physiclaw = MagicMock()
    physiclaw.disconnect_camera.side_effect = RuntimeError("close failed")

    resp = await handle_disconnect_camera(_fake_request(), physiclaw)

    assert resp.status_code == 500
    physiclaw.release.assert_called_once()


# ---------- handle_camera_preview ----------


@pytest.mark.asyncio
async def test_handle_camera_preview_happy_path(mocker) -> None:
    mocker.patch.object(handler, "camera_preview", return_value=b"JPEG-bytes")

    resp = await handle_camera_preview(
        _fake_request(path_params={"index": "3"}, query_params={}),
    )

    body = _read_json(resp)
    assert body["status"] == "ok"
    assert body["index"] == 3
    assert body["image"] == base64.b64encode(b"JPEG-bytes").decode()


@pytest.mark.asyncio
async def test_handle_camera_preview_passes_watermark_query_param(mocker) -> None:
    spy = mocker.patch.object(handler, "camera_preview", return_value=b"x")

    await handle_camera_preview(
        _fake_request(path_params={"index": "5"}, query_params={"watermark": "1"}),
    )

    spy.assert_called_once_with(5, True)


@pytest.mark.asyncio
async def test_handle_camera_preview_default_watermark_is_false(mocker) -> None:
    spy = mocker.patch.object(handler, "camera_preview", return_value=b"x")

    await handle_camera_preview(
        _fake_request(path_params={"index": "0"}),
    )

    spy.assert_called_once_with(0, False)


@pytest.mark.asyncio
async def test_handle_camera_preview_returns_404_on_capture_failure(mocker) -> None:
    mocker.patch.object(
        handler,
        "camera_preview",
        side_effect=RuntimeError("no frame"),
    )

    resp = await handle_camera_preview(
        _fake_request(path_params={"index": "0"}),
    )

    assert resp.status_code == 404
    body = _read_json(resp)
    assert body["status"] == "error"
    assert "no frame" in body["message"]
