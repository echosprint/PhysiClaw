"""Tests for `physiclaw.core.calibration.handler` — calibration HTTP routes."""

from __future__ import annotations

import dataclasses
import json
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

from physiclaw.core.calibration import handler
from physiclaw.core.calibration.handler import (
    _err,
    _ok,
    _read_body,
    _run_blocking,
    handle_calibrate_arm,
    handle_calibrate_camera_frame,
    handle_compute_camera_mapping,
    handle_measure_viewport_shift,
    handle_show_assistive_touch,
    handle_trace_edge,
    handle_validate_calibration,
    handle_verify_assistive_touch,
)
from physiclaw.core.calibration.transforms import ViewportShift
from tests.core.conftest import wire_locked


def _async(value: Any):
    async def _coro():
        return value

    return _coro


def _async_raise(exc: Exception):
    async def _coro():
        raise exc

    return _coro


def _rig_mock() -> MagicMock:
    """A MagicMock rig whose ``locked()`` delegates to ``acquire()`` /
    ``release()`` (see ``wire_locked``), so the handler tests assert the real
    lock bracket (acquire → park-on-failure → release) rather than an inert
    context-manager mock."""
    return wire_locked(MagicMock(name="rig"))


def _fake_request(json_obj: Any = None, raise_on_json: bool = False):
    req = SimpleNamespace()
    if raise_on_json:
        req.json = _async_raise(RuntimeError("bad body"))
    else:
        req.json = _async(json_obj)
    return req


def _read_json(response) -> dict:
    return json.loads(bytes(response.body).decode())


# ---------- helpers ----------


def test_ok_wraps_payload() -> None:
    resp = _ok({"a": 1})

    assert _read_json(resp) == {"status": "ok", "a": 1}


def test_err_default_500() -> None:
    resp = _err("boom")

    assert resp.status_code == 500
    assert _read_json(resp) == {"status": "error", "message": "boom"}


def test_err_custom_status_code() -> None:
    resp = _err("bad input", status_code=400)

    assert resp.status_code == 400


@pytest.mark.asyncio
async def test_read_body_parses_json() -> None:
    out = await _read_body(_fake_request(json_obj={"x": 1}))

    assert out == {"x": 1}


@pytest.mark.asyncio
async def test_read_body_returns_empty_dict_on_failure() -> None:
    out = await _read_body(_fake_request(raise_on_json=True))

    assert out == {}


@pytest.mark.asyncio
async def test_run_blocking_runs_sync_callable() -> None:
    out = await _run_blocking(lambda: "result")

    assert out == "result"


# ---------- handle_measure_viewport_shift ----------


def _viewport_shift() -> ViewportShift:
    return ViewportShift(
        offset_x=0.0,
        offset_y=0.0,
        dpr=3.0,
        screenshot_width=1170,
        screenshot_height=2532,
    )


@pytest.mark.asyncio
async def test_handle_measure_viewport_shift_happy_path(mocker) -> None:
    rig = _rig_mock()
    rig.calibration = SimpleNamespace()
    calib = MagicMock()
    bridge = MagicMock()
    phone = MagicMock()
    vs = _viewport_shift()
    spy = mocker.patch.object(handler, "measure_viewport_shift", return_value=vs)

    resp = await handle_measure_viewport_shift(
        _fake_request(json_obj={"fresh": True}),
        rig,
        calib,
        bridge,
        phone,
    )

    body = _read_json(resp)
    assert body["status"] == "ok"
    assert body == {"status": "ok", **dataclasses.asdict(vs)}
    spy.assert_called_once_with(calib, bridge, fresh=True)
    phone.set_mode.assert_called_once_with("calibrate", phase="screenshot_cal")
    assert rig.calibration.viewport_shift is vs


@pytest.mark.asyncio
async def test_handle_measure_viewport_shift_default_fresh_false(mocker) -> None:
    rig = _rig_mock()
    calib = MagicMock()
    bridge = MagicMock()
    phone = MagicMock()
    vs = _viewport_shift()
    spy = mocker.patch.object(handler, "measure_viewport_shift", return_value=vs)

    await handle_measure_viewport_shift(
        _fake_request(raise_on_json=True),
        rig,
        calib,
        bridge,
        phone,
    )

    spy.assert_called_once_with(calib, bridge, fresh=False)


@pytest.mark.asyncio
async def test_handle_measure_viewport_shift_returns_err_on_exception(mocker) -> None:
    rig = _rig_mock()
    calib = MagicMock()
    mocker.patch.object(
        handler,
        "measure_viewport_shift",
        side_effect=RuntimeError("no upload"),
    )

    resp = await handle_measure_viewport_shift(
        _fake_request(json_obj={}),
        rig,
        calib,
        MagicMock(),
        MagicMock(),
    )

    assert resp.status_code == 500
    assert "no upload" in _read_json(resp)["message"]


# ---------- handle_calibrate_arm ----------


def _identity_pct_to_grbl() -> np.ndarray:
    return np.array(
        [
            [10.0, 0.0, 0.0],
            [0.0, 20.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )


@pytest.mark.asyncio
async def test_handle_calibrate_arm_happy_path(mocker) -> None:
    rig = _rig_mock()
    rig.arm = MagicMock()
    rig.calibration = SimpleNamespace(pct_to_grbl=None)
    calib = MagicMock()
    phone = MagicMock()
    pct_to_grbl = _identity_pct_to_grbl()
    mocker.patch.object(
        handler,
        "calibrate_arm",
        return_value=(pct_to_grbl, 0.01, [(0.1, 0.1, 0)]),
    )
    mocker.patch.object(handler, "TILT_ALIGNED_THRESHOLD", 0.05)

    resp = await handle_calibrate_arm(
        _fake_request(json_obj={}),
        rig,
        calib,
        phone,
    )

    body = _read_json(resp)
    assert body["status"] == "ok"
    assert body["pairs"] == 4  # 1 touch + 3 probe
    assert body["aligned"] is True
    # The affine lands through the one installer (which wires the arm's
    # direction mapping and pins the frame — pinned by test_rig).
    rig.install_arm_calibration.assert_called_once_with(pct_to_grbl)
    rig.park.assert_called_once()
    rig.release.assert_called_once()


@pytest.mark.asyncio
async def test_handle_calibrate_arm_from_park_centers_stylus(mocker) -> None:
    """Auto mode (from_park): with a prior mapping, drive the parked stylus to
    center before calibrating."""
    rig = _rig_mock()
    rig.arm = MagicMock()
    rig.calibration = SimpleNamespace(
        pct_to_grbl=_identity_pct_to_grbl(),
        pct_to_grbl_mm=lambda x, y: (5.0, 6.0),
    )
    mocker.patch.object(
        handler,
        "calibrate_arm",
        return_value=(_identity_pct_to_grbl(), 0.01, []),
    )
    mocker.patch.object(handler, "TILT_ALIGNED_THRESHOLD", 0.05)

    resp = await handle_calibrate_arm(
        _fake_request(json_obj={"from_park": True}),
        rig,
        MagicMock(),
        MagicMock(),
    )

    assert _read_json(resp)["status"] == "ok"
    rig.restore_park_origin.assert_called_once()
    rig.arm.rapid_to.assert_called_once_with(5.0, 6.0)  # screen center
    # No set_origin: the affine is center-rebased so the tip already rests
    # AT the frame zero, and a G92 here would silently void the pin that
    # restore_park_origin just earned — the probe must run pinned so a
    # mid-probe failure park still lands at the true park spot.
    rig.arm.set_origin.assert_not_called()


@pytest.mark.asyncio
async def test_handle_calibrate_arm_from_park_without_bundle_errors(mocker) -> None:
    """from_park with no in-memory mapping and no saved bundle → clear error."""
    rig = _rig_mock()
    rig.arm = MagicMock()
    rig.calibration = SimpleNamespace(pct_to_grbl=None)
    mocker.patch.object(handler.Calibration, "load", return_value=None)
    cal_arm = mocker.patch.object(handler, "calibrate_arm")

    resp = await handle_calibrate_arm(
        _fake_request(json_obj={"from_park": True}),
        rig,
        MagicMock(),
        MagicMock(),
    )

    body = _read_json(resp)
    assert body["status"] == "error"
    assert "previous calibration" in body["message"]
    cal_arm.assert_not_called()  # never reached the probe
    rig.release.assert_called_once()  # lock still released


@pytest.mark.asyncio
async def test_handle_calibrate_arm_arm_not_connected() -> None:
    rig = _rig_mock()
    rig.arm = None

    resp = await handle_calibrate_arm(
        _fake_request(json_obj={}),
        rig,
        MagicMock(),
        MagicMock(),
    )

    # Precondition failures are client state, not server faults.
    assert resp.status_code == 409
    assert "Arm not connected" in _read_json(resp)["message"]


@pytest.mark.asyncio
async def test_handle_calibrate_arm_releases_on_failure(mocker) -> None:
    rig = _rig_mock()
    rig.arm = MagicMock()
    rig.calibration = SimpleNamespace(pct_to_grbl=None)
    mocker.patch.object(
        handler,
        "calibrate_arm",
        side_effect=RuntimeError("probe failed"),
    )

    resp = await handle_calibrate_arm(
        _fake_request(json_obj={}),
        rig,
        MagicMock(),
        MagicMock(),
    )

    assert resp.status_code == 500
    rig.release.assert_called_once()


# ---------- handle_calibrate_camera_frame ----------


@pytest.mark.asyncio
async def test_handle_calibrate_camera_frame_happy_path(mocker) -> None:
    rig = _rig_mock()
    rig.cam = MagicMock()
    rig.calibration = SimpleNamespace()
    mocker.patch.object(
        handler,
        "calibrate_camera_frame",
        return_value={"rotation": 90, "issues": []},
    )

    resp = await handle_calibrate_camera_frame(
        _fake_request(json_obj={}),
        rig,
        MagicMock(),
    )

    body = _read_json(resp)
    assert body["rotation"] == 90
    assert rig.calibration.cam_rotation == 90
    assert rig.cam.rotation == 90
    rig.park.assert_called_once()
    rig.release.assert_called_once()


@pytest.mark.asyncio
async def test_handle_calibrate_camera_frame_camera_not_connected() -> None:
    rig = _rig_mock()
    rig.cam = None

    resp = await handle_calibrate_camera_frame(
        _fake_request(),
        rig,
        MagicMock(),
    )

    # Precondition failures are client state, not server faults.
    assert resp.status_code == 409
    assert "Camera not connected" in _read_json(resp)["message"]


@pytest.mark.asyncio
async def test_handle_calibrate_camera_frame_releases_on_failure(mocker) -> None:
    rig = _rig_mock()
    rig.cam = MagicMock()
    mocker.patch.object(
        handler,
        "calibrate_camera_frame",
        side_effect=RuntimeError("no markers"),
    )

    resp = await handle_calibrate_camera_frame(
        _fake_request(),
        rig,
        MagicMock(),
    )

    assert resp.status_code == 500
    rig.release.assert_called_once()


# ---------- handle_compute_camera_mapping ----------


@pytest.mark.asyncio
async def test_handle_compute_camera_mapping_happy_path(mocker) -> None:
    rig = _rig_mock()
    rig.cam = MagicMock()
    rig.calibration = SimpleNamespace(effective_rotation=lambda: 90)
    pct_to_cam = np.eye(3)
    mocker.patch.object(
        handler,
        "compute_camera_mapping",
        return_value=(pct_to_cam, (1920, 1080), 137.0),
    )

    resp = await handle_compute_camera_mapping(
        _fake_request(),
        rig,
        MagicMock(),
    )

    body = _read_json(resp)
    assert body["ok"] is True
    assert body["dots"] == 15
    assert body["cam_size"] == [1920, 1080]
    assert body["cam_focus"] == 137.0
    assert rig.calibration.pct_to_cam is pct_to_cam
    assert rig.calibration.cam_size == (1920, 1080)
    assert rig.calibration.cam_focus == 137.0


@pytest.mark.asyncio
async def test_handle_compute_camera_mapping_camera_not_connected() -> None:
    rig = _rig_mock()
    rig.cam = None

    resp = await handle_compute_camera_mapping(
        _fake_request(),
        rig,
        MagicMock(),
    )

    # Precondition failures are client state, not server faults.
    assert resp.status_code == 409
    assert "Camera not connected" in _read_json(resp)["message"]


@pytest.mark.asyncio
async def test_handle_compute_camera_mapping_releases_on_failure(mocker) -> None:
    rig = _rig_mock()
    rig.cam = MagicMock()
    rig.calibration = SimpleNamespace(effective_rotation=lambda: 0)
    mocker.patch.object(
        handler,
        "compute_camera_mapping",
        side_effect=RuntimeError("dots missing"),
    )

    resp = await handle_compute_camera_mapping(
        _fake_request(),
        rig,
        MagicMock(),
    )

    assert resp.status_code == 500
    rig.release.assert_called_once()


# ---------- handle_validate_calibration ----------


@pytest.mark.asyncio
async def test_handle_validate_calibration_happy_path_and_persists(mocker) -> None:
    rig = _rig_mock()
    rig.arm = MagicMock()
    cal = MagicMock()
    cal.transforms_ready = True
    cal.pct_to_grbl = _identity_pct_to_grbl()
    cal.pct_to_cam = np.eye(3)
    cal.cam_size = (1920, 1080)
    cal.effective_rotation.return_value = 0
    rig.calibration = cal
    calib = MagicMock()
    calib.screen_dimension = (390, 844)
    phone = MagicMock()

    results = [{"passed": True}, {"passed": True}, {"passed": False}]
    mocker.patch.object(handler, "validate_calibration", return_value=results)

    resp = await handle_validate_calibration(
        _fake_request(),
        rig,
        calib,
        phone,
    )

    body = _read_json(resp)
    assert body["passed"] == 2
    assert body["total"] == 3
    assert body["calibrated"] is True
    phone.set_mode.assert_called_once_with("bridge")
    cal.save.assert_called_once()
    assert cal.screen_dimension == (390, 844)


@pytest.mark.asyncio
async def test_handle_validate_calibration_does_not_save_when_not_calibrated(
    mocker,
) -> None:
    rig = _rig_mock()
    rig.arm = MagicMock()
    cal = MagicMock()
    cal.transforms_ready = True
    cal.pct_to_grbl = _identity_pct_to_grbl()
    cal.pct_to_cam = np.eye(3)
    cal.cam_size = (1920, 1080)
    cal.effective_rotation.return_value = 0
    rig.calibration = cal
    phone = MagicMock()
    mocker.patch.object(
        handler,
        "validate_calibration",
        return_value=[{"passed": False}, {"passed": True}],
    )

    resp = await handle_validate_calibration(
        _fake_request(),
        rig,
        MagicMock(),
        phone,
    )

    body = _read_json(resp)
    assert body["calibrated"] is False
    cal.save.assert_not_called()
    phone.set_mode.assert_not_called()


@pytest.mark.asyncio
async def test_handle_validate_calibration_arm_not_connected() -> None:
    rig = _rig_mock()
    rig.arm = None

    resp = await handle_validate_calibration(
        _fake_request(),
        rig,
        MagicMock(),
        MagicMock(),
    )

    # Precondition failures are client state, not server faults.
    assert resp.status_code == 409
    assert "Arm not connected" in _read_json(resp)["message"]


@pytest.mark.asyncio
async def test_handle_validate_calibration_requires_transforms_ready() -> None:
    rig = _rig_mock()
    rig.arm = MagicMock()
    cal = MagicMock()
    cal.transforms_ready = False
    rig.calibration = cal

    resp = await handle_validate_calibration(
        _fake_request(),
        rig,
        MagicMock(),
        MagicMock(),
    )

    # Precondition failures are client state, not server faults.
    assert resp.status_code == 409
    assert "arm calibration" in _read_json(resp)["message"]


# ---------- handle_trace_edge ----------


@pytest.mark.asyncio
async def test_handle_trace_edge_happy_path(mocker) -> None:
    rig = _rig_mock()
    rig.transforms = MagicMock()  # truthy → calibrated
    phone = MagicMock()
    spy = mocker.patch(
        "physiclaw.core.calibration.calibrate.trace_screen_edge",
    )

    resp = await handle_trace_edge(_fake_request(), rig, phone)

    assert _read_json(resp) == {"status": "ok", "ok": True}
    phone.set_mode.assert_called_once_with("bridge")
    rig.park.assert_called_once()
    spy.assert_called_once()


@pytest.mark.asyncio
async def test_handle_trace_edge_uncalibrated_returns_error() -> None:
    rig = _rig_mock()
    rig.transforms = None

    resp = await handle_trace_edge(_fake_request(), rig, MagicMock())

    # Precondition failures are client state, not server faults.
    assert resp.status_code == 409
    assert "Not calibrated" in _read_json(resp)["message"]


@pytest.mark.asyncio
async def test_handle_trace_edge_releases_on_failure(mocker) -> None:
    rig = _rig_mock()
    rig.transforms = MagicMock()
    mocker.patch(
        "physiclaw.core.calibration.calibrate.trace_screen_edge",
        side_effect=RuntimeError("arm off"),
    )

    resp = await handle_trace_edge(_fake_request(), rig, MagicMock())

    assert resp.status_code == 500
    rig.release.assert_called_once()


# ---------- handle_show_assistive_touch ----------


@pytest.mark.asyncio
async def test_handle_show_assistive_touch_happy_path(mocker) -> None:
    rig = _rig_mock()
    rig.assistive_touch.at_screen = (0.1, 0.2)
    calib = SimpleNamespace(viewport_shift=_viewport_shift())
    phone = MagicMock()
    mocker.patch.object(handler, "generate_nonce", return_value=[1, 0, 1])

    resp = await handle_show_assistive_touch(
        _fake_request(),
        rig,
        calib,
        phone,
    )

    body = _read_json(resp)
    assert body["status"] == "ok"
    assert body["at_screen"] == [0.1, 0.2]
    assert body["nonce_count"] == 3
    rig.assistive_touch.compute_at_screen_pos.assert_called_once_with(
        calib.viewport_shift,
    )
    phone.set_mode.assert_called_once_with(
        "calibrate",
        phase="assistive_touch",
        nonce_bits=[1, 0, 1],
    )


@pytest.mark.asyncio
async def test_handle_show_assistive_touch_requires_viewport_shift() -> None:
    rig = _rig_mock()
    calib = SimpleNamespace(viewport_shift=None)

    resp = await handle_show_assistive_touch(
        _fake_request(),
        rig,
        calib,
        MagicMock(),
    )

    # Precondition failures are client state (409) — same contract as
    # every other calibration precheck.
    assert resp.status_code == 409
    assert "Viewport shift not measured" in _read_json(resp)["message"]


# ---------- handle_verify_assistive_touch ----------


@pytest.mark.asyncio
async def test_handle_verify_assistive_touch_happy_path(mocker) -> None:
    rig = _rig_mock()
    rig.arm = MagicMock()
    rig.calibration = SimpleNamespace(pct_to_grbl=_identity_pct_to_grbl())
    rig.assistive_touch.at_screen = (0.1, 0.2)
    spy = mocker.patch.object(
        handler,
        "verify_assistive_touch",
        return_value={"verified": True},
    )

    resp = await handle_verify_assistive_touch(
        _fake_request(),
        rig,
        MagicMock(),
        MagicMock(),
        MagicMock(),
    )

    assert _read_json(resp) == {"status": "ok", "verified": True}
    spy.assert_called_once()
    rig.release.assert_called_once()


@pytest.mark.asyncio
async def test_handle_verify_assistive_touch_arm_not_connected() -> None:
    rig = _rig_mock()
    rig.arm = None

    resp = await handle_verify_assistive_touch(
        _fake_request(),
        rig,
        MagicMock(),
        MagicMock(),
        MagicMock(),
    )

    # Precondition failures are client state, not server faults.
    assert resp.status_code == 409
    assert "Arm not connected" in _read_json(resp)["message"]


@pytest.mark.asyncio
async def test_handle_verify_assistive_touch_requires_pct_to_grbl() -> None:
    rig = _rig_mock()
    rig.arm = MagicMock()
    rig.calibration = SimpleNamespace(pct_to_grbl=None)

    resp = await handle_verify_assistive_touch(
        _fake_request(),
        rig,
        MagicMock(),
        MagicMock(),
        MagicMock(),
    )

    # Precondition failures are client state, not server faults.
    assert resp.status_code == 409
    assert "arm calibration" in _read_json(resp)["message"]


@pytest.mark.asyncio
async def test_handle_verify_assistive_touch_requires_at_show() -> None:
    rig = _rig_mock()
    rig.arm = MagicMock()
    rig.calibration = SimpleNamespace(pct_to_grbl=_identity_pct_to_grbl())
    rig.assistive_touch.at_screen = None  # show step not run

    resp = await handle_verify_assistive_touch(
        _fake_request(),
        rig,
        MagicMock(),
        MagicMock(),
        MagicMock(),
    )

    # Precondition failures are client state, not server faults.
    assert resp.status_code == 409
    assert "assistive-touch/show" in _read_json(resp)["message"]
