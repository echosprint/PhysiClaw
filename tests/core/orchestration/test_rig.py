"""Tests for `physiclaw.core.orchestration.rig` — HardwareRig.

Devices, calibration, the busy lock, lifecycle, and primitive
positioning moves. Gesture dispatch and the observation bracket are
covered in `test_orchestrator.py`; detectors in `test_perception.py`.
Hardware doubles come from this directory's conftest.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from physiclaw.core.orchestration import rig as rig_mod
from physiclaw.core.orchestration.rig import HardwareRig

# ---------- Fixtures ----------


@pytest.fixture
def rig(at_double) -> HardwareRig:
    """Construct a HardwareRig and pre-mock the assistive_touch."""
    r = HardwareRig()
    r._assistive_touch = at_double()
    return r


# ---------- Construction / wiring ----------


def test_init_default_state() -> None:
    r = HardwareRig()

    assert r._arm is None
    assert r._cam is None
    assert r._bridge is None
    assert r._ready is False
    assert r.calibration is not None
    assert r._lock is not None


def test_attach_bridge(bridge_double) -> None:
    r = HardwareRig()
    bridge = bridge_double()

    r.attach_bridge(bridge)

    assert r.bridge is bridge


def test_bridge_property_none_before_assembly() -> None:
    assert HardwareRig().bridge is None


# ---------- ready / hardware_ready ----------


def test_ready_false_until_marked_and_hardware_up(rig, wire_rig) -> None:
    wire_rig(rig)
    rig.mark_ready()

    assert rig.ready is True


def test_ready_false_when_marked_but_hardware_down() -> None:
    r = HardwareRig()
    r.mark_ready()

    assert r.ready is False


def test_hardware_ready_requires_arm_cam_and_transforms(
    rig, arm_double, cam_double
) -> None:
    rig.calibration = MagicMock()
    rig.calibration.transforms_ready = False
    rig._arm = arm_double()
    rig._cam = cam_double()

    assert rig.hardware_ready is False


# ---------- spec'd doubles ----------


def test_arm_double_rejects_nonexistent_methods(rig, wire_rig) -> None:
    # Pins the drift guarantee this suite relies on: the hardware doubles
    # are spec'd, so production code calling a method StylusArm doesn't
    # have (the `frombuffer` rot) fails loudly instead of passing silently.
    wire_rig(rig)

    with pytest.raises(AttributeError):
        rig._arm.frombuffer(b"\x00")


# ---------- status ----------


def test_status_includes_calibration_summary(rig, wire_rig) -> None:
    wire_rig(rig)
    rig._arm.MOVE_DIRECTIONS = None  # alignment not set
    rig._assistive_touch.ready = False

    out = rig.status()

    assert out["arm"] is True
    assert out["camera"] is True
    assert out["bridge"] is False  # no bridge attached
    assert out["calibrated"] is True
    assert out["steps"] == {"step1": "OK"}


def test_status_includes_alignment_when_arm_aligned(rig, wire_rig) -> None:
    wire_rig(rig)
    rig._assistive_touch.ready = False

    out = rig.status()

    assert out["steps"]["alignment"] == "OK"


def test_status_includes_assistive_touch_when_ready(rig, wire_rig) -> None:
    wire_rig(rig)

    out = rig.status()

    assert "assistive_touch" in out["steps"]
    assert "0.050" in out["steps"]["assistive_touch"]


def test_status_includes_bridge_connected(rig, bridge_double) -> None:
    bridge = bridge_double()
    bridge.connected = True
    rig.attach_bridge(bridge)

    out = rig.status()

    assert out["bridge"] is True


def test_status_reports_layout_learned(rig, mocker) -> None:
    mocker.patch(
        "physiclaw.core.orchestration.rig._layout_learned",
        return_value=True,
    )

    assert rig.status()["layout_learned"] is True


def test_status_reports_layout_not_learned(rig, mocker) -> None:
    mocker.patch(
        "physiclaw.core.orchestration.rig._layout_learned",
        return_value=False,
    )

    assert rig.status()["layout_learned"] is False


# ---------- _layout_learned (reads the marker screen_layout writes) ----------


def _write_layout(physiclaw_home, payload: str) -> None:
    from physiclaw.common import paths

    paths.screen_layout_dir().mkdir(parents=True, exist_ok=True)
    paths.screen_layout_json().write_text(payload, encoding="utf-8")


def test_layout_learned_false_when_file_missing(physiclaw_home) -> None:
    assert rig_mod._layout_learned() is False


def test_layout_learned_true_when_marker_set(physiclaw_home) -> None:
    _write_layout(physiclaw_home, '{"layout_learned": true}')

    assert rig_mod._layout_learned() is True


def test_layout_learned_false_when_marker_unset(physiclaw_home) -> None:
    _write_layout(physiclaw_home, '{"send": [0, 0, 1, 1], "layout_learned": false}')

    assert rig_mod._layout_learned() is False


def test_layout_learned_false_when_marker_absent(physiclaw_home) -> None:
    _write_layout(physiclaw_home, '{"send": [0, 0, 1, 1]}')

    assert rig_mod._layout_learned() is False


def test_layout_learned_false_on_malformed_json(physiclaw_home) -> None:
    _write_layout(physiclaw_home, "{ not json")

    assert rig_mod._layout_learned() is False


def test_layout_learned_round_trips_with_screen_layout_record(physiclaw_home) -> None:
    # The agent-side writer and the core-side reader must agree: once
    # `record()` captures the final box, the rig sees learned=True.
    from physiclaw.agent.engine import screen_layout

    assert rig_mod._layout_learned() is False
    for page, field, bbox in [
        ("spotlight", "spotlight_input", [0.1, 0.55, 0.9, 0.65]),
        ("spotlight", "spotlight_paste", [0.3, 0.45, 0.5, 0.5]),
        ("spotlight", "backspace", [0.85, 0.85, 0.95, 0.92]),
        ("spotlight", "return", [0.85, 0.85, 0.95, 0.92]),
        ("spotlight", "space", [0.3, 0.85, 0.7, 0.95]),
        ("chat-no-keyboard", "chat_input_kb_hidden", [0.1, 0.9, 0.7, 0.96]),
        ("chat-keyboard", "chat_input_kb_visible", [0.1, 0.55, 0.7, 0.65]),
        ("chat-keyboard", "send", [0.85, 0.55, 0.98, 0.65]),
        ("chat-keyboard", "chat_paste", [0.3, 0.45, 0.5, 0.5]),
    ]:
        screen_layout.record(page, field, bbox, app="wechat")

    assert rig_mod._layout_learned() is True


# ---------- require_hardware ----------


def test_require_hardware_raises_when_not_ready() -> None:
    r = HardwareRig()

    with pytest.raises(RuntimeError, match="Hardware not set up"):
        r.require_hardware()


def test_require_hardware_passes_when_ready(rig, wire_rig) -> None:
    wire_rig(rig)

    rig.require_hardware()  # no raise


def test_require_arm_and_cam_raise_before_setup() -> None:
    r = HardwareRig()

    with pytest.raises(RuntimeError, match="Arm not connected"):
        r.require_arm()
    with pytest.raises(RuntimeError, match="Camera not connected"):
        r.require_cam()


def test_require_arm_and_cam_return_connected_hardware(rig, wire_rig) -> None:
    wire_rig(rig)

    assert rig.require_arm() is rig.arm
    assert rig.require_cam() is rig.cam


def test_require_assistive_touch_raises_when_not_ready(rig) -> None:
    rig._assistive_touch.ready = False

    with pytest.raises(RuntimeError, match="AssistiveTouch not calibrated"):
        rig.require_assistive_touch()


def test_require_assistive_touch_returns_calibrated(rig) -> None:
    assert rig.require_assistive_touch() is rig._assistive_touch


# ---------- AssistiveTouch operations ----------


def test_take_screenshot_wires_at_with_rig_devices(rig, wire_rig) -> None:
    wire_rig(rig)
    rig._bridge = MagicMock()
    rig._assistive_touch.take_screenshot.return_value = b"PNG"

    rig.acquire()  # AT operations require the hardware lock
    out = rig.take_screenshot(timeout=7.0)
    rig.release()

    assert out == b"PNG"
    rig._assistive_touch.take_screenshot.assert_called_once_with(
        rig._arm, rig._bridge, rig.transforms.pct_to_grbl, timeout=7.0
    )


def test_at_long_press_wires_at_with_rig_devices(rig, wire_rig) -> None:
    wire_rig(rig)

    rig.acquire()
    rig.at_long_press()
    rig.release()

    rig._assistive_touch.long_press.assert_called_once_with(
        rig._arm, rig.transforms.pct_to_grbl
    )


def test_sync_clipboard_timeout_defers_to_expire_text(
    rig, wire_rig, bridge_double
) -> None:
    # The timeout path must NOT clear_text() directly: a late Shortcut
    # fetching between the wait timing out and the clear would receive
    # the text while we report it didn't. expire_text decides under the
    # bridge lock — and "already fetched" counts as a late success.
    wire_rig(rig)
    bridge = bridge_double()
    bridge.wait_clipboard.return_value = False
    bridge.expire_text.return_value = True
    rig.attach_bridge(bridge)

    rig.acquire()
    try:
        assert rig.sync_clipboard("x", timeout=0.01) is True
    finally:
        rig.release()

    bridge.expire_text.assert_called_once_with()
    bridge.clear_text.assert_not_called()


def test_sync_clipboard_timeout_without_fetch_returns_false(
    rig, wire_rig, bridge_double
) -> None:
    wire_rig(rig)
    bridge = bridge_double()
    bridge.wait_clipboard.return_value = False
    bridge.expire_text.return_value = False
    rig.attach_bridge(bridge)

    rig.acquire()
    try:
        assert rig.sync_clipboard("x", timeout=0.01) is False
    finally:
        rig.release()

    bridge.expire_text.assert_called_once_with()


# ---------- assert_locked ----------


def test_assert_locked_raises_when_lock_not_held(rig, wire_rig) -> None:
    """Caller-must-hold-lock methods fail loudly instead of silently
    touching hardware unserialized."""
    wire_rig(rig)

    with pytest.raises(RuntimeError, match="hardware lock not held"):
        rig.move_to_bbox_center([0.0, 0.0, 1.0, 1.0])
    with pytest.raises(RuntimeError, match="hardware lock not held"):
        rig.take_screenshot()
    with pytest.raises(RuntimeError, match="hardware lock not held"):
        rig.at_long_press()
    rig._arm.rapid_to.assert_not_called()
    rig._assistive_touch.take_screenshot.assert_not_called()
    rig._assistive_touch.long_press.assert_not_called()


# ---------- acquire / release / locked ----------


def test_acquire_raises_when_already_held(rig) -> None:
    rig.acquire()
    try:
        with pytest.raises(RuntimeError, match="busy"):
            rig.acquire()
    finally:
        rig.release()


def test_locked_acquires_and_parks_on_exit(rig, wire_rig) -> None:
    wire_rig(rig)
    rig.calibration.pct_to_grbl_mm.return_value = (5.0, 6.0)

    with rig.locked():
        pass

    rig._arm.rapid_to.assert_called_once_with(5.0, 6.0)
    # Lock released — re-acquire should succeed.
    rig.acquire()
    rig.release()


def test_locked_swallows_park_exception(rig, wire_rig) -> None:
    wire_rig(rig)
    rig._arm.rapid_to.side_effect = RuntimeError("arm jammed")

    # park() exception inside locked() must not surface.
    with rig.locked():
        pass
    rig.acquire()  # lock should have been released anyway
    rig.release()


# ---------- connect_arm / connect_camera ----------


def test_connect_arm_closes_existing(mocker, rig, arm_double) -> None:
    old = arm_double()
    rig._arm = old
    new = arm_double()
    new.MOVE_DIRECTIONS = None
    mocker.patch.object(rig_mod, "StylusArm", return_value=new)

    rig.connect_arm()

    old.close.assert_called_once()
    new.setup.assert_called_once()
    assert rig._arm is new


def test_connect_arm_applies_cached_mapping(
    mocker, arm_double, diag_10_20_affine
) -> None:
    r = HardwareRig()
    r.calibration.pct_to_grbl = diag_10_20_affine
    new = arm_double()
    mocker.patch.object(rig_mod, "StylusArm", return_value=new)

    r.connect_arm()

    new.set_direction_mapping.assert_called_once_with((10.0, 0.0), (0.0, 20.0))


def test_restore_park_origin_repins_from_park_spot(
    rig, arm_double, diag_10_20_affine
) -> None:
    # diag(10, 20) affine → park (-0.1, -0.05) maps to GRBL (-1.0, -1.0).
    rig._arm = arm_double()
    rig.calibration.pct_to_grbl = diag_10_20_affine

    assert rig.restore_park_origin() is True
    rig._arm.set_work_position.assert_called_once_with(-1.0, -1.0)


def test_restore_park_origin_noop_without_arm(diag_10_20_affine) -> None:
    r = HardwareRig()
    r.calibration.pct_to_grbl = diag_10_20_affine

    assert r.restore_park_origin() is False  # arm not connected


def test_restore_park_origin_noop_without_calibration(rig, arm_double) -> None:
    rig._arm = arm_double()
    # pct_to_grbl is None on a fresh bundle → no target to re-pin.

    assert rig.restore_park_origin() is False
    rig._arm.set_work_position.assert_not_called()


def test_connect_camera_closes_existing(mocker, cam_double) -> None:
    r = HardwareRig()
    old = cam_double()
    r._cam = old
    new = cam_double()
    mocker.patch.object(rig_mod, "Camera", return_value=new)

    r.connect_camera(2)

    old.close.assert_called_once()
    assert r._cam is new


def test_connect_camera_propagates_rotation(mocker, cam_double) -> None:
    r = HardwareRig()
    r.calibration.cam_rotation = 90
    new = cam_double()
    mocker.patch.object(rig_mod, "Camera", return_value=new)

    r.connect_camera(0)

    assert new.rotation == 90


def test_connect_camera_clears_ready_to_force_a_re_settle(mocker, cam_double) -> None:
    # A fresh Camera (live AF, default exposure) must not inherit the old
    # camera's "ready" — the next become_ready has to re-settle it.
    r = HardwareRig()
    r.mark_ready()
    mocker.patch.object(rig_mod, "Camera", return_value=cam_double())

    r.connect_camera(0)

    assert r._ready is False


def test_disconnect_camera_clears_ready(cam_double) -> None:
    r = HardwareRig()
    r._cam = cam_double()
    r.mark_ready()

    r.disconnect_camera()

    assert r._ready is False


def test_apply_bundle_to_arm_noop_when_no_arm() -> None:
    r = HardwareRig()
    r._apply_bundle_to_arm()  # no raise


# ---------- park ----------


def test_park_noop_when_arm_none() -> None:
    r = HardwareRig()
    r.park()  # no raise


def test_park_noop_when_pct_to_grbl_unset(rig, wire_rig) -> None:
    wire_rig(rig)
    rig.calibration.pct_to_grbl_mm.return_value = None

    rig.park()

    rig._arm.rapid_to.assert_not_called()


def test_park_moves_arm_to_off_screen(rig, wire_rig) -> None:
    wire_rig(rig)

    rig.park()

    rig._arm.rapid_to.assert_called_once_with(5.0, 6.0)
    rig._arm.wait_idle.assert_called_once()


# ---------- move_to_bbox_center ----------


def test_move_to_bbox_center_raises_when_uncalibrated() -> None:
    r = HardwareRig()

    r.acquire()
    with pytest.raises(RuntimeError, match="Screen calibration not done"):
        r.move_to_bbox_center([0.1, 0.1, 0.2, 0.2])
    r.release()


def test_move_to_bbox_center_dispatches(rig, wire_rig) -> None:
    wire_rig(rig)

    rig.acquire()
    rig.move_to_bbox_center([0.0, 0.0, 1.0, 1.0])
    rig.release()

    rig._arm.rapid_to.assert_called_once()


def test_move_to_bbox_center_requires_arm(rig, wire_rig) -> None:
    # Calibrated transforms but no arm — the honest accessor must raise,
    # not AttributeError on None.
    wire_rig(rig)
    rig._arm = None

    rig.acquire()
    with pytest.raises(RuntimeError, match="Arm not connected"):
        rig.move_to_bbox_center([0.0, 0.0, 1.0, 1.0])
    rig.release()


# ---------- accessor properties ----------


def test_arm_and_assistive_touch_properties(rig, arm_double) -> None:
    rig._arm = arm_double()

    assert rig.arm is rig._arm
    assert rig.assistive_touch is rig._assistive_touch


# ---------- shutdown ----------


def test_shutdown_closes_arm_and_camera(rig, arm_double, cam_double) -> None:
    # Uncalibrated fixture (pct_to_grbl is None) → park has no target, so
    # teardown falls back to homing.
    rig._arm = arm_double()
    rig._cam = cam_double()

    rig.shutdown()

    rig._arm.lift_stylus.assert_called_once()
    rig._arm.return_to_origin.assert_called_once()
    rig._arm.close.assert_called_once()
    rig._cam.close.assert_called_once()


def test_shutdown_parks_off_screen_when_calibrated(rig, arm_double, cam_double) -> None:
    # With calibration loaded, teardown rests the tip at the same off-screen
    # park spot used between taps — not the machine origin — so the phone
    # stays clear for placement / removal.
    rig._arm = arm_double()
    rig._cam = cam_double()
    rig.calibration.pct_to_grbl = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

    rig.shutdown()

    rig._arm.lift_stylus.assert_called_once()
    rig._arm.return_to_origin.assert_not_called()
    # park() drives a fast move to the calibrated park coordinate.
    rig._arm.rapid_to.assert_called_once_with(-0.1, -0.05)
    rig._arm.close.assert_called_once()
    rig._cam.close.assert_called_once()


def test_shutdown_handles_no_hardware() -> None:
    r = HardwareRig()

    r.shutdown()  # no raise


def test_shutdown_continues_when_coil_release_fails(
    rig, arm_double, cam_double
) -> None:
    # A failed stylus lift must not strand the serial/camera handles.
    rig._arm = arm_double()
    rig._arm.lift_stylus.side_effect = RuntimeError("serial timeout")
    rig._cam = cam_double()

    rig.shutdown()  # swallows the error

    rig._arm.return_to_origin.assert_called_once()
    rig._arm.close.assert_called_once()
    rig._cam.close.assert_called_once()


def test_shutdown_continues_when_arm_close_fails(rig, arm_double, cam_double) -> None:
    # Camera must still close even if the arm teardown raises.
    rig._arm = arm_double()
    rig._arm.return_to_origin.side_effect = RuntimeError("GRBL alarm")
    rig._arm.close.side_effect = RuntimeError("port gone")
    rig._cam = cam_double()

    rig.shutdown()

    rig._arm.close.assert_called_once()  # attempted despite the prior failure
    rig._cam.close.assert_called_once()


def test_shutdown_swallows_camera_close_failure(rig, cam_double) -> None:
    rig._cam = cam_double()
    rig._cam.close.side_effect = RuntimeError("camera busy")

    rig.shutdown()  # no raise
