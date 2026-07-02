"""Tests for `physiclaw.core.orchestration.orchestrator` — PhysiClaw class."""
from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from physiclaw.core.orchestration import orchestrator
from physiclaw.core.orchestration.orchestrator import PhysiClaw


# ---------- Fixtures ----------


def _identity_pct_to_grbl() -> np.ndarray:
    return np.array([
        [10.0, 0.0, 0.0],
        [0.0, 20.0, 0.0],
        [0.0, 0.0, 1.0],
    ])


def _fake_transforms(*, swipe_end=(0.5, 0.6)):
    t = MagicMock()
    t.bbox_center_pct.side_effect = lambda bbox: (
        (bbox[0] + bbox[2]) / 2,
        (bbox[1] + bbox[3]) / 2,
    )
    t.pct_to_grbl_mm.side_effect = lambda x, y: (x * 10, y * 20)
    t.swipe_end_pct.return_value = swipe_end
    t.pct_to_grbl = _identity_pct_to_grbl()
    return t


@pytest.fixture
def pc(mocker) -> PhysiClaw:
    """Construct PhysiClaw and pre-mock the assistive_touch + watchdog."""
    p = PhysiClaw()
    p._assistive_touch = MagicMock()
    p._assistive_touch.ready = True
    p._assistive_touch.at_screen = (0.05, 0.1)
    p._assistive_touch.overlaps_at.return_value = False
    p._assistive_touch.swipe_crosses_at.return_value = False
    p._watchdog = MagicMock()
    return p


def _wire_hardware(pc: PhysiClaw, *, transforms=None):
    pc._arm = MagicMock()
    pc._arm.MOVE_DIRECTIONS = {"up": "x"}
    pc._arm.SWIPE_SPEEDS = {"slow": 100, "medium": 500, "fast": 1500}
    pc._cam = MagicMock()
    t = transforms or _fake_transforms()
    pc.calibration = MagicMock()
    pc.calibration.transforms_ready = True
    pc.calibration.transforms.return_value = t
    pc.calibration.summary.return_value = {"step1": "OK"}
    pc.calibration.cam_rotation = None
    pc.calibration.pct_to_grbl = None
    pc.calibration.pct_to_grbl_mm.return_value = (5.0, 6.0)


# ---------- Construction / wiring ----------


def test_init_default_state() -> None:
    p = PhysiClaw()

    assert p._arm is None
    assert p._cam is None
    assert p._bridge is None
    assert p._ocr_reader is None
    assert p._icon_detector is None
    assert p._ready is False
    assert p.calibration is not None
    assert p._lock is not None


def test_attach_bridge() -> None:
    p = PhysiClaw()
    bridge = MagicMock()

    p.attach_bridge(bridge)

    assert p._bridge is bridge


# ---------- ready / hardware_ready ----------


def test_ready_false_until_marked_and_hardware_up(pc: PhysiClaw) -> None:
    _wire_hardware(pc)
    pc.mark_ready()

    assert pc.ready is True


def test_ready_false_when_marked_but_hardware_down() -> None:
    p = PhysiClaw()
    p.mark_ready()

    assert p.ready is False


def test_hardware_ready_requires_arm_cam_and_transforms(pc: PhysiClaw) -> None:
    pc.calibration = MagicMock()
    pc.calibration.transforms_ready = False
    pc._arm = MagicMock()
    pc._cam = MagicMock()

    assert pc.hardware_ready is False


# ---------- status ----------


def test_status_includes_calibration_summary(pc: PhysiClaw) -> None:
    _wire_hardware(pc)
    pc._arm.MOVE_DIRECTIONS = None  # alignment not set
    pc._assistive_touch.ready = False

    out = pc.status()

    assert out["arm"] is True
    assert out["camera"] is True
    assert out["bridge"] is False  # no bridge attached
    assert out["calibrated"] is True
    assert out["steps"] == {"step1": "OK"}


def test_status_includes_alignment_when_arm_aligned(pc: PhysiClaw) -> None:
    _wire_hardware(pc)
    pc._assistive_touch.ready = False

    out = pc.status()

    assert out["steps"]["alignment"] == "OK"


def test_status_includes_assistive_touch_when_ready(pc: PhysiClaw) -> None:
    _wire_hardware(pc)

    out = pc.status()

    assert "assistive_touch" in out["steps"]
    assert "0.050" in out["steps"]["assistive_touch"]


def test_status_includes_bridge_connected(pc: PhysiClaw) -> None:
    bridge = MagicMock()
    bridge.connected = True
    pc.attach_bridge(bridge)

    out = pc.status()

    assert out["bridge"] is True


def test_status_reports_layout_learned(pc: PhysiClaw, mocker) -> None:
    mocker.patch(
        "physiclaw.core.orchestration.orchestrator._layout_learned",
        return_value=True,
    )

    assert pc.status()["layout_learned"] is True


def test_status_reports_layout_not_learned(pc: PhysiClaw, mocker) -> None:
    mocker.patch(
        "physiclaw.core.orchestration.orchestrator._layout_learned",
        return_value=False,
    )

    assert pc.status()["layout_learned"] is False


# ---------- _layout_learned (reads the marker screen_layout writes) ----------


def _write_layout(physiclaw_home, payload: str) -> None:
    from physiclaw import paths

    paths.screen_layout_dir().mkdir(parents=True, exist_ok=True)
    paths.screen_layout_json().write_text(payload, encoding="utf-8")


def test_layout_learned_false_when_file_missing(physiclaw_home) -> None:
    assert orchestrator._layout_learned() is False


def test_layout_learned_true_when_marker_set(physiclaw_home) -> None:
    _write_layout(physiclaw_home, '{"layout_learned": true}')

    assert orchestrator._layout_learned() is True


def test_layout_learned_false_when_marker_unset(physiclaw_home) -> None:
    _write_layout(physiclaw_home, '{"send": [0, 0, 1, 1], "layout_learned": false}')

    assert orchestrator._layout_learned() is False


def test_layout_learned_false_when_marker_absent(physiclaw_home) -> None:
    _write_layout(physiclaw_home, '{"send": [0, 0, 1, 1]}')

    assert orchestrator._layout_learned() is False


def test_layout_learned_false_on_malformed_json(physiclaw_home) -> None:
    _write_layout(physiclaw_home, "{ not json")

    assert orchestrator._layout_learned() is False


def test_layout_learned_round_trips_with_screen_layout_record(physiclaw_home) -> None:
    # The agent-side writer and the core-side reader must agree: once
    # `record()` captures the final box, the orchestrator sees learned=True.
    from physiclaw.agent.engine import screen_layout

    assert orchestrator._layout_learned() is False
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

    assert orchestrator._layout_learned() is True


# ---------- require_hardware ----------


def test_require_hardware_raises_when_not_ready() -> None:
    p = PhysiClaw()

    with pytest.raises(RuntimeError, match="Hardware not set up"):
        p.require_hardware()


def test_require_hardware_passes_when_ready(pc: PhysiClaw) -> None:
    _wire_hardware(pc)

    pc.require_hardware()  # no raise


# ---------- acquire / release / locked ----------


def test_acquire_raises_when_already_held(pc: PhysiClaw) -> None:
    pc.acquire()
    try:
        with pytest.raises(RuntimeError, match="busy"):
            pc.acquire()
    finally:
        pc.release()


def test_locked_acquires_and_parks_on_exit(pc: PhysiClaw) -> None:
    _wire_hardware(pc)
    pc.calibration.pct_to_grbl_mm.return_value = (5.0, 6.0)

    with pc.locked():
        pass

    pc._arm._fast_move.assert_called_once_with(5.0, 6.0)
    # Lock released — re-acquire should succeed.
    pc.acquire()
    pc.release()


def test_locked_swallows_park_exception(pc: PhysiClaw) -> None:
    _wire_hardware(pc)
    pc._arm._fast_move.side_effect = RuntimeError("arm jammed")

    # park() exception inside locked() must not surface.
    with pc.locked():
        pass
    pc.acquire()  # lock should have been released anyway
    pc.release()


# ---------- watch ----------


def test_watch_returns_no_wake_when_frame_none(pc: PhysiClaw) -> None:
    _wire_hardware(pc)
    pc._cam.peek.return_value = None

    out = pc.watch()

    assert out == {"wake": False, "reason": ""}


def test_watch_polls_watchdog_with_frame(pc: PhysiClaw) -> None:
    _wire_hardware(pc)
    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    pc._cam.peek.return_value = frame
    pc._watchdog.poll.return_value = {"wake": True, "reason": "screen change"}

    out = pc.watch()

    assert out == {"wake": True, "reason": "screen change"}
    pc._watchdog.poll.assert_called_once()


# ---------- connect_arm / connect_camera ----------


def test_connect_arm_closes_existing(mocker, pc: PhysiClaw) -> None:
    old = MagicMock()
    pc._arm = old
    new = MagicMock()
    new.MOVE_DIRECTIONS = None
    mocker.patch.object(orchestrator, "StylusArm", return_value=new)

    pc.connect_arm()

    old.close.assert_called_once()
    new.setup.assert_called_once()
    assert pc._arm is new


def test_connect_arm_applies_cached_mapping(mocker) -> None:
    p = PhysiClaw()
    p.calibration.pct_to_grbl = _identity_pct_to_grbl()
    new = MagicMock()
    mocker.patch.object(orchestrator, "StylusArm", return_value=new)

    p.connect_arm()

    new.set_direction_mapping.assert_called_once_with((10.0, 0.0), (0.0, 20.0))


def test_restore_park_origin_repins_from_park_spot(pc: PhysiClaw) -> None:
    # diag(10, 20) affine → park (-0.1, -0.05) maps to GRBL (-1.0, -1.0).
    pc._arm = MagicMock()
    pc.calibration.pct_to_grbl = _identity_pct_to_grbl()

    assert pc.restore_park_origin() is True
    pc._arm.set_work_position.assert_called_once_with(-1.0, -1.0)


def test_restore_park_origin_noop_without_arm() -> None:
    p = PhysiClaw()
    p.calibration.pct_to_grbl = _identity_pct_to_grbl()

    assert p.restore_park_origin() is False  # arm not connected


def test_restore_park_origin_noop_without_calibration(pc: PhysiClaw) -> None:
    pc._arm = MagicMock()
    # pct_to_grbl is None on a fresh bundle → no target to re-pin.

    assert pc.restore_park_origin() is False
    pc._arm.set_work_position.assert_not_called()


def test_connect_camera_closes_existing(mocker) -> None:
    p = PhysiClaw()
    old = MagicMock()
    p._cam = old
    new = MagicMock()
    mocker.patch.object(orchestrator, "Camera", return_value=new)

    p.connect_camera(2)

    old.close.assert_called_once()
    assert p._cam is new


def test_connect_camera_propagates_rotation(mocker) -> None:
    p = PhysiClaw()
    p.calibration.cam_rotation = 90
    new = MagicMock()
    new.rotation = None
    mocker.patch.object(orchestrator, "Camera", return_value=new)

    p.connect_camera(0)

    assert new.rotation == 90


def test_apply_bundle_to_arm_noop_when_no_arm() -> None:
    p = PhysiClaw()
    p._apply_bundle_to_arm()  # no raise


# ---------- park ----------


def test_park_noop_when_arm_none() -> None:
    p = PhysiClaw()
    p.park()  # no raise


def test_park_noop_when_pct_to_grbl_unset(pc: PhysiClaw) -> None:
    _wire_hardware(pc)
    pc.calibration.pct_to_grbl_mm.return_value = None

    pc.park()

    pc._arm._fast_move.assert_not_called()


def test_park_moves_arm_to_off_screen(pc: PhysiClaw) -> None:
    _wire_hardware(pc)

    pc.park()

    pc._arm._fast_move.assert_called_once_with(5.0, 6.0)
    pc._arm.wait_idle.assert_called_once()


# ---------- camera_view ----------


def test_camera_view_raises_when_snapshot_none(pc: PhysiClaw) -> None:
    _wire_hardware(pc)
    pc._cam.snapshot.return_value = None

    with pytest.raises(RuntimeError, match="Camera capture failed"):
        pc.camera_view()


def test_camera_view_returns_frame(pc: PhysiClaw) -> None:
    _wire_hardware(pc)
    frame = np.zeros((1, 1, 3), dtype=np.uint8)
    pc._cam.snapshot.return_value = frame

    assert pc.camera_view() is frame


# ---------- move_to_bbox_center ----------


def test_move_to_bbox_center_raises_when_uncalibrated() -> None:
    p = PhysiClaw()

    with pytest.raises(RuntimeError, match="Screen calibration not done"):
        p.move_to_bbox_center([0.1, 0.1, 0.2, 0.2])


def test_move_to_bbox_center_dispatches(pc: PhysiClaw) -> None:
    _wire_hardware(pc)

    pc.move_to_bbox_center([0.0, 0.0, 1.0, 1.0])

    pc._arm._fast_move.assert_called_once()


# ---------- AT guards ----------


def test_require_assistive_touch_raises_when_not_ready(pc: PhysiClaw) -> None:
    pc._assistive_touch.ready = False

    with pytest.raises(RuntimeError, match="AssistiveTouch not calibrated"):
        pc._require_assistive_touch()


def test_require_no_at_overlap_raises(pc: PhysiClaw) -> None:
    _wire_hardware(pc)
    pc._assistive_touch.overlaps_at.return_value = True

    with pytest.raises(ValueError, match="overlaps AssistiveTouch"):
        pc._require_no_at_overlap([0.0, 0.0, 0.1, 0.1], "tap")


def test_require_no_at_crossing_raises(pc: PhysiClaw) -> None:
    _wire_hardware(pc)
    pc._assistive_touch.swipe_crosses_at.return_value = True

    with pytest.raises(ValueError, match="crosses AssistiveTouch"):
        pc._require_no_at_crossing([0.0, 0.0, 0.1, 0.1], "up")


# ---------- lazy detectors ----------


def test_get_ocr_reader_lazy_caches(mocker, pc: PhysiClaw) -> None:
    fake = MagicMock()
    spy = mocker.patch.object(orchestrator, "OCRReader", return_value=fake)

    a = pc._get_ocr_reader()
    b = pc._get_ocr_reader()

    assert a is fake
    assert b is fake
    spy.assert_called_once()


def test_get_icon_detector_lazy_caches(mocker, pc: PhysiClaw) -> None:
    fake = MagicMock()
    spy = mocker.patch.object(orchestrator, "IconDetector", return_value=fake)

    a = pc._get_icon_detector()
    b = pc._get_icon_detector()

    assert a is fake
    assert b is fake
    spy.assert_called_once()


# ---------- accessor properties ----------


def test_arm_and_assistive_touch_properties(pc: PhysiClaw) -> None:
    pc._arm = MagicMock()

    assert pc.arm is pc._arm
    assert pc.assistive_touch is pc._assistive_touch


# ---------- _scan_text ----------


def test_scan_text_filters_offscreen(mocker, pc: PhysiClaw) -> None:
    _wire_hardware(pc)
    pc._ocr_reader = MagicMock()
    pc._cam.snapshot.return_value = np.zeros((4, 4, 3), dtype=np.uint8)
    mocker.patch.object(orchestrator, "phone_screen_crop_box", return_value=None)
    mocker.patch.object(orchestrator, "results_to_elements", return_value=[
        {"bbox": [0.1, 0.1, 0.2, 0.2]},
        {"bbox": [-1.0, -1.0, -0.5, -0.5]},
    ])
    mocker.patch.object(
        orchestrator, "bbox_on_screen",
        side_effect=lambda b: b[0] >= 0,
    )

    out = pc._scan_text()

    assert len(out) == 1
    assert out[0]["bbox"][0] == 0.1


# ---------- _detect / _scan_text ----------


def test_detect_calls_ui_pipeline(mocker, pc: PhysiClaw) -> None:
    pc._ocr_reader = MagicMock()
    pc._icon_detector = MagicMock()
    elements = [{"id": 0}]
    annotated = np.zeros((4, 4, 3), dtype=np.uint8)
    mocker.patch.object(
        orchestrator, "detect_ui_elements", return_value=(elements, annotated),
    )
    mocker.patch.object(orchestrator, "elements_to_json", return_value=[{"id": 0}])
    mocker.patch.object(orchestrator, "format_elements", return_value="LISTING")

    listing, ann = pc._detect(np.zeros((4, 4, 3), dtype=np.uint8))

    assert listing == "LISTING"
    assert ann is annotated


# ---------- peek ----------


def test_peek_retries_on_blurry_frame(mocker, pc: PhysiClaw) -> None:
    _wire_hardware(pc)
    pc._ocr_reader = MagicMock()
    pc._icon_detector = MagicMock()
    pc._cam.snapshot.return_value = np.zeros((10, 10, 3), dtype=np.uint8)
    mocker.patch.object(orchestrator, "crop_to_phone_screen", side_effect=lambda f, t: f)
    blur_values = iter([10.0, 100.0])
    mocker.patch.object(
        orchestrator, "laplacian_variance",
        side_effect=lambda *_: next(blur_values),
    )
    mocker.patch.object(orchestrator.time, "sleep")
    mocker.patch.object(
        orchestrator, "detect_ui_elements",
        return_value=([], np.zeros((4, 4, 3), dtype=np.uint8)),
    )
    mocker.patch.object(orchestrator, "elements_to_json", return_value=[])
    mocker.patch.object(orchestrator, "format_elements", return_value="ok")
    mocker.patch.object(orchestrator, "encode_jpeg", return_value=b"JPG")

    jpg, listing = pc.peek()

    assert jpg == b"JPG"
    assert listing == "ok"
    # Snapshot called twice (initial + retry after blur).
    assert pc._cam.snapshot.call_count == 2


def test_peek_does_not_retry_on_sharp_frame(mocker, pc: PhysiClaw) -> None:
    _wire_hardware(pc)
    pc._ocr_reader = MagicMock()
    pc._icon_detector = MagicMock()
    pc._cam.snapshot.return_value = np.zeros((4, 4, 3), dtype=np.uint8)
    mocker.patch.object(orchestrator, "crop_to_phone_screen", side_effect=lambda f, t: f)
    mocker.patch.object(orchestrator, "laplacian_variance", return_value=200.0)
    sleep_spy = mocker.patch.object(orchestrator.time, "sleep")
    mocker.patch.object(
        orchestrator, "detect_ui_elements",
        return_value=([], np.zeros((4, 4, 3), dtype=np.uint8)),
    )
    mocker.patch.object(orchestrator, "elements_to_json", return_value=[])
    mocker.patch.object(orchestrator, "format_elements", return_value="ok")
    mocker.patch.object(orchestrator, "encode_jpeg", return_value=b"JPG")

    pc.peek()

    sleep_spy.assert_not_called()
    assert pc._cam.snapshot.call_count == 1


# ---------- screenshot ----------


def test_screenshot_raises_on_timeout(mocker, pc: PhysiClaw) -> None:
    _wire_hardware(pc)
    pc.attach_bridge(MagicMock())
    pc._assistive_touch.take_screenshot.return_value = None

    with pytest.raises(TimeoutError, match="Screenshot upload timed out"):
        pc.screenshot()


def test_screenshot_decodes_and_detects(mocker, pc: PhysiClaw) -> None:
    _wire_hardware(pc)
    pc._ocr_reader = MagicMock()
    pc._icon_detector = MagicMock()
    pc.attach_bridge(MagicMock())
    pc._assistive_touch.take_screenshot.return_value = b"PNG"
    mocker.patch.object(orchestrator, "decode_image", return_value=np.zeros((4, 4, 3)))
    mocker.patch.object(
        orchestrator, "detect_ui_elements",
        return_value=([], np.zeros((4, 4, 3), dtype=np.uint8)),
    )
    mocker.patch.object(orchestrator, "elements_to_json", return_value=[])
    mocker.patch.object(orchestrator, "format_elements", return_value="L")
    mocker.patch.object(orchestrator, "encode_jpeg", return_value=b"JPG")

    jpg, listing = pc.screenshot()

    assert jpg == b"JPG"
    assert listing == "L"


# ---------- public gestures ----------


def test_tap_validates_and_dispatches(mocker, pc: PhysiClaw) -> None:
    _wire_hardware(pc)

    out = pc.tap([0.1, 0.1, 0.2, 0.2])

    assert "Tapped" in out
    pc._arm.tap.assert_called_once()


def test_double_tap_validates_and_dispatches(pc: PhysiClaw) -> None:
    _wire_hardware(pc)

    out = pc.double_tap([0.1, 0.1, 0.2, 0.2])

    assert "Double tapped" in out
    pc._arm.double_tap.assert_called_once()


def test_long_press_validates_and_dispatches(pc: PhysiClaw) -> None:
    _wire_hardware(pc)

    out = pc.long_press([0.1, 0.1, 0.2, 0.2])

    assert "Long pressed" in out
    pc._arm.long_press.assert_called_once()


def test_validate_swipe_rejects_bad_direction(pc: PhysiClaw) -> None:
    with pytest.raises(ValueError, match="direction must be"):
        pc._validate_swipe([0.1, 0.1, 0.2, 0.2], "diagonal", "m", "medium")


def test_validate_swipe_rejects_bad_size(pc: PhysiClaw) -> None:
    with pytest.raises(ValueError, match="size must be"):
        pc._validate_swipe([0.1, 0.1, 0.2, 0.2], "up", "huge", "medium")


def test_validate_swipe_rejects_bad_speed(pc: PhysiClaw) -> None:
    with pytest.raises(ValueError, match="speed must be"):
        pc._validate_swipe([0.1, 0.1, 0.2, 0.2], "up", "m", "warp")


def test_swipe_dispatches(pc: PhysiClaw) -> None:
    _wire_hardware(pc)

    out = pc.swipe([0.1, 0.1, 0.2, 0.2], "up", "m", "fast")

    assert "Swiped up m" in out
    pc._arm.swipe_to.assert_called_once()


# ---------- send_to_clipboard ----------


def test_send_to_clipboard_happy_path(pc: PhysiClaw) -> None:
    _wire_hardware(pc)
    bridge = MagicMock()
    bridge.wait_clipboard.return_value = True
    pc.attach_bridge(bridge)

    out = pc.send_to_clipboard("hello world")

    assert "Copied 'hello world'" in out
    bridge.send_text.assert_called_once_with("hello world")
    pc._assistive_touch.long_press.assert_called_once()


def test_send_to_clipboard_unconfirmed(pc: PhysiClaw) -> None:
    _wire_hardware(pc)
    bridge = MagicMock()
    bridge.wait_clipboard.return_value = False
    pc.attach_bridge(bridge)

    out = pc.send_to_clipboard("x")

    assert "clipboard not confirmed" in out


# ---------- _run_step / sequence ----------


def test_run_step_dispatches_each_tool(pc: PhysiClaw) -> None:
    _wire_hardware(pc)

    assert "Tapped" in pc._run_step("tap", [0.1, 0.1, 0.2, 0.2])
    assert "Double tapped" in pc._run_step("double_tap", [0.1, 0.1, 0.2, 0.2])
    assert "Long pressed" in pc._run_step("long_press", [0.1, 0.1, 0.2, 0.2])


def test_run_step_swipe_requires_dict_with_keys(pc: PhysiClaw) -> None:
    _wire_hardware(pc)

    with pytest.raises(ValueError, match="swipe arg needs bbox"):
        pc._run_step("swipe", "not-a-dict")
    with pytest.raises(ValueError, match="swipe arg needs bbox"):
        pc._run_step("swipe", {"bbox": [0, 0, 1, 1]})


def test_run_step_swipe_happy_path(pc: PhysiClaw) -> None:
    _wire_hardware(pc)

    out = pc._run_step("swipe", {
        "bbox": [0.1, 0.1, 0.2, 0.2], "direction": "up",
    })

    assert "Swiped up m" in out


def test_run_step_send_to_clipboard_requires_string(pc: PhysiClaw) -> None:
    _wire_hardware(pc)

    with pytest.raises(ValueError, match="must be a string"):
        pc._run_step("send_to_clipboard", 42)


def test_run_step_send_to_clipboard_dispatches(pc: PhysiClaw) -> None:
    _wire_hardware(pc)
    bridge = MagicMock()
    bridge.wait_clipboard.return_value = True
    pc.attach_bridge(bridge)

    out = pc._run_step("send_to_clipboard", "hi")

    assert "Copied 'hi'" in out


def test_run_step_unknown_tool_raises(pc: PhysiClaw) -> None:
    with pytest.raises(ValueError, match="not allowed in sequence"):
        pc._run_step("delete_app", "anything")


def test_sequence_runs_steps_in_order(pc: PhysiClaw) -> None:
    _wire_hardware(pc)

    out = pc.sequence([
        {"tool_name": "tap", "arg": [0.1, 0.1, 0.2, 0.2]},
        {"tool_name": "double_tap", "arg": [0.3, 0.3, 0.4, 0.4]},
    ])

    lines = out.splitlines()
    assert lines[0].startswith("1 tap ok")
    assert lines[1].startswith("2 double_tap ok")


def test_sequence_stops_on_first_failure(pc: PhysiClaw) -> None:
    _wire_hardware(pc)
    pc._arm.tap.side_effect = [None, RuntimeError("arm jammed")]

    out = pc.sequence([
        {"tool_name": "tap", "arg": [0.1, 0.1, 0.2, 0.2]},
        {"tool_name": "tap", "arg": [0.3, 0.3, 0.4, 0.4]},
        {"tool_name": "tap", "arg": [0.5, 0.5, 0.6, 0.6]},
    ])

    lines = out.splitlines()
    assert lines[0].startswith("1 tap ok")
    assert lines[1].startswith("2 tap FAIL")
    assert len(lines) == 2  # third step skipped


# ---------- macro gestures ----------


def test_home_screen_swipes_up(pc: PhysiClaw) -> None:
    _wire_hardware(pc)

    out = pc.home_screen()

    assert "Went to home screen" in out
    pc._arm.swipe_to.assert_called_once()


def test_go_back_swipes_right(pc: PhysiClaw) -> None:
    _wire_hardware(pc)

    out = pc.go_back()

    assert "Went back" in out


def test_force_quit_runs_four_gestures(pc: PhysiClaw) -> None:
    _wire_hardware(pc)

    out = pc.force_quit()

    assert "Force-quit" in out
    # Three swipes + one tap.
    assert pc._arm.swipe_to.call_count == 3
    assert pc._arm.tap.call_count == 1


# ---------- unlock_phone ----------


def test_unlock_phone_returns_when_keypad_not_found(mocker, pc: PhysiClaw) -> None:
    _wire_hardware(pc)
    mocker.patch.object(orchestrator.time, "sleep")
    mocker.patch.object(pc, "_scan_text", return_value=[])
    mocker.patch.object(orchestrator, "find_numpad_digit", return_value=None)

    out = pc.unlock_phone()

    assert "Failed to find passcode keypad" in out


def test_unlock_phone_taps_six_times_when_keypad_found(mocker, pc: PhysiClaw) -> None:
    _wire_hardware(pc)
    mocker.patch.object(orchestrator.time, "sleep")
    mocker.patch.object(pc, "_scan_text", return_value=[])
    mocker.patch.object(
        orchestrator, "find_numpad_digit", return_value=[0.1, 0.1, 0.2, 0.2],
    )

    out = pc.unlock_phone()

    assert "Passcode entered" in out
    # 1 wake-tap + 6 digit-taps = 7 taps.
    assert pc._arm.tap.call_count == 7


# ---------- shutdown ----------


def test_shutdown_closes_arm_and_camera(pc: PhysiClaw) -> None:
    # Uncalibrated fixture (pct_to_grbl is None) → park has no target, so
    # teardown falls back to homing.
    pc._arm = MagicMock()
    pc._cam = MagicMock()

    pc.shutdown()

    pc._arm.lift_stylus.assert_called_once()
    pc._arm.return_to_origin.assert_called_once()
    pc._arm.close.assert_called_once()
    pc._cam.close.assert_called_once()


def test_shutdown_parks_off_screen_when_calibrated(pc: PhysiClaw) -> None:
    # With calibration loaded, teardown rests the tip at the same off-screen
    # park spot used between taps — not the machine origin — so the phone
    # stays clear for placement / removal.
    pc._arm = MagicMock()
    pc._cam = MagicMock()
    pc.calibration.pct_to_grbl = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

    pc.shutdown()

    pc._arm.lift_stylus.assert_called_once()
    pc._arm.return_to_origin.assert_not_called()
    # park() drives a fast move to the calibrated park coordinate.
    pc._arm._fast_move.assert_called_once_with(-0.1, -0.05)
    pc._arm.close.assert_called_once()
    pc._cam.close.assert_called_once()


def test_shutdown_handles_no_hardware() -> None:
    p = PhysiClaw()

    p.shutdown()  # no raise


def test_shutdown_continues_when_coil_release_fails(pc: PhysiClaw) -> None:
    # A failed stylus lift must not strand the serial/camera handles.
    pc._arm = MagicMock()
    pc._arm.lift_stylus.side_effect = RuntimeError("serial timeout")
    pc._cam = MagicMock()

    pc.shutdown()  # swallows the error

    pc._arm.return_to_origin.assert_called_once()
    pc._arm.close.assert_called_once()
    pc._cam.close.assert_called_once()


def test_shutdown_continues_when_arm_close_fails(pc: PhysiClaw) -> None:
    # Camera must still close even if the arm teardown raises.
    pc._arm = MagicMock()
    pc._arm.return_to_origin.side_effect = RuntimeError("GRBL alarm")
    pc._arm.close.side_effect = RuntimeError("port gone")
    pc._cam = MagicMock()

    pc.shutdown()

    pc._arm.close.assert_called_once()  # attempted despite the prior failure
    pc._cam.close.assert_called_once()


def test_shutdown_swallows_camera_close_failure(pc: PhysiClaw) -> None:
    pc._cam = MagicMock()
    pc._cam.close.side_effect = RuntimeError("camera busy")

    pc.shutdown()  # no raise
