"""Tests for `physiclaw.core.orchestration.orchestrator` — PhysiClaw class."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from physiclaw.core.orchestration import observation, orchestrator
from physiclaw.core.orchestration.clipboard import ClipboardSyncError
from physiclaw.core.orchestration.orchestrator import PhysiClaw


# ---------- Fixtures ----------


def _identity_pct_to_grbl() -> np.ndarray:
    return np.array(
        [
            [10.0, 0.0, 0.0],
            [0.0, 20.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )


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
    """Construct PhysiClaw and pre-mock the assistive_touch + watchdog.
    The post-gesture settle sleep is zeroed so gesture tests stay fast."""
    p = PhysiClaw()
    p._assistive_touch = MagicMock()
    p._assistive_touch.ready = True
    p._assistive_touch.at_screen = (0.05, 0.1)
    p._assistive_touch.overlaps_at.return_value = False
    p._assistive_touch.swipe_crosses_at.return_value = False
    p._watchdog = MagicMock()
    p._observer.GESTURE_SETTLE_SECONDS = 0
    # Synthetic test frames are flat (Laplacian variance 0) — disable the
    # autofocus blur-retry so grabs stay one-frame-per-call.
    p._observer.GRAB_BLUR_THRESHOLD = 0
    # Flat frames also read as blurry to the camera-quality monitor —
    # neutralize it so its ⚠ line doesn't join every listing. The
    # camera-quality tests below re-arm it per test.
    p._observer._quality = MagicMock()
    p._observer._quality.observe.return_value = None
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
    from physiclaw.common import paths

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


def test_tap_blocked_when_target_overlaps_assistive_touch(pc: PhysiClaw) -> None:
    # Guard unit tests live in test_gestures.py — this pins the wiring:
    # dispatch consults the validator before moving the arm.
    _wire_hardware(pc)
    pc._assistive_touch.overlaps_at.return_value = True

    with pytest.raises(ValueError, match="overlaps AssistiveTouch"):
        pc.tap([0.0, 0.0, 0.1, 0.1])

    pc._arm.tap.assert_not_called()


def test_swipe_blocked_when_crossing_assistive_touch(pc: PhysiClaw) -> None:
    _wire_hardware(pc)
    pc._assistive_touch.swipe_crosses_at.return_value = True

    with pytest.raises(ValueError, match="crosses AssistiveTouch"):
        pc.swipe([0.0, 0.0, 0.1, 0.1], "up")

    pc._arm.swipe_to.assert_not_called()


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
    mocker.patch.object(
        orchestrator,
        "results_to_elements",
        return_value=[
            {"bbox": [0.1, 0.1, 0.2, 0.2]},
            {"bbox": [-1.0, -1.0, -0.5, -0.5]},
        ],
    )
    mocker.patch.object(
        orchestrator,
        "bbox_on_screen",
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
        orchestrator,
        "detect_ui_elements",
        return_value=(elements, annotated),
    )
    mocker.patch.object(orchestrator, "elements_to_json", return_value=[{"id": 0}])
    mocker.patch.object(orchestrator, "format_elements", return_value="LISTING")

    listing, ann = pc._detect(np.zeros((4, 4, 3), dtype=np.uint8))

    assert listing == "LISTING"
    assert ann is annotated


# ---------- peek ----------


def _wire_peek(mocker, pc: PhysiClaw, *, listing: str = "ok", sharpness=200.0):
    """Stub peek's whole pipeline: hardware, pass-through crop, sharpness,
    detection (returning `listing`), and JPEG encode. `sharpness` is a
    single score, or a list consumed one grab at a time (blur-retry
    tests). Returns the patched `time.sleep` spy."""
    _wire_hardware(pc)
    pc._ocr_reader = MagicMock()
    pc._icon_detector = MagicMock()
    pc._cam.snapshot.return_value = np.zeros((10, 10, 3), dtype=np.uint8)
    mocker.patch.object(
        orchestrator, "crop_to_phone_screen", side_effect=lambda f, t: f
    )
    if isinstance(sharpness, list):
        blur_values = iter(sharpness)
        mocker.patch.object(
            observation,
            "laplacian_variance",
            side_effect=lambda *_: next(blur_values),
        )
    else:
        mocker.patch.object(
            observation,
            "laplacian_variance",
            return_value=sharpness,
        )
    sleep_spy = mocker.patch.object(observation.time, "sleep")
    mocker.patch.object(
        orchestrator,
        "detect_ui_elements",
        return_value=([], np.zeros((4, 4, 3), dtype=np.uint8)),
    )
    mocker.patch.object(orchestrator, "elements_to_json", return_value=[])
    mocker.patch.object(orchestrator, "format_elements", return_value=listing)
    mocker.patch.object(orchestrator, "encode_jpeg", return_value=b"JPG")
    return sleep_spy


def test_peek_retries_on_blurry_frame(mocker, pc: PhysiClaw) -> None:
    _wire_peek(mocker, pc, sharpness=[10.0, 100.0])

    jpg, listing = pc.peek()

    assert jpg == b"JPG"
    assert listing == "ok"
    # Snapshot called twice (initial + retry after blur).
    assert pc._cam.snapshot.call_count == 2


def test_peek_does_not_retry_on_sharp_frame(mocker, pc: PhysiClaw) -> None:
    sleep_spy = _wire_peek(mocker, pc)

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
        orchestrator,
        "detect_ui_elements",
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

    assert "Tapped" in out.text
    pc._arm.tap.assert_called_once()


def test_double_tap_validates_and_dispatches(pc: PhysiClaw) -> None:
    _wire_hardware(pc)

    out = pc.double_tap([0.1, 0.1, 0.2, 0.2])

    assert "Double tapped" in out.text
    pc._arm.double_tap.assert_called_once()


def test_long_press_validates_and_dispatches(pc: PhysiClaw) -> None:
    _wire_hardware(pc)

    out = pc.long_press([0.1, 0.1, 0.2, 0.2])

    assert "Long pressed" in out.text
    pc._arm.long_press.assert_called_once()


def test_swipe_rejects_out_of_range_args(pc: PhysiClaw) -> None:
    # Range-check unit tests live in test_gestures.py — this pins the
    # wiring: the public path validates before touching hardware.
    with pytest.raises(ValueError, match="direction must be"):
        pc.swipe([0.1, 0.1, 0.2, 0.2], "diagonal")


def test_swipe_dispatches(pc: PhysiClaw) -> None:
    _wire_hardware(pc)

    out = pc.swipe([0.1, 0.1, 0.2, 0.2], "up", "m", "fast")

    assert "Swiped up m" in out.text
    pc._arm.swipe_to.assert_called_once()


# ---------- send_to_clipboard ----------


def _wire_failing_bridge(pc: PhysiClaw) -> MagicMock:
    """Wire hardware + a bridge whose clipboard sync never confirms."""
    _wire_hardware(pc)
    bridge = MagicMock()
    bridge.wait_clipboard.return_value = False
    pc.attach_bridge(bridge)
    return bridge


def test_send_to_clipboard_happy_path(pc: PhysiClaw) -> None:
    _wire_hardware(pc)
    bridge = MagicMock()
    bridge.wait_clipboard.return_value = True
    pc.attach_bridge(bridge)

    out = pc.send_to_clipboard("hello world")

    # Length, not the text itself: the echoed content would join the
    # verdict-scanned action text on the sequence path (see
    # _send_to_clipboard docstring).
    assert "Copied 11 chars" in out
    assert "hello world" not in out
    bridge.send_text.assert_called_once_with("hello world")
    pc._assistive_touch.long_press.assert_called_once()


def test_send_to_clipboard_unconfirmed_raises(pc: PhysiClaw) -> None:
    # Raise, not return: a returned message once let `sequence` mark the
    # step "ok" and paste+send the phone's STALE clipboard into an IM.
    bridge = _wire_failing_bridge(pc)

    with pytest.raises(ClipboardSyncError, match="do NOT paste"):
        pc.send_to_clipboard("x")

    bridge.wait_clipboard.assert_called_once_with(timeout=pc._clipboard.CONFIRM_SECONDS)


def test_send_to_clipboard_miss_clears_queued_text(pc: PhysiClaw) -> None:
    # A LATE Shortcut run must not fetch the text after we've told the
    # agent "the phone clipboard still holds the previous content".
    bridge = _wire_failing_bridge(pc)

    with pytest.raises(ClipboardSyncError):
        pc.send_to_clipboard("x")

    bridge.clear_text.assert_called_once()


def test_send_to_clipboard_success_resets_miss_counter(pc: PhysiClaw) -> None:
    # Escalation/decay/timeout policy is unit-tested in test_clipboard.py —
    # this pins the wiring: a miss is recorded, a confirmed sync resets it,
    # and the state's window reaches wait_clipboard on every call.
    _wire_hardware(pc)
    bridge = MagicMock()
    bridge.wait_clipboard.side_effect = [False, True, False]
    pc.attach_bridge(bridge)

    with pytest.raises(ClipboardSyncError):
        pc.send_to_clipboard("x")
    assert "Copied" in pc.send_to_clipboard("x")
    # Post-reset miss is #1 again: full window, no escalation text.
    with pytest.raises(ClipboardSyncError) as e:
        pc.send_to_clipboard("x")

    assert "Miss #" not in str(e.value)
    assert bridge.wait_clipboard.call_args_list[2].kwargs["timeout"] == (
        pc._clipboard.CONFIRM_SECONDS
    )


# ---------- _run_step / sequence ----------


def test_run_step_dispatches_each_tool(pc: PhysiClaw) -> None:
    _wire_hardware(pc)

    assert "Tapped" in pc._run_step("tap", [0.1, 0.1, 0.2, 0.2])
    assert "Double tapped" in pc._run_step("double_tap", [0.1, 0.1, 0.2, 0.2])
    assert "Long pressed" in pc._run_step("long_press", [0.1, 0.1, 0.2, 0.2])


def test_run_step_swipe_happy_path(pc: PhysiClaw) -> None:
    _wire_hardware(pc)

    out = pc._run_step(
        "swipe",
        {
            "bbox": [0.1, 0.1, 0.2, 0.2],
            "direction": "up",
        },
    )

    assert "Swiped up m" in out


def test_run_step_send_to_clipboard_dispatches(pc: PhysiClaw) -> None:
    _wire_hardware(pc)
    bridge = MagicMock()
    bridge.wait_clipboard.return_value = True
    pc.attach_bridge(bridge)

    out = pc._run_step("send_to_clipboard", "hi")

    assert "Copied 2 chars" in out


def test_run_step_unknown_tool_raises(pc: PhysiClaw) -> None:
    # Parse-error unit tests live in test_gestures.py — this pins the
    # wiring: validator errors surface from `_run_step` (and so abort a
    # `sequence` step).
    with pytest.raises(ValueError, match="not allowed in sequence"):
        pc._run_step("delete_app", "anything")


def test_sequence_runs_steps_in_order(pc: PhysiClaw) -> None:
    _wire_hardware(pc)

    out = pc.sequence(
        [
            {"tool_name": "tap", "arg": [0.1, 0.1, 0.2, 0.2]},
            {"tool_name": "double_tap", "arg": [0.3, 0.3, 0.4, 0.4]},
        ]
    )

    lines = out.text.splitlines()
    assert lines[0].startswith("1 tap ok")
    assert lines[1].startswith("2 double_tap ok")


def test_sequence_stops_on_first_failure(pc: PhysiClaw) -> None:
    _wire_hardware(pc)
    pc._arm.tap.side_effect = [None, RuntimeError("arm jammed")]

    out = pc.sequence(
        [
            {"tool_name": "tap", "arg": [0.1, 0.1, 0.2, 0.2]},
            {"tool_name": "tap", "arg": [0.3, 0.3, 0.4, 0.4]},
            {"tool_name": "tap", "arg": [0.5, 0.5, 0.6, 0.6]},
        ]
    )

    lines = out.text.splitlines()
    assert lines[0].startswith("1 tap ok")
    assert lines[1].startswith("2 tap FAIL")
    assert len(lines) == 2  # third step skipped


def test_sequence_aborts_before_paste_on_unconfirmed_clipboard(
    pc: PhysiClaw,
) -> None:
    # The 18:36 incident shape: [send_to_clipboard, long_press, tap Paste,
    # tap Send]. An unconfirmed sync must kill the batch so the stale
    # clipboard is never pasted and sent.
    _wire_failing_bridge(pc)

    out = pc.sequence(
        [
            {"tool_name": "send_to_clipboard", "arg": "新消息"},
            {"tool_name": "long_press", "arg": [0.1, 0.9, 0.7, 0.95]},
            {"tool_name": "tap", "arg": [0.1, 0.1, 0.2, 0.2]},
        ]
    )

    lines = out.text.splitlines()
    assert lines[0].startswith("1 send_to_clipboard FAIL")
    assert "do NOT paste" in lines[0]
    assert len(lines) == 1  # long_press + tap never ran
    pc._arm.tap.assert_not_called()
    pc._arm.long_press.assert_not_called()


# ---------- screen-change verdict ----------


def _wire_verdict_frames(mocker, pc: PhysiClaw, frames: list[np.ndarray]) -> None:
    """Route `_grab_screen`'s crop through controlled frames — first call
    returns frames[0] (before), second frames[1] (after). Detection on
    the after-frame is stubbed so tests don't need the ONNX model."""
    pc._cam.snapshot.return_value = np.zeros((4, 4, 3), dtype=np.uint8)
    mocker.patch.object(orchestrator, "crop_to_phone_screen", side_effect=frames)
    mocker.patch.object(pc, "_detect", return_value=("LISTING", MagicMock()))
    mocker.patch.object(observation, "encode_jpeg", return_value=b"VIEW_JPG")


def test_tap_appends_changed_verdict(mocker, pc: PhysiClaw) -> None:
    _wire_hardware(pc)
    before = np.full((200, 100, 3), 128, dtype=np.uint8)
    after = np.full((200, 100, 3), 30, dtype=np.uint8)
    _wire_verdict_frames(mocker, pc, [before, after])

    out = pc.tap([0.1, 0.1, 0.2, 0.2])

    assert out.text.endswith("| screen: changed")


def test_tap_appends_unchanged_verdict(mocker, pc: PhysiClaw) -> None:
    _wire_hardware(pc)
    frame = np.full((200, 100, 3), 128, dtype=np.uint8)
    _wire_verdict_frames(mocker, pc, [frame, frame.copy()])

    out = pc.tap([0.1, 0.1, 0.2, 0.2])

    assert out.text.endswith("| screen: no visible change")


def test_tap_unmarked_when_camera_fails(mocker, pc: PhysiClaw) -> None:
    # Fail open: no frames → no marker, gesture result intact.
    _wire_hardware(pc)
    pc._cam.snapshot.return_value = None

    out = pc.tap([0.1, 0.1, 0.2, 0.2])

    assert out.text == "Tapped at bbox [0.1, 0.1, 0.2, 0.2]"
    assert out.jpeg is None and out.listing is None
    pc._arm.tap.assert_called_once()


def test_sequence_appends_whole_batch_verdict(mocker, pc: PhysiClaw) -> None:
    _wire_hardware(pc)
    before = np.full((200, 100, 3), 128, dtype=np.uint8)
    after = np.full((200, 100, 3), 30, dtype=np.uint8)
    _wire_verdict_frames(mocker, pc, [before, after])

    out = pc.sequence([{"tool_name": "tap", "arg": [0.1, 0.1, 0.2, 0.2]}])

    assert out.text.splitlines()[0].startswith("1 tap ok")
    assert out.text.endswith("| screen: changed")


def test_go_back_unchanged_verdict_signals_no_pop(mocker, pc: PhysiClaw) -> None:
    _wire_hardware(pc)
    frame = np.full((200, 100, 3), 128, dtype=np.uint8)
    _wire_verdict_frames(mocker, pc, [frame, frame.copy()])

    out = pc.go_back()

    assert out.text.endswith("| screen: no visible change")


def test_grab_screen_wiring_retries_through_orchestrator(mocker, pc: PhysiClaw) -> None:
    # The observer's blur-retry drives the orchestrator's park + crop
    # wiring — a blurry first crop grabs a second one. (The retry logic
    # itself is unit-tested in test_observation.py.)
    _wire_hardware(pc)
    pc._observer.GRAB_BLUR_THRESHOLD = 50.0
    pc._cam.snapshot.return_value = np.zeros((4, 4, 3), dtype=np.uint8)
    rng = np.random.default_rng(7)
    sharp = rng.integers(0, 255, size=(200, 100, 3), dtype=np.uint8)
    blurry = np.full((200, 100, 3), 128, dtype=np.uint8)
    mocker.patch.object(
        orchestrator, "crop_to_phone_screen", side_effect=[blurry, sharp]
    )
    mocker.patch.object(observation.time, "sleep")

    frame, sharp_flag = pc._observer.grab_screen()

    assert frame is sharp
    assert sharp_flag is True
    assert orchestrator.crop_to_phone_screen.call_count == 2


def test_blurry_after_frame_withholds_verdict_but_keeps_view(
    mocker, pc: PhysiClaw
) -> None:
    # Sharp-vs-blurry frames diff as a full-screen change — the verdict
    # must fail open (no marker), while the view still attaches.
    _wire_hardware(pc)
    before = np.full((200, 100, 3), 128, dtype=np.uint8)
    after = np.full((200, 100, 3), 30, dtype=np.uint8)
    # Every grab reads blurry → each grab_screen consumes two crops.
    _wire_verdict_frames(mocker, pc, [before, before, after, after])
    pc._observer.GRAB_BLUR_THRESHOLD = float("inf")
    mocker.patch.object(observation.time, "sleep")

    out = pc.tap([0.1, 0.1, 0.2, 0.2])

    assert out.text == "Tapped at bbox [0.1, 0.1, 0.2, 0.2]"  # unmarked
    assert out.jpeg == b"VIEW_JPG"
    assert out.listing == "LISTING"


# ---------- fused post-gesture view ----------


def test_gesture_returns_fused_view(mocker, pc: PhysiClaw) -> None:
    # The after-frame that feeds the verdict also feeds detection: the
    # gesture result carries the annotated JPEG + listing like a peek.
    _wire_hardware(pc)
    before = np.full((200, 100, 3), 128, dtype=np.uint8)
    after = np.full((200, 100, 3), 30, dtype=np.uint8)
    _wire_verdict_frames(mocker, pc, [before, after])

    out = pc.tap([0.1, 0.1, 0.2, 0.2])

    assert out.jpeg == b"VIEW_JPG"
    assert out.listing == "LISTING"


def test_gesture_view_fails_open_when_detection_raises(mocker, pc: PhysiClaw) -> None:
    _wire_hardware(pc)
    before = np.full((200, 100, 3), 128, dtype=np.uint8)
    after = np.full((200, 100, 3), 30, dtype=np.uint8)
    pc._cam.snapshot.return_value = np.zeros((4, 4, 3), dtype=np.uint8)
    mocker.patch.object(
        orchestrator, "crop_to_phone_screen", side_effect=[before, after]
    )
    mocker.patch.object(pc, "_detect", side_effect=RuntimeError("model missing"))

    out = pc.tap([0.1, 0.1, 0.2, 0.2])

    # Verdict still attaches; only the view is absent.
    assert out.text.endswith("| screen: changed")
    assert out.jpeg is None and out.listing is None


def test_gesture_view_absent_without_after_frame(mocker, pc: PhysiClaw) -> None:
    _wire_hardware(pc)
    pc._cam.snapshot.return_value = None  # camera down

    out = pc.tap([0.1, 0.1, 0.2, 0.2])

    assert out.text == "Tapped at bbox [0.1, 0.1, 0.2, 0.2]"
    assert out.jpeg is None and out.listing is None


# ---------- macro gestures ----------


def test_home_screen_swipes_up(pc: PhysiClaw) -> None:
    _wire_hardware(pc)

    out = pc.home_screen()

    assert "Went to home screen" in out.text
    pc._arm.swipe_to.assert_called_once()


def test_go_back_swipes_right(pc: PhysiClaw) -> None:
    _wire_hardware(pc)

    out = pc.go_back()

    assert "Went back" in out.text


def test_force_quit_runs_four_gestures(pc: PhysiClaw) -> None:
    _wire_hardware(pc)

    out = pc.force_quit()

    assert "Force-quit" in out.text
    # Three swipes + one tap.
    assert pc._arm.swipe_to.call_count == 3
    assert pc._arm.tap.call_count == 1


# ---------- unlock_phone ----------


def test_unlock_phone_returns_when_keypad_not_found(mocker, pc: PhysiClaw) -> None:
    _wire_hardware(pc)
    pc._ocr_reader = MagicMock()
    mocker.patch.object(orchestrator.time, "sleep")
    mocker.patch.object(pc, "_scan_text", return_value=[])
    mocker.patch.object(orchestrator, "find_numpad_digit", return_value=None)

    out = pc.unlock_phone()

    assert "Failed to find passcode keypad" in out.text


def test_unlock_phone_taps_six_times_when_keypad_found(mocker, pc: PhysiClaw) -> None:
    _wire_hardware(pc)
    pc._ocr_reader = MagicMock()
    mocker.patch.object(orchestrator.time, "sleep")
    mocker.patch.object(pc, "_scan_text", return_value=[])
    mocker.patch.object(
        orchestrator,
        "find_numpad_digit",
        return_value=[0.1, 0.1, 0.2, 0.2],
    )

    out = pc.unlock_phone()

    assert "Passcode entered" in out.text
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


# ---------- camera-quality warning (AF/AE failure) ----------


def test_peek_appends_quality_warning_to_listing(mocker, pc: PhysiClaw) -> None:
    _wire_peek(mocker, pc, listing="rows")
    pc._observer._quality.observe.return_value = "⚠ camera: bad"

    _, listing = pc.peek()

    assert listing == "rows\n⚠ camera: bad"


def test_gesture_view_carries_quality_warning(mocker, pc: PhysiClaw) -> None:
    _wire_hardware(pc)
    before = np.full((200, 100, 3), 128, dtype=np.uint8)
    after = np.full((200, 100, 3), 30, dtype=np.uint8)
    _wire_verdict_frames(mocker, pc, [before, after])
    pc._observer._quality.observe.return_value = "⚠ camera: bad"

    out = pc.tap([0.1, 0.1, 0.2, 0.2])

    assert out.listing == "LISTING\n⚠ camera: bad"
    assert out.text.endswith("| screen: changed")  # verdict untouched


def test_gesture_warning_rides_action_text_when_detect_fails(
    mocker, pc: PhysiClaw
) -> None:
    # Detection crashing must not silence the observation: a blurry/blown
    # frame is a plausible CAUSE of the failed view, so with no listing to
    # carry the ⚠ line it rides the action text instead.
    _wire_hardware(pc)
    before = np.full((200, 100, 3), 128, dtype=np.uint8)
    after = np.full((200, 100, 3), 30, dtype=np.uint8)
    _wire_verdict_frames(mocker, pc, [before, after])
    pc._detect.side_effect = RuntimeError("model gone")
    pc._observer._quality.observe.return_value = "⚠ camera: bad"

    out = pc.tap([0.1, 0.1, 0.2, 0.2])

    assert out.listing is None
    assert out.text.endswith("\n⚠ camera: bad")
    pc._observer._quality.observe.assert_called_once()


def test_quality_check_failure_never_costs_the_view(mocker, pc: PhysiClaw) -> None:
    _wire_hardware(pc)
    before = np.full((200, 100, 3), 128, dtype=np.uint8)
    after = np.full((200, 100, 3), 30, dtype=np.uint8)
    _wire_verdict_frames(mocker, pc, [before, after])
    pc._observer._quality.observe.side_effect = RuntimeError("boom")

    out = pc.tap([0.1, 0.1, 0.2, 0.2])

    assert out.listing == "LISTING"  # fail-open: view intact, no line


# ---------- tune_exposure ----------


def _tune_ok() -> object:
    from physiclaw.core.hardware.exposure import TuneResult

    return TuneResult(mode="auto", exposure=None, ok=True, detail="in band")


def test_tune_exposure_noop_without_camera(mocker, pc: PhysiClaw) -> None:
    conv = mocker.patch.object(orchestrator.exposure, "converge")

    pc.tune_exposure()  # no camera, no transforms — silently returns

    conv.assert_not_called()


def test_tune_exposure_noop_when_platform_not_tunable(mocker, pc: PhysiClaw) -> None:
    _wire_hardware(pc)
    pc._cam.exposure_tunable = False
    conv = mocker.patch.object(orchestrator.exposure, "converge")

    pc.tune_exposure()  # the macOS path

    conv.assert_not_called()


def test_tune_exposure_skips_when_busy(mocker, pc: PhysiClaw) -> None:
    _wire_hardware(pc)
    pc._cam.exposure_tunable = True
    conv = mocker.patch.object(orchestrator.exposure, "converge")
    pc.acquire()
    try:
        pc.tune_exposure()
    finally:
        pc.release()

    conv.assert_not_called()


def test_tune_exposure_runs_converge_and_releases_lock(mocker, pc: PhysiClaw) -> None:
    _wire_hardware(pc)
    pc._cam.exposure_tunable = True
    conv = mocker.patch.object(
        orchestrator.exposure,
        "converge",
        return_value=_tune_ok(),
    )

    pc.tune_exposure()

    conv.assert_called_once()
    kwargs = conv.call_args.kwargs
    from physiclaw.common.config import CONFIG

    assert kwargs["start"] == CONFIG.camera.exposure
    assert kwargs["prefer_auto"] == CONFIG.camera.auto_exposure
    # Setters are the camera's own bound methods.
    args = conv.call_args.args
    assert args[1] == pc._cam.set_auto_exposure
    assert args[2] == pc._cam.set_manual_exposure
    pc.acquire()  # lock released after the tune
    pc.release()


def test_tune_exposure_meter_crops_and_assesses(mocker, pc: PhysiClaw) -> None:
    _wire_hardware(pc)
    pc._cam.exposure_tunable = True
    frame = np.full((200, 100, 3), 128, dtype=np.uint8)
    pc._cam.wait_frames.return_value = True
    pc._cam.peek.return_value = frame
    cropped = np.full((100, 50, 3), 128, dtype=np.uint8)
    mocker.patch.object(orchestrator, "crop_to_phone_screen", return_value=cropped)
    assess = mocker.patch.object(orchestrator.quality, "assess")
    conv = mocker.patch.object(
        orchestrator.exposure,
        "converge",
        return_value=_tune_ok(),
    )

    pc.tune_exposure()

    meter = conv.call_args.args[0]
    report = meter()
    assess.assert_called_once_with(cropped)
    assert report is assess.return_value


def test_tune_exposure_meter_fails_open_on_stalled_reader(
    mocker, pc: PhysiClaw
) -> None:
    _wire_hardware(pc)
    pc._cam.exposure_tunable = True
    pc._cam.wait_frames.return_value = False  # reader stalled
    conv = mocker.patch.object(
        orchestrator.exposure,
        "converge",
        return_value=_tune_ok(),
    )

    pc.tune_exposure()

    assert conv.call_args.args[0]() is None


def test_tune_exposure_swallows_converge_crash(mocker, pc: PhysiClaw) -> None:
    _wire_hardware(pc)
    pc._cam.exposure_tunable = True
    mocker.patch.object(
        orchestrator.exposure,
        "converge",
        side_effect=RuntimeError("boom"),
    )

    pc.tune_exposure()  # no raise

    pc.acquire()  # and the lock is not leaked
    pc.release()
