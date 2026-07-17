"""Tests for `physiclaw.core.orchestration.orchestrator` — the PhysiClaw facade.

Covers composition, the validated gesture dispatch, the lock +
observation bracket, and the tool operations. Rig lifecycle lives in
`test_rig.py`; detectors and the exposure tune in `test_perception.py`.
Hardware doubles come from this directory's conftest.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from physiclaw.core.orchestration import gestures, observation, orchestrator, perception
from physiclaw.core.orchestration.clipboard import ClipboardSyncError
from physiclaw.core.orchestration.orchestrator import PhysiClaw

# ---------- Fixtures ----------


@pytest.fixture
def pc(at_double) -> PhysiClaw:
    """Construct PhysiClaw and pre-mock the rig's assistive_touch.
    The post-gesture settle sleep is zeroed so gesture tests stay fast."""
    p = PhysiClaw()
    p.rig._assistive_touch = at_double()
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


# ---------- Composition ----------


def test_init_composes_rig_perception_and_observer() -> None:
    p = PhysiClaw()

    assert p.perception._rig is p.rig
    assert p._observer is not None
    assert p._validator is not None
    assert p._clipboard is not None


def test_shutdown_delegates_to_rig(mocker, pc: PhysiClaw) -> None:
    spy = mocker.patch.object(pc.rig, "shutdown")

    pc.shutdown()

    spy.assert_called_once_with()


def test_become_ready_settles_camera_before_flipping_ready(
    mocker, pc: PhysiClaw
) -> None:
    """The agent polls `ready` and peeks immediately — the flip must not
    be observable before the camera settle has run."""
    order: list[str] = []
    mocker.patch.object(
        pc.perception, "settle_camera", side_effect=lambda: order.append("settle")
    )
    mocker.patch.object(pc.rig, "mark_ready", side_effect=lambda: order.append("ready"))

    pc.become_ready()

    assert order == ["settle", "ready"]


def test_validator_reads_current_rig_state(pc: PhysiClaw) -> None:
    # The validator holds accessor lambdas, not objects — replacing the
    # rig's AssistiveTouch after construction must be visible to it.
    replacement = MagicMock()
    pc.rig._assistive_touch = replacement

    assert pc._validator._assistive_touch() is replacement


# ---------- AT guards ----------


def test_tap_blocked_when_target_overlaps_assistive_touch(
    pc: PhysiClaw, wire_rig
) -> None:
    # Guard unit tests live in test_gestures.py — this pins the wiring:
    # dispatch consults the validator before moving the arm.
    wire_rig(pc.rig)
    pc.rig._assistive_touch.overlaps_at.return_value = True

    with pytest.raises(ValueError, match="overlaps AssistiveTouch"):
        pc.tap([0.0, 0.0, 0.1, 0.1])

    pc.rig._arm.tap.assert_not_called()


def test_swipe_blocked_when_crossing_assistive_touch(pc: PhysiClaw, wire_rig) -> None:
    wire_rig(pc.rig)
    pc.rig._assistive_touch.swipe_crosses_at.return_value = True

    with pytest.raises(ValueError, match="crosses AssistiveTouch"):
        pc.swipe([0.0, 0.0, 0.1, 0.1], "up")

    pc.rig._arm.swipe_to.assert_not_called()


# ---------- peek ----------


def _wire_peek(mocker, pc, wire_rig, *, listing: str = "ok", sharpness=200.0):
    """Stub peek's whole pipeline: hardware, pass-through crop, sharpness,
    detection (returning `listing`), and JPEG encode. `sharpness` is a
    single score, or a list consumed one grab at a time (blur-retry
    tests). Returns the patched `time.sleep` spy."""
    wire_rig(pc.rig)
    pc.perception._ocr_reader = MagicMock()
    pc.perception._icon_detector = MagicMock()
    pc.rig._cam.snapshot.return_value = np.zeros((10, 10, 3), dtype=np.uint8)
    mocker.patch.object(perception, "crop_to_phone_screen", side_effect=lambda f, t: f)

    def _report(score: float):
        # peek's retry decision reads a full QualityReport now; only
        # sharpness matters to these tests (clip 0 = never blown).
        return observation.quality.QualityReport(
            sharpness=score, clip_pct=0.0, median_luma=100.0
        )

    if isinstance(sharpness, list):
        blur_values = iter(sharpness)
        mocker.patch.object(
            observation.quality,
            "assess",
            side_effect=lambda *_: _report(next(blur_values)),
        )
    else:
        mocker.patch.object(
            observation.quality,
            "assess",
            return_value=_report(sharpness),
        )
    sleep_spy = mocker.patch.object(observation.time, "sleep")
    mocker.patch.object(
        perception,
        "detect_ui_elements",
        return_value=([], np.zeros((4, 4, 3), dtype=np.uint8)),
    )
    mocker.patch.object(perception, "elements_to_json", return_value=[])
    mocker.patch.object(perception, "format_elements", return_value=listing)
    mocker.patch.object(orchestrator, "encode_view_jpeg", return_value=b"JPG")
    return sleep_spy


def test_peek_retries_on_blurry_frame(mocker, pc: PhysiClaw, wire_rig) -> None:
    _wire_peek(mocker, pc, wire_rig, sharpness=[10.0, 100.0])

    jpg, listing = pc.peek()

    assert jpg == b"JPG"
    assert listing == "ok"
    # Snapshot called twice (initial + retry after blur).
    assert pc.rig._cam.snapshot.call_count == 2


def test_peek_does_not_retry_on_sharp_frame(mocker, pc: PhysiClaw, wire_rig) -> None:
    sleep_spy = _wire_peek(mocker, pc, wire_rig)

    pc.peek()

    sleep_spy.assert_not_called()
    assert pc.rig._cam.snapshot.call_count == 1


# ---------- screenshot ----------


def test_screenshot_raises_on_timeout(pc: PhysiClaw, wire_rig, bridge_double) -> None:
    wire_rig(pc.rig)
    pc.rig.attach_bridge(bridge_double())
    pc.rig._assistive_touch.take_screenshot.return_value = None

    with pytest.raises(TimeoutError, match="Screenshot upload timed out"):
        pc.screenshot()


def test_screenshot_decodes_and_detects(
    mocker, pc: PhysiClaw, wire_rig, bridge_double
) -> None:
    wire_rig(pc.rig)
    pc.perception._ocr_reader = MagicMock()
    pc.perception._icon_detector = MagicMock()
    pc.rig.attach_bridge(bridge_double())
    pc.rig._assistive_touch.take_screenshot.return_value = b"PNG"
    mocker.patch.object(orchestrator, "decode_image", return_value=np.zeros((4, 4, 3)))
    mocker.patch.object(
        perception,
        "detect_ui_elements",
        return_value=([], np.zeros((4, 4, 3), dtype=np.uint8)),
    )
    mocker.patch.object(perception, "elements_to_json", return_value=[])
    mocker.patch.object(perception, "format_elements", return_value="L")
    mocker.patch.object(orchestrator, "encode_view_jpeg", return_value=b"JPG")

    jpg, listing = pc.screenshot()

    assert jpg == b"JPG"
    assert listing == "L"


# ---------- public gestures ----------


def test_tap_validates_and_dispatches(pc: PhysiClaw, wire_rig) -> None:
    wire_rig(pc.rig)

    out = pc.tap([0.1, 0.1, 0.2, 0.2])

    assert "Tapped" in out.text
    pc.rig._arm.tap.assert_called_once()


def test_double_tap_validates_and_dispatches(pc: PhysiClaw, wire_rig) -> None:
    wire_rig(pc.rig)

    out = pc.double_tap([0.1, 0.1, 0.2, 0.2])

    assert "Double tapped" in out.text
    pc.rig._arm.double_tap.assert_called_once()


def test_long_press_validates_and_dispatches(pc: PhysiClaw, wire_rig) -> None:
    wire_rig(pc.rig)

    out = pc.long_press([0.1, 0.1, 0.2, 0.2])

    assert "Long pressed" in out.text
    pc.rig._arm.long_press.assert_called_once()


def test_swipe_rejects_out_of_range_args(pc: PhysiClaw) -> None:
    # Range-check unit tests live in test_gestures.py — this pins the
    # wiring: the public path validates before touching hardware.
    with pytest.raises(ValueError, match="direction must be"):
        pc.swipe([0.1, 0.1, 0.2, 0.2], "diagonal")


def test_swipe_dispatches(pc: PhysiClaw, wire_rig) -> None:
    wire_rig(pc.rig)

    out = pc.swipe([0.1, 0.1, 0.2, 0.2], "up", "m", "fast")

    assert "Swiped up m" in out.text
    pc.rig._arm.swipe_to.assert_called_once()


# ---------- send_to_clipboard ----------


def _wire_failing_bridge(pc, wire_rig, bridge_double):
    """Wire hardware + a bridge whose clipboard sync never confirms."""
    wire_rig(pc.rig)
    bridge = bridge_double()
    bridge.wait_clipboard.return_value = False
    # No late fetch either — the expiry finds the text unfetched.
    bridge.expire_text.return_value = False
    pc.rig.attach_bridge(bridge)
    return bridge


def test_send_to_clipboard_happy_path(pc: PhysiClaw, wire_rig, bridge_double) -> None:
    wire_rig(pc.rig)
    bridge = bridge_double()
    bridge.wait_clipboard.return_value = True
    pc.rig.attach_bridge(bridge)

    out = pc.send_to_clipboard("hello world")

    # Length, not the text itself: the echoed content would join the
    # verdict-scanned action text on the sequence path (see
    # _send_to_clipboard docstring).
    assert "Copied 11 chars" in out
    assert "hello world" not in out
    bridge.send_text.assert_called_once_with("hello world")
    pc.rig._assistive_touch.long_press.assert_called_once()


def test_send_to_clipboard_unconfirmed_raises(
    pc: PhysiClaw, wire_rig, bridge_double
) -> None:
    # Raise, not return: a returned message once let `sequence` mark the
    # step "ok" and paste+send the phone's STALE clipboard into an IM.
    bridge = _wire_failing_bridge(pc, wire_rig, bridge_double)

    with pytest.raises(ClipboardSyncError, match="do NOT paste"):
        pc.send_to_clipboard("x")

    bridge.wait_clipboard.assert_called_once_with(timeout=pc._clipboard.CONFIRM_SECONDS)


def test_send_to_clipboard_miss_expires_queued_text(
    pc: PhysiClaw, wire_rig, bridge_double
) -> None:
    # A LATE Shortcut run must not fetch the text after we've told the
    # agent "the phone clipboard still holds the previous content" —
    # expiry is atomic with the fetch (BridgeState.expire_text), not a
    # separate clear_text() a fetch could race.
    bridge = _wire_failing_bridge(pc, wire_rig, bridge_double)

    with pytest.raises(ClipboardSyncError):
        pc.send_to_clipboard("x")

    bridge.expire_text.assert_called_once()


def test_send_to_clipboard_late_fetch_counts_as_success(
    pc: PhysiClaw, wire_rig, bridge_double
) -> None:
    # The Shortcut fetched the text after wait_clipboard timed out but
    # before the expiry — the phone HAS the text, so no error.
    bridge = _wire_failing_bridge(pc, wire_rig, bridge_double)
    bridge.expire_text.return_value = True

    assert "Copied" in pc.send_to_clipboard("x")


def test_send_to_clipboard_success_resets_miss_counter(
    pc: PhysiClaw, wire_rig, bridge_double
) -> None:
    # Escalation/decay/timeout policy is unit-tested in test_clipboard.py —
    # this pins the wiring: a miss is recorded, a confirmed sync resets it,
    # and the state's window reaches wait_clipboard on every call.
    wire_rig(pc.rig)
    bridge = bridge_double()
    bridge.wait_clipboard.side_effect = [False, True, False]
    bridge.expire_text.return_value = False  # misses stay misses
    pc.rig.attach_bridge(bridge)

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


def test_run_step_dispatches_each_tool(pc: PhysiClaw, wire_rig) -> None:
    wire_rig(pc.rig)

    with pc.rig.locked():
        assert "Tapped" in pc._run_step("tap", [0.1, 0.1, 0.2, 0.2])
        assert "Double tapped" in pc._run_step("double_tap", [0.1, 0.1, 0.2, 0.2])
        assert "Long pressed" in pc._run_step("long_press", [0.1, 0.1, 0.2, 0.2])


def test_run_step_swipe_happy_path(pc: PhysiClaw, wire_rig) -> None:
    wire_rig(pc.rig)

    with pc.rig.locked():
        out = pc._run_step(
            "swipe",
            {
                "bbox": [0.1, 0.1, 0.2, 0.2],
                "direction": "up",
            },
        )

    assert "Swiped up m" in out


def test_run_step_send_to_clipboard_dispatches(
    pc: PhysiClaw, wire_rig, bridge_double
) -> None:
    wire_rig(pc.rig)
    bridge = bridge_double()
    bridge.wait_clipboard.return_value = True
    pc.rig.attach_bridge(bridge)

    with pc.rig.locked():
        out = pc._run_step("send_to_clipboard", "hi")

    assert "Copied 2 chars" in out


def test_run_step_unknown_tool_raises(pc: PhysiClaw) -> None:
    # Parse-error unit tests live in test_gestures.py — this pins the
    # wiring: validator errors surface from `_run_step` (and so abort a
    # `sequence` step).
    with pytest.raises(ValueError, match="not allowed in sequence"):
        pc._run_step("delete_app", "anything")


def test_sequence_runs_steps_in_order(pc: PhysiClaw, wire_rig) -> None:
    wire_rig(pc.rig)

    out = pc.sequence(
        [
            {"tool_name": "tap", "arg": [0.1, 0.1, 0.2, 0.2]},
            {"tool_name": "double_tap", "arg": [0.3, 0.3, 0.4, 0.4]},
        ]
    )

    lines = out.text.splitlines()
    assert lines[0].startswith("1 tap ok")
    assert lines[1].startswith("2 double_tap ok")


def test_sequence_stops_on_first_failure(pc: PhysiClaw, wire_rig) -> None:
    wire_rig(pc.rig)
    pc.rig._arm.tap.side_effect = [None, RuntimeError("arm jammed")]

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


def test_sequence_step_missing_tool_name_fails_that_step_only(
    pc: PhysiClaw, wire_rig
) -> None:
    # The schema is only list[dict] — a step without "tool_name" must
    # surface as a per-step FAIL line (keeping prior steps' ok lines and
    # the fused view), not a bare KeyError that discards the whole
    # batch's report.
    wire_rig(pc.rig)

    out = pc.sequence(
        [
            {"tool_name": "tap", "arg": [0.1, 0.1, 0.2, 0.2]},
            {"arg": [0.3, 0.3, 0.4, 0.4]},
        ]
    )

    lines = out.text.splitlines()
    assert lines[0].startswith("1 tap ok")
    assert lines[1].startswith("2 ? FAIL")
    assert "tool_name" in lines[1]
    pc.rig._arm.tap.assert_called_once()  # step 1 ran; the batch stopped


def test_sequence_non_dict_step_fails_that_step_only(pc: PhysiClaw, wire_rig) -> None:
    # Same contract for outright junk: a non-dict step is a per-step
    # FAIL line, never an escaping TypeError.
    wire_rig(pc.rig)

    out = pc.sequence([{"tool_name": "tap", "arg": [0.1, 0.1, 0.2, 0.2]}, "junk"])

    lines = out.text.splitlines()
    assert lines[0].startswith("1 tap ok")
    assert lines[1].startswith("2 ? FAIL")


def test_sequence_aborts_before_paste_on_unconfirmed_clipboard(
    pc: PhysiClaw, wire_rig, bridge_double
) -> None:
    # The 18:36 incident shape: [send_to_clipboard, long_press, tap Paste,
    # tap Send]. An unconfirmed sync must kill the batch so the stale
    # clipboard is never pasted and sent.
    _wire_failing_bridge(pc, wire_rig, bridge_double)

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
    pc.rig._arm.tap.assert_not_called()
    pc.rig._arm.long_press.assert_not_called()


# ---------- screen-change verdict ----------


def _wire_verdict_frames(mocker, pc: PhysiClaw, frames: list[np.ndarray]) -> None:
    """Route the observer's grabs through controlled frames — first call
    returns frames[0] (before), second frames[1] (after). Detection on
    the after-frame is stubbed so tests don't need the ONNX model."""
    pc.rig._cam.snapshot.return_value = np.zeros((4, 4, 3), dtype=np.uint8)
    mocker.patch.object(perception, "crop_to_phone_screen", side_effect=frames)
    mocker.patch.object(pc.perception, "detect", return_value=("LISTING", MagicMock()))
    mocker.patch.object(observation, "encode_view_jpeg", return_value=b"VIEW_JPG")


def test_tap_appends_changed_verdict(mocker, pc: PhysiClaw, wire_rig) -> None:
    wire_rig(pc.rig)
    before = np.full((200, 100, 3), 128, dtype=np.uint8)
    after = np.full((200, 100, 3), 30, dtype=np.uint8)
    _wire_verdict_frames(mocker, pc, [before, after])

    out = pc.tap([0.1, 0.1, 0.2, 0.2])

    assert out.text.endswith("| screen: changed")


def test_tap_appends_unchanged_verdict(mocker, pc: PhysiClaw, wire_rig) -> None:
    wire_rig(pc.rig)
    frame = np.full((200, 100, 3), 128, dtype=np.uint8)
    _wire_verdict_frames(mocker, pc, [frame, frame.copy()])

    out = pc.tap([0.1, 0.1, 0.2, 0.2])

    assert out.text.endswith("| screen: no visible change")


def test_tap_unmarked_when_camera_fails(mocker, pc: PhysiClaw, wire_rig) -> None:
    # Fail open: no frames → no marker, gesture result intact.
    wire_rig(pc.rig)
    pc.rig._cam.snapshot.return_value = None

    out = pc.tap([0.1, 0.1, 0.2, 0.2])

    assert out.text == "Tapped at bbox [0.1, 0.1, 0.2, 0.2]"
    assert out.jpeg is None and out.listing is None
    pc.rig._arm.tap.assert_called_once()


def test_sequence_appends_whole_batch_verdict(mocker, pc: PhysiClaw, wire_rig) -> None:
    wire_rig(pc.rig)
    before = np.full((200, 100, 3), 128, dtype=np.uint8)
    after = np.full((200, 100, 3), 30, dtype=np.uint8)
    _wire_verdict_frames(mocker, pc, [before, after])

    out = pc.sequence([{"tool_name": "tap", "arg": [0.1, 0.1, 0.2, 0.2]}])

    assert out.text.splitlines()[0].startswith("1 tap ok")
    assert out.text.endswith("| screen: changed")


def test_go_back_unchanged_verdict_signals_no_pop(
    mocker, pc: PhysiClaw, wire_rig
) -> None:
    wire_rig(pc.rig)
    frame = np.full((200, 100, 3), 128, dtype=np.uint8)
    _wire_verdict_frames(mocker, pc, [frame, frame.copy()])

    out = pc.go_back()

    assert out.text.endswith("| screen: no visible change")


def test_grab_screen_wiring_retries_through_orchestrator(
    mocker, pc: PhysiClaw, wire_rig
) -> None:
    # The observer's blur-retry drives the rig-park + perception-crop
    # wiring — a blurry first crop grabs a second one. (The retry logic
    # itself is unit-tested in test_observation.py.)
    wire_rig(pc.rig)
    pc._observer.GRAB_BLUR_THRESHOLD = 50.0
    pc.rig._cam.snapshot.return_value = np.zeros((4, 4, 3), dtype=np.uint8)
    rng = np.random.default_rng(7)
    sharp = rng.integers(0, 255, size=(200, 100, 3), dtype=np.uint8)
    blurry = np.full((200, 100, 3), 128, dtype=np.uint8)
    mocker.patch.object(perception, "crop_to_phone_screen", side_effect=[blurry, sharp])
    mocker.patch.object(observation.time, "sleep")

    frame, sharp_flag, _report, _retuned = pc._observer.grab_screen()

    assert frame is sharp
    assert sharp_flag is True
    assert perception.crop_to_phone_screen.call_count == 2


def test_blurry_after_frame_withholds_verdict_but_keeps_view(
    mocker, pc: PhysiClaw, wire_rig
) -> None:
    # Sharp-vs-blurry frames diff as a full-screen change — the verdict
    # must fail open (no marker), while the view still attaches.
    wire_rig(pc.rig)
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


def test_gesture_returns_fused_view(mocker, pc: PhysiClaw, wire_rig) -> None:
    # The after-frame that feeds the verdict also feeds detection: the
    # gesture result carries the annotated JPEG + listing like a peek.
    wire_rig(pc.rig)
    before = np.full((200, 100, 3), 128, dtype=np.uint8)
    after = np.full((200, 100, 3), 30, dtype=np.uint8)
    _wire_verdict_frames(mocker, pc, [before, after])

    out = pc.tap([0.1, 0.1, 0.2, 0.2])

    assert out.jpeg == b"VIEW_JPG"
    assert out.listing == "LISTING"


def test_gesture_view_fails_open_when_detection_raises(
    mocker, pc: PhysiClaw, wire_rig
) -> None:
    wire_rig(pc.rig)
    before = np.full((200, 100, 3), 128, dtype=np.uint8)
    after = np.full((200, 100, 3), 30, dtype=np.uint8)
    pc.rig._cam.snapshot.return_value = np.zeros((4, 4, 3), dtype=np.uint8)
    mocker.patch.object(perception, "crop_to_phone_screen", side_effect=[before, after])
    mocker.patch.object(
        pc.perception, "detect", side_effect=RuntimeError("model missing")
    )

    out = pc.tap([0.1, 0.1, 0.2, 0.2])

    # Verdict still attaches; only the view is absent.
    assert out.text.endswith("| screen: changed")
    assert out.jpeg is None and out.listing is None


def test_gesture_view_absent_without_after_frame(
    mocker, pc: PhysiClaw, wire_rig
) -> None:
    wire_rig(pc.rig)
    pc.rig._cam.snapshot.return_value = None  # camera down

    out = pc.tap([0.1, 0.1, 0.2, 0.2])

    assert out.text == "Tapped at bbox [0.1, 0.1, 0.2, 0.2]"
    assert out.jpeg is None and out.listing is None


# ---------- macro gestures ----------


def test_home_screen_swipes_up(pc: PhysiClaw, wire_rig) -> None:
    wire_rig(pc.rig)

    out = pc.home_screen()

    assert "Went to home screen" in out.text
    pc.rig._arm.swipe_to.assert_called_once()


def test_go_back_swipes_right(pc: PhysiClaw, wire_rig) -> None:
    wire_rig(pc.rig)

    out = pc.go_back()

    assert "Went back" in out.text


def test_go_back_dwells_at_the_edge(pc: PhysiClaw, wire_rig) -> None:
    # The interactive-pop recognizer only arms when the touch-down is
    # seen inside the edge zone — the recipe's start_dwell must reach
    # arm.swipe_to.
    wire_rig(pc.rig)

    pc.go_back()

    kwargs = pc.rig._arm.swipe_to.call_args.kwargs
    assert kwargs["start_dwell"] == gestures.BACK_EDGE_DWELL_SECONDS


def test_force_quit_runs_the_four_step_recipe(pc: PhysiClaw, wire_rig) -> None:
    wire_rig(pc.rig)

    out = pc.force_quit()

    assert "Force-quit" in out.text
    # open + card drag + fling, then the dismiss tap.
    assert pc.rig._arm.swipe_to.call_count == 3
    assert pc.rig._arm.tap.call_count == 1


def test_force_quit_open_swipe_holds_before_lift(pc: PhysiClaw, wire_rig) -> None:
    # The pause before the lift is what iOS reads as "switcher" — the
    # recipe's end_dwell must reach arm.swipe_to on the OPEN swipe only.
    wire_rig(pc.rig)

    pc.force_quit()

    dwells = [c.kwargs["end_dwell"] for c in pc.rig._arm.swipe_to.call_args_list]
    assert dwells[0] == gestures.SWITCHER_HOLD_SECONDS
    assert dwells[1:] == [0.0, 0.0]


# ---------- unlock_phone ----------


def _wire_unlock(mocker, pc: PhysiClaw, wire_rig, *, digit_bbox, scan=()):
    """Shared unlock harness: wired rig, warmed OCR double, camera
    settle and keypad poll stubbed, verify scan returning `scan`."""
    wire_rig(pc.rig)
    pc.perception._ocr_reader = MagicMock()
    mocker.patch.object(orchestrator.time, "sleep")
    mocker.patch.object(pc.perception, "ensure_focus_locked")
    mocker.patch.object(pc.perception, "wait_for_numpad_digit", return_value=digit_bbox)
    mocker.patch.object(pc.perception, "scan_text", return_value=list(scan))


def test_unlock_phone_returns_when_keypad_not_found(
    mocker, pc: PhysiClaw, wire_rig
) -> None:
    _wire_unlock(mocker, pc, wire_rig, digit_bbox=None)

    out = pc.unlock_phone()

    assert "Failed to find passcode keypad" in out.text


def test_unlock_phone_settles_focus_before_the_swipe(
    mocker, pc: PhysiClaw, wire_rig
) -> None:
    # The keypad clock starts at the swipe — the lens freeze must come
    # before it, or the settle spends the window it exists to protect.
    _wire_unlock(mocker, pc, wire_rig, digit_bbox=None)
    order: list[str] = []
    pc.perception.ensure_focus_locked.side_effect = lambda: order.append("lock")
    pc.rig._arm.swipe_to.side_effect = lambda *a, **k: order.append("swipe")

    pc.unlock_phone()

    assert order == ["lock", "swipe"]


def test_unlock_phone_taps_six_times_when_keypad_found(
    mocker, pc: PhysiClaw, wire_rig
) -> None:
    _wire_unlock(mocker, pc, wire_rig, digit_bbox=[0.1, 0.1, 0.2, 0.2])

    out = pc.unlock_phone()

    assert "Passcode entered" in out.text
    assert "still shows the lock screen" not in out.text
    # 1 wake-tap + 6 digit-taps = 7 taps.
    assert pc.rig._arm.tap.call_count == 7


def test_unlock_phone_reports_still_locked_when_verify_sees_lock_screen(
    mocker, pc: PhysiClaw, wire_rig
) -> None:
    _wire_unlock(
        mocker,
        pc,
        wire_rig,
        digit_bbox=[0.1, 0.1, 0.2, 0.2],
        scan=[
            {
                "label": "Swipe up for Face ID or Enter Passcode",
                "bbox": [0.06, 0.18, 0.93, 0.22],
            }
        ],
    )

    out = pc.unlock_phone()

    assert "still shows the lock screen" in out.text
    # The taps did run — the verify is what failed.
    assert pc.rig._arm.tap.call_count == 7


def test_unlock_phone_verify_camera_hiccup_still_reports_entered(
    mocker, pc: PhysiClaw, wire_rig
) -> None:
    # The taps already landed — a dropped verify frame must fail toward
    # "entered", not surface as a tool error that steers the agent to
    # STUCK on an unlocked phone (the attached view shows the truth).
    _wire_unlock(mocker, pc, wire_rig, digit_bbox=[0.1, 0.1, 0.2, 0.2])
    pc.perception.scan_text.side_effect = RuntimeError("Camera capture failed")

    out = pc.unlock_phone()

    assert "Passcode entered" in out.text
    assert "still shows the lock screen" not in out.text
    assert pc.rig._arm.tap.call_count == 7


# ---------- camera-quality warning (AF/AE failure) ----------


def test_peek_appends_quality_warning_to_listing(
    mocker, pc: PhysiClaw, wire_rig
) -> None:
    _wire_peek(mocker, pc, wire_rig, listing="rows")
    pc._observer._quality.observe.return_value = "⚠ camera: bad"

    _, listing = pc.peek()

    assert listing == "rows\n⚠ camera: bad"


def test_gesture_view_carries_quality_warning(mocker, pc: PhysiClaw, wire_rig) -> None:
    wire_rig(pc.rig)
    before = np.full((200, 100, 3), 128, dtype=np.uint8)
    after = np.full((200, 100, 3), 30, dtype=np.uint8)
    _wire_verdict_frames(mocker, pc, [before, after])
    pc._observer._quality.observe.return_value = "⚠ camera: bad"

    out = pc.tap([0.1, 0.1, 0.2, 0.2])

    assert out.listing == "LISTING\n⚠ camera: bad"
    assert out.text.endswith("| screen: changed")  # verdict untouched


def test_gesture_warning_rides_action_text_when_detect_fails(
    mocker, pc: PhysiClaw, wire_rig
) -> None:
    # Detection crashing must not silence the observation: a blurry/blown
    # frame is a plausible CAUSE of the failed view, so with no listing to
    # carry the ⚠ line it rides the action text instead.
    wire_rig(pc.rig)
    before = np.full((200, 100, 3), 128, dtype=np.uint8)
    after = np.full((200, 100, 3), 30, dtype=np.uint8)
    _wire_verdict_frames(mocker, pc, [before, after])
    mocker.patch.object(pc.perception, "detect", side_effect=RuntimeError("model gone"))
    pc._observer._quality.observe.return_value = "⚠ camera: bad"

    out = pc.tap([0.1, 0.1, 0.2, 0.2])

    assert out.listing is None
    assert out.text.endswith("\n⚠ camera: bad")
    pc._observer._quality.observe.assert_called_once()


def test_quality_check_failure_never_costs_the_view(
    mocker, pc: PhysiClaw, wire_rig
) -> None:
    wire_rig(pc.rig)
    before = np.full((200, 100, 3), 128, dtype=np.uint8)
    after = np.full((200, 100, 3), 30, dtype=np.uint8)
    _wire_verdict_frames(mocker, pc, [before, after])
    pc._observer._quality.observe.side_effect = RuntimeError("boom")

    out = pc.tap([0.1, 0.1, 0.2, 0.2])

    assert out.listing == "LISTING"  # fail-open: view intact, no line
