"""Tests for `physiclaw.core.orchestration.perception` — Perception.

Lazy detectors, frame acquisition, the watchdog poll, and the exposure
tune. Uses a real HardwareRig with mocked devices (conftest `wire_rig`)
so the lock semantics (tune skips when busy, releases after) are
exercised for real.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from physiclaw.core.orchestration import perception as perception_mod
from physiclaw.core.orchestration.perception import Perception
from physiclaw.core.orchestration.rig import HardwareRig

# ---------- Fixtures ----------


@pytest.fixture
def rig(wire_rig) -> HardwareRig:
    r = HardwareRig()
    wire_rig(r)
    return r


@pytest.fixture
def per(rig: HardwareRig) -> Perception:
    return Perception(rig)


# ---------- lazy detectors ----------


def test_ocr_reader_lazy_caches(mocker, per: Perception) -> None:
    fake = MagicMock()
    spy = mocker.patch.object(perception_mod, "OCRReader", return_value=fake)

    a = per.ocr_reader()
    b = per.ocr_reader()

    assert a is fake
    assert b is fake
    spy.assert_called_once()


def test_icon_detector_lazy_caches(mocker, per: Perception) -> None:
    fake = MagicMock()
    spy = mocker.patch.object(perception_mod, "IconDetector", return_value=fake)

    a = per.icon_detector()
    b = per.icon_detector()

    assert a is fake
    assert b is fake
    spy.assert_called_once()


# ---------- camera_view ----------


def test_camera_view_raises_when_snapshot_none(rig, per: Perception) -> None:
    rig._cam.snapshot.return_value = None

    with pytest.raises(RuntimeError, match="Camera capture failed"):
        per.camera_view()


def test_camera_view_raises_before_camera_connect(per: Perception, rig) -> None:
    # The rig's honest accessor speaks, not an AttributeError on None.
    rig._cam = None

    with pytest.raises(RuntimeError, match="Camera not connected"):
        per.camera_view()


def test_camera_view_returns_frame(rig, per: Perception) -> None:
    frame = np.zeros((1, 1, 3), dtype=np.uint8)
    rig._cam.snapshot.return_value = frame

    assert per.camera_view() is frame


def test_cropped_view_crops_with_current_transforms(
    mocker, rig, per: Perception
) -> None:
    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    cropped = np.zeros((2, 2, 3), dtype=np.uint8)
    rig._cam.snapshot.return_value = frame
    crop = mocker.patch.object(
        perception_mod, "crop_to_phone_screen", return_value=cropped
    )

    assert per.cropped_view() is cropped
    crop.assert_called_once_with(frame, rig.transforms)


# ---------- detect / _ocr_elements ----------


def test_detect_degrades_to_icons_when_ocr_construction_fails(
    mocker, per: Perception
) -> None:
    """A broken OCR install (models raise in __init__, before
    _detect_texts' guard can run) costs the text channel — icons
    survive — and the failure is memoized: later frames neither
    re-attempt the heavy model load nor re-warn."""
    per._icon_detector = MagicMock()
    ctor = mocker.patch.object(
        perception_mod, "OCRReader", side_effect=RuntimeError("corrupt model")
    )
    pipeline = mocker.patch.object(
        perception_mod,
        "detect_ui_elements",
        return_value=([], np.zeros((4, 4, 3), dtype=np.uint8)),
    )
    mocker.patch.object(perception_mod, "format_elements", return_value="")

    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    per.detect(frame)
    per.detect(frame)

    assert ctor.call_count == 1  # memoized — no per-frame model reload
    assert pipeline.call_count == 2  # both passes still detected icons
    ocr_arg = pipeline.call_args.kwargs["ocr_reader"]
    assert ocr_arg.read(frame) == []  # the null reader, not None


def test_detect_calls_ui_pipeline(mocker, per: Perception) -> None:
    per._ocr_reader = MagicMock()
    per._icon_detector = MagicMock()
    elements = [{"id": 0}]
    annotated = np.zeros((4, 4, 3), dtype=np.uint8)
    mocker.patch.object(
        perception_mod,
        "detect_ui_elements",
        return_value=(elements, annotated),
    )
    mocker.patch.object(perception_mod, "format_elements", return_value="LISTING")

    listing, ann = per.detect(np.zeros((4, 4, 3), dtype=np.uint8))

    assert listing == "LISTING"
    assert ann is annotated


def test_ocr_elements_filters_offscreen(mocker, rig, per: Perception) -> None:
    per._ocr_reader = MagicMock()
    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    mocker.patch.object(perception_mod, "phone_screen_crop_box", return_value=None)
    mocker.patch.object(
        perception_mod,
        "results_to_elements",
        return_value=[
            {"bbox": [0.1, 0.1, 0.2, 0.2]},
            {"bbox": [-1.0, -1.0, -0.5, -0.5]},
        ],
    )
    mocker.patch.object(
        perception_mod,
        "bbox_on_screen",
        side_effect=lambda b: b[0] >= 0,
    )

    out = per._ocr_elements(frame)

    assert len(out) == 1
    assert out[0]["bbox"][0] == 0.1


def _wire_numpad_poll(mocker, rig):
    """Stub the poll's acquisition — a stock camera frame and patched
    sleep (returned for cadence assertions). Tests stub `_ocr_elements`
    / `find_numpad_digit` for the detection result."""
    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    rig._cam.snapshot.return_value = frame
    return mocker.patch.object(perception_mod.time, "sleep")


def test_wait_for_numpad_digit_polls_until_found(mocker, rig, per: Perception) -> None:
    _wire_numpad_poll(mocker, rig)
    bbox = [0.1, 0.1, 0.2, 0.2]
    ocr = mocker.patch.object(
        per, "_ocr_elements", side_effect=[[], [], [{"hit": True}]]
    )
    mocker.patch.object(
        perception_mod,
        "find_numpad_digit",
        side_effect=lambda els, d: bbox if els else None,
    )

    rig.acquire()
    out = per.wait_for_numpad_digit("1")
    rig.release()

    assert out == bbox
    assert ocr.call_count == 3


def test_wait_for_numpad_digit_gives_up_at_timeout(
    mocker, rig, per: Perception
) -> None:
    sleep = _wire_numpad_poll(mocker, rig)
    mocker.patch.object(perception_mod, "find_numpad_digit", return_value=None)
    # Fake clock: a real monotonic would busy-spin the no-op-sleep loop
    # for the whole timeout, recording thousands of mock calls for the
    # same deterministic iterations. The sleep side-effect advances the
    # clock past the timeout so the loop terminates.
    clock = {"now": 0.0}
    mocker.patch.object(
        perception_mod.time, "monotonic", side_effect=lambda: clock["now"]
    )
    sleep.side_effect = lambda s: clock.__setitem__("now", clock["now"] + s)

    def ocr_pass(frame, max_edge=None) -> list:
        clock["now"] += 0.03
        return []

    ocr = mocker.patch.object(per, "_ocr_elements", side_effect=ocr_pass)

    rig.acquire()
    out = per.wait_for_numpad_digit("1", timeout=0.05)
    rig.release()

    assert out is None
    assert ocr.called


# ---------- watch ----------


def test_watch_returns_no_wake_when_frame_none(rig, per: Perception) -> None:
    rig._cam.peek.return_value = None

    out = per.watch()

    assert out == {"wake": False, "reason": ""}


def test_watch_polls_watchdog_with_frame(rig, per: Perception) -> None:
    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    rig._cam.peek.return_value = frame
    per._watchdog = MagicMock()
    per._watchdog.poll.return_value = {"wake": True, "reason": "screen change"}

    out = per.watch()

    assert out == {"wake": True, "reason": "screen change"}
    per._watchdog.poll.assert_called_once()


# ---------- tune_exposure ----------


def _tune_ok() -> object:
    from physiclaw.core.hardware.exposure import TuneResult

    return TuneResult(mode="auto", exposure=None, ok=True, detail="in band")


def test_tune_exposure_noop_without_camera(mocker, rig, per: Perception) -> None:
    rig._cam = None
    conv = mocker.patch.object(perception_mod.exposure, "converge")

    per.tune_exposure()  # no camera — silently returns

    conv.assert_not_called()


def test_tune_exposure_noop_when_platform_not_tunable(
    mocker, rig, per: Perception
) -> None:
    rig._cam.exposure_tunable = False
    conv = mocker.patch.object(perception_mod.exposure, "converge")

    per.tune_exposure()  # the macOS path

    conv.assert_not_called()


def test_tune_exposure_skips_when_busy(mocker, rig, per: Perception) -> None:
    rig._cam.exposure_tunable = True
    conv = mocker.patch.object(perception_mod.exposure, "converge")
    rig.acquire()
    try:
        per.tune_exposure()
    finally:
        rig.release()

    conv.assert_not_called()


def test_tune_exposure_runs_converge_and_releases_lock(
    mocker, rig, per: Perception
) -> None:
    rig._cam.exposure_tunable = True
    conv = mocker.patch.object(
        perception_mod.exposure,
        "converge",
        return_value=_tune_ok(),
    )

    per.tune_exposure()

    conv.assert_called_once()
    kwargs = conv.call_args.kwargs
    from physiclaw.common.config import CONFIG

    assert kwargs["start"] == CONFIG.camera.exposure
    assert kwargs["prefer_auto"] == CONFIG.camera.auto_exposure
    # Setters are the camera's own bound methods.
    args = conv.call_args.args
    assert args[1] == rig._cam.set_auto_exposure
    assert args[2] == rig._cam.set_manual_exposure
    rig.acquire()  # lock released after the tune
    rig.release()


def test_tune_exposure_meter_crops_and_assesses(mocker, rig, per: Perception) -> None:
    rig._cam.exposure_tunable = True
    frame = np.full((200, 100, 3), 128, dtype=np.uint8)
    rig._cam.wait_frames.return_value = True
    rig._cam.peek.return_value = frame
    cropped = np.full((100, 50, 3), 128, dtype=np.uint8)
    mocker.patch.object(perception_mod, "crop_to_phone_screen", return_value=cropped)
    assess = mocker.patch.object(perception_mod.quality, "assess")
    conv = mocker.patch.object(
        perception_mod.exposure,
        "converge",
        return_value=_tune_ok(),
    )

    per.tune_exposure()

    meter = conv.call_args.args[0]
    report = meter()
    assess.assert_called_once_with(cropped)
    assert report is assess.return_value


def test_tune_exposure_meter_fails_open_on_stalled_reader(
    mocker, rig, per: Perception
) -> None:
    rig._cam.exposure_tunable = True
    rig._cam.wait_frames.return_value = False  # reader stalled
    conv = mocker.patch.object(
        perception_mod.exposure,
        "converge",
        return_value=_tune_ok(),
    )

    per.tune_exposure()

    assert conv.call_args.args[0]() is None


def test_tune_exposure_swallows_converge_crash(mocker, rig, per: Perception) -> None:
    rig._cam.exposure_tunable = True
    mocker.patch.object(
        perception_mod.exposure,
        "converge",
        side_effect=RuntimeError("boom"),
    )

    per.tune_exposure()  # no raise

    rig.acquire()  # and the lock is not leaked
    rig.release()


# ---------- re-tune policy (on_quality_report) ----------


def _report(median: float = 120.0, clip: float = 0.0) -> object:
    from physiclaw.core.vision.quality import QualityReport

    return QualityReport(sharpness=500.0, clip_pct=clip, median_luma=median)


def _deferred() -> object:
    from physiclaw.core.hardware.exposure import TuneResult

    return TuneResult(
        mode="auto", exposure=None, ok=False, detail="deferred", deferred=True
    )


BLOWN_REPORT = _report(median=150.0, clip=0.3)


def test_on_quality_noop_when_not_tunable(mocker, rig, per: Perception) -> None:
    rig._cam.exposure_tunable = False
    sched = mocker.patch.object(per, "_schedule_retune")

    per.on_quality_report(BLOWN_REPORT, streak=3)

    sched.assert_not_called()


def test_needs_inline_fix_true_on_blown_view(rig, per: Perception) -> None:
    rig._cam.exposure_tunable = True

    assert per.needs_inline_fix(BLOWN_REPORT) is True


def test_needs_inline_fix_true_on_bright_view_after_deferred_tune(
    rig, per: Perception
) -> None:
    rig._cam.exposure_tunable = True
    per._last_tune = _deferred()

    assert per.needs_inline_fix(_report(median=120.0)) is True


def test_needs_inline_fix_false_on_dark_view_after_deferred_tune(
    rig, per: Perception
) -> None:
    # Still no reference — the lock screen is what deferred the tune.
    rig._cam.exposure_tunable = True
    per._last_tune = _deferred()

    assert per.needs_inline_fix(_report(median=5.0)) is False


def test_needs_inline_fix_false_on_clean_view_after_completed_tune(
    rig, per: Perception
) -> None:
    rig._cam.exposure_tunable = True
    per._last_tune = _tune_ok()

    assert per.needs_inline_fix(_report(median=120.0)) is False


def test_needs_inline_fix_false_when_not_tunable(rig, per: Perception) -> None:
    rig._cam.exposure_tunable = False

    assert per.needs_inline_fix(BLOWN_REPORT) is False


def test_single_blown_view_schedules_retune(mocker, rig, per: Perception) -> None:
    # One washed-out view is already proof the held exposure is wrong for
    # the current screen — no streak required (auto's failure mode
    # alternates with content, so a streak may never form). Thrash is
    # bounded by _schedule_retune's single-flight guard, not by counting.
    rig._cam.exposure_tunable = True
    sched = mocker.patch.object(per, "_schedule_retune")

    per.on_quality_report(BLOWN_REPORT, streak=1)

    sched.assert_called_once()


def test_bright_view_after_deferred_tune_schedules_retune(
    mocker, rig, per: Perception
) -> None:
    rig._cam.exposure_tunable = True
    per._last_tune = _deferred()
    sched = mocker.patch.object(per, "_schedule_retune")

    per.on_quality_report(_report(median=120.0), streak=0)

    sched.assert_called_once()


def test_dark_view_after_deferred_tune_keeps_waiting(
    mocker, rig, per: Perception
) -> None:
    # Still no reference — the lock screen is what deferred the tune.
    rig._cam.exposure_tunable = True
    per._last_tune = _deferred()
    sched = mocker.patch.object(per, "_schedule_retune")

    per.on_quality_report(_report(median=5.0), streak=0)

    sched.assert_not_called()


def test_good_view_after_completed_tune_schedules_nothing(
    mocker, rig, per: Perception
) -> None:
    rig._cam.exposure_tunable = True
    per._last_tune = _tune_ok()
    sched = mocker.patch.object(per, "_schedule_retune")

    per.on_quality_report(_report(median=120.0), streak=0)

    sched.assert_not_called()


def test_schedule_retune_runs_tune_on_a_thread_once(
    mocker, rig, per: Perception
) -> None:
    mocker.patch.object(perception_mod.time, "sleep")
    tune = mocker.patch.object(per, "tune_exposure")
    thread = mocker.patch.object(perception_mod.threading, "Thread")

    per._schedule_retune("test")
    per._schedule_retune("test again")  # single-flight: second is a no-op

    thread.assert_called_once()
    thread.return_value.start.assert_called_once()
    body = thread.call_args.kwargs["target"]
    body()  # the thread body: runs the tune, then clears the pending flag
    tune.assert_called_once()

    per._schedule_retune("after completion")  # pending flag was cleared
    assert thread.call_count == 2


def test_tune_exposure_records_result_for_the_retune_policy(
    mocker, rig, per: Perception
) -> None:
    rig._cam.exposure_tunable = True
    result = _deferred()
    mocker.patch.object(perception_mod.exposure, "converge", return_value=result)

    per.tune_exposure()

    assert per._last_tune is result


def test_tune_now_runs_while_rig_lock_is_held(mocker, rig, per: Perception) -> None:
    # The inline fix runs mid-grab, INSIDE rig.engaged() — unlike
    # tune_exposure it must not try to acquire (that would busy-skip).
    rig._cam.exposure_tunable = True
    conv = mocker.patch.object(
        perception_mod.exposure, "converge", return_value=_tune_ok()
    )
    rig.acquire()
    try:
        per.tune_now()
    finally:
        rig.release()

    conv.assert_called_once()


# ---------- focus: no runtime machinery ----------


def _blurry_report() -> object:
    from physiclaw.core.vision.quality import QualityReport

    return QualityReport(sharpness=20.0, clip_pct=0.0, median_luma=120.0)


def test_blurry_views_schedule_nothing(mocker, rig, per: Perception) -> None:
    # The lens is pinned at the calibrated position (rig.connect_camera);
    # persistent blur means the rig physically changed — recalibrate,
    # never a runtime adjustment.
    rig._cam.exposure_tunable = True
    sched = mocker.patch.object(per, "_schedule_retune")

    per.on_quality_report(_blurry_report(), streak=10)

    sched.assert_not_called()


# ---------- settle_camera ----------


def test_settle_camera_tunes_under_the_rig_hold_and_releases(
    mocker, rig, per: Perception
) -> None:
    tune = mocker.patch.object(per, "tune_now")

    per.settle_camera()

    tune.assert_called_once()
    rig.acquire()  # released after the settle
    rig.release()


def test_settle_camera_noop_without_camera(mocker, rig, per: Perception) -> None:
    rig._cam = None
    tune = mocker.patch.object(per, "tune_now")

    per.settle_camera()

    tune.assert_not_called()


def test_settle_camera_skips_when_busy(mocker, rig, per: Perception) -> None:
    tune = mocker.patch.object(per, "tune_now")
    rig.acquire()
    try:
        per.settle_camera()
    finally:
        rig.release()

    tune.assert_not_called()


def test_dark_view_schedules_retune(mocker, rig, per: Perception) -> None:
    # A stale bright-scene exposure crushes a night-dimmed screen to
    # black — dark views must trigger the correction machinery.
    rig._cam.exposure_tunable = True
    sched = mocker.patch.object(per, "_schedule_retune")

    per.on_quality_report(_report(median=5.0), streak=1)

    sched.assert_called_once()


def test_stuck_dark_scene_schedules_nothing_under_the_hold(
    mocker, rig, per: Perception
) -> None:
    # The deferral recorded the scene's darkness — an unchanged scene
    # re-proves nothing (the thrash bound for sessions staring at a
    # sleeping phone).
    from physiclaw.core.hardware import exposure as exposure_mod

    rig._cam.exposure_tunable = True
    per._last_tune = exposure_mod.TuneResult(
        "auto", None, False, "dark", deferred=True, median=7.0
    )
    sched = mocker.patch.object(per, "_schedule_retune")

    per.on_quality_report(_report(median=8.0), streak=5)

    sched.assert_not_called()


def test_tune_crash_records_a_deferred_sentinel(mocker, rig, per: Perception) -> None:
    # Without a recorded result the dark-hold has no state — a flaky
    # camera on a dark scene would re-fire the tune on every view.
    rig._cam.exposure_tunable = True
    mocker.patch.object(
        perception_mod.exposure, "converge", side_effect=RuntimeError("boom")
    )
    rig.acquire()
    try:
        per.tune_now()
    finally:
        rig.release()

    assert per._last_tune is not None and per._last_tune.deferred


def test_scheduled_retune_superseded_by_a_fresher_tune(
    mocker, rig, per: Perception
) -> None:
    # An inline fix that lands while the background task waits makes its
    # conclusion fresher than the trigger — re-running the probe would
    # re-learn the same answer seconds later.
    mocker.patch.object(perception_mod.time, "sleep")
    tune = mocker.patch.object(per, "tune_exposure")
    thread = mocker.patch.object(perception_mod.threading, "Thread")

    per._schedule_retune("test")
    per._last_tune = perception_mod.exposure.TuneResult("auto", None, True, "fresh")
    thread.call_args.kwargs["target"]()

    tune.assert_not_called()
