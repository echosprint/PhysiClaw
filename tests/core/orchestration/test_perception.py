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


# ---------- detect / scan_text ----------


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
    mocker.patch.object(perception_mod, "elements_to_json", return_value=[{"id": 0}])
    mocker.patch.object(perception_mod, "format_elements", return_value="LISTING")

    listing, ann = per.detect(np.zeros((4, 4, 3), dtype=np.uint8))

    assert listing == "LISTING"
    assert ann is annotated


def test_scan_text_filters_offscreen(mocker, rig, per: Perception) -> None:
    per._ocr_reader = MagicMock()
    rig._cam.snapshot.return_value = np.zeros((4, 4, 3), dtype=np.uint8)
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

    out = per.scan_text()

    assert len(out) == 1
    assert out[0]["bbox"][0] == 0.1


def test_scan_text_parks_first(mocker, rig, per: Perception) -> None:
    # The stylus must be off the glass before the OCR frame is grabbed.
    per._ocr_reader = MagicMock()
    rig._cam.snapshot.return_value = np.zeros((4, 4, 3), dtype=np.uint8)
    mocker.patch.object(perception_mod, "phone_screen_crop_box", return_value=None)
    mocker.patch.object(perception_mod, "results_to_elements", return_value=[])

    per.scan_text()

    rig._arm.rapid_to.assert_called_once_with(5.0, 6.0)


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
    # The inline fix runs mid-grab, INSIDE rig.locked() — unlike
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


# ---------- focus lock ----------


def _lock_ok() -> object:
    from physiclaw.core.hardware.focus import LockResult

    return LockResult(locked=True, detail="focus locked (sharpness 400)")


def _lock_deferred() -> object:
    from physiclaw.core.hardware.focus import LockResult

    return LockResult(locked=False, detail="deferred", deferred=True)


def _blurry_report() -> object:
    from physiclaw.core.vision.quality import QualityReport

    return QualityReport(sharpness=20.0, clip_pct=0.0, median_luma=120.0)


def test_lock_focus_noop_without_camera(mocker, rig, per: Perception) -> None:
    rig._cam = None
    lk = mocker.patch.object(perception_mod.focus, "lock")

    per.lock_focus()

    lk.assert_not_called()


def test_lock_focus_noop_when_platform_not_lockable(
    mocker, rig, per: Perception
) -> None:
    rig._cam.focus_lockable = False
    lk = mocker.patch.object(perception_mod.focus, "lock")

    per.lock_focus()  # macOS without UVC, or a fixed-focus camera

    lk.assert_not_called()


def test_lock_focus_skips_when_busy(mocker, rig, per: Perception) -> None:
    rig._cam.focus_lockable = True
    lk = mocker.patch.object(perception_mod.focus, "lock")
    rig.acquire()
    try:
        per.lock_focus()
    finally:
        rig.release()

    lk.assert_not_called()


def test_lock_focus_runs_lock_and_releases_rig(mocker, rig, per: Perception) -> None:
    rig._cam.focus_lockable = True
    lk = mocker.patch.object(perception_mod.focus, "lock", return_value=_lock_ok())

    per.lock_focus()

    lk.assert_called_once()
    args = lk.call_args.args
    # Freeze/unfreeze are the camera's own bound methods.
    assert args[1] == rig._cam.lock_focus
    assert args[2] == rig._cam.unlock_focus
    assert per._last_lock is lk.return_value
    rig.acquire()  # lock released after the attempt
    rig.release()


def test_lock_focus_rerun_af_unlocks_before_relocking(
    mocker, rig, per: Perception
) -> None:
    # The persistent-blur path: the frozen position is stale — AF must
    # re-converge before the re-freeze.
    rig._cam.focus_lockable = True
    mocker.patch.object(perception_mod.focus, "lock", return_value=_lock_ok())

    per.lock_focus(rerun_af=True)

    rig._cam.unlock_focus.assert_called_once()


def test_lock_focus_meter_crops_and_scores_sharpness(
    mocker, rig, per: Perception
) -> None:
    # The focus meter scores ONLY sharpness — laplacian_variance on the
    # crop, not the full assess() report (whose histogram/blob passes
    # would be discarded work here).
    rig._cam.focus_lockable = True
    frame = np.full((200, 100, 3), 128, dtype=np.uint8)
    rig._cam.wait_frames.return_value = True
    rig._cam.peek.return_value = frame
    cropped = np.full((100, 50, 3), 128, dtype=np.uint8)
    mocker.patch.object(perception_mod, "crop_to_phone_screen", return_value=cropped)
    variance = mocker.patch.object(
        perception_mod.quality, "laplacian_variance", return_value=345.0
    )
    lk = mocker.patch.object(perception_mod.focus, "lock", return_value=_lock_ok())

    per.lock_focus()

    meter = lk.call_args.args[0]
    sharpness = meter()
    variance.assert_called_once_with(cropped)
    assert sharpness == 345.0


def test_lock_focus_meter_fails_open_on_stalled_reader(
    mocker, rig, per: Perception
) -> None:
    rig._cam.focus_lockable = True
    rig._cam.wait_frames.return_value = False  # reader stalled
    lk = mocker.patch.object(perception_mod.focus, "lock", return_value=_lock_ok())

    per.lock_focus()

    assert lk.call_args.args[0]() is None


def test_lock_focus_swallows_lock_crash(mocker, rig, per: Perception) -> None:
    rig._cam.focus_lockable = True
    mocker.patch.object(perception_mod.focus, "lock", side_effect=RuntimeError("boom"))

    per.lock_focus()  # no raise

    rig.acquire()  # and the rig lock is not leaked
    rig.release()


# ---------- settle_camera ----------


def test_settle_camera_runs_tune_and_lock_under_one_hold(
    mocker, rig, per: Perception
) -> None:
    # One acquire/park for both phases, exposure before focus (a blown
    # frame's sharpness means nothing to the lock meter).
    calls: list[str] = []
    mocker.patch.object(per, "tune_now", side_effect=lambda: calls.append("tune"))
    mocker.patch.object(per, "lock_focus_now", side_effect=lambda: calls.append("lock"))

    per.settle_camera()

    assert calls == ["tune", "lock"]
    rig.acquire()  # released after the settle
    rig.release()


def test_settle_camera_noop_without_camera(mocker, rig, per: Perception) -> None:
    rig._cam = None
    tune = mocker.patch.object(per, "tune_now")

    per.settle_camera()

    tune.assert_not_called()


def test_settle_camera_busy_skip_seeds_a_deferred_lock(
    mocker, rig, per: Perception
) -> None:
    # A busy startup must not strand the lens on AF forever: the seeded
    # deferred result lets _focus_policy retry on the next sharp view.
    rig._cam.focus_lockable = True
    rig.acquire()
    try:
        per.settle_camera()
    finally:
        rig.release()

    assert per._last_lock is not None and per._last_lock.deferred


def test_settle_camera_busy_skip_seeds_nothing_when_not_lockable(
    mocker, rig, per: Perception
) -> None:
    # A deferred seed on an unlockable rig would schedule doomed
    # re-locks on every sharp view.
    rig._cam.focus_lockable = False
    rig.acquire()
    try:
        per.settle_camera()
    finally:
        rig.release()

    assert per._last_lock is None


# ---------- re-lock policy (_focus_policy via on_quality_report) ----------


def test_no_focus_policy_before_the_startup_lock(mocker, rig, per: Perception) -> None:
    # _last_lock is None until the startup lock runs (or forever, on a
    # rig that can't lock) — no scheduling either way.
    sched = mocker.patch.object(per, "_schedule_refocus")

    per.on_quality_report(_blurry_report(), streak=5)

    sched.assert_not_called()


def test_sharp_view_after_deferred_lock_schedules_relock(
    mocker, rig, per: Perception
) -> None:
    per._last_lock = _lock_deferred()
    sched = mocker.patch.object(per, "_schedule_refocus")

    per.on_quality_report(_report(median=120.0), streak=0)

    sched.assert_called_once()
    assert sched.call_args.kwargs["rerun_af"] is False


def test_soft_view_after_deferred_lock_keeps_waiting(
    mocker, rig, per: Perception
) -> None:
    # Still no converged focus to freeze — the blur is what deferred it.
    per._last_lock = _lock_deferred()
    sched = mocker.patch.object(per, "_schedule_refocus")

    per.on_quality_report(_blurry_report(), streak=1)

    sched.assert_not_called()


def test_blur_streak_under_locked_focus_reruns_af(mocker, rig, per: Perception) -> None:
    from physiclaw.core.vision import quality

    per._last_lock = _lock_ok()
    sched = mocker.patch.object(per, "_schedule_refocus")

    per.on_quality_report(_blurry_report(), streak=quality.PERSIST_AFTER)

    sched.assert_called_once()
    assert sched.call_args.kwargs["rerun_af"] is True


def test_single_blurry_view_under_lock_waits_for_a_streak(
    mocker, rig, per: Perception
) -> None:
    # One blurry frame is a normal transient the grab-retry handles — an
    # unlock/relock cycle costs seconds of rig time and needs a streak.
    per._last_lock = _lock_ok()
    sched = mocker.patch.object(per, "_schedule_refocus")

    per.on_quality_report(_blurry_report(), streak=1)

    sched.assert_not_called()


def test_sharp_view_under_locked_focus_schedules_nothing(
    mocker, rig, per: Perception
) -> None:
    per._last_lock = _lock_ok()
    sched = mocker.patch.object(per, "_schedule_refocus")

    per.on_quality_report(_report(median=120.0), streak=0)

    sched.assert_not_called()


def test_schedule_refocus_runs_lock_on_a_thread_once(
    mocker, rig, per: Perception
) -> None:
    mocker.patch.object(perception_mod.time, "sleep")
    lk = mocker.patch.object(per, "lock_focus")
    thread = mocker.patch.object(perception_mod.threading, "Thread")

    per._schedule_refocus("test", rerun_af=True)
    per._schedule_refocus("test again", rerun_af=True)  # single-flight

    thread.assert_called_once()
    thread.return_value.start.assert_called_once()
    body = thread.call_args.kwargs["target"]
    body()  # runs the lock, then clears the pending flag
    lk.assert_called_once_with(rerun_af=True)

    per._schedule_refocus("after completion", rerun_af=False)
    assert thread.call_count == 2
