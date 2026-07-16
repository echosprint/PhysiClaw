"""Perception — the seeing side of the orchestration layer.

Owns the lazily-loaded detectors (OCR, icons), the camera-frame
acquisition helpers, the phone-screen watchdog, and the startup camera
settle (exposure tune + focus lock). It never gestures: it borrows the
rig for devices,
transforms, parking, and the busy lock, and hands frames/element
listings back to whoever asked (the orchestrator's tool ops, the
observer's callbacks, the /api/phone/watch route).

Pixel work itself still lives in physiclaw.core.vision — this class
only coordinates it.
"""

import logging
import threading
import time

from physiclaw.common.config import CONFIG
from physiclaw.core.hardware import exposure, focus
from physiclaw.core.orchestration.rig import HardwareRig
from physiclaw.core.vision import quality
from physiclaw.core.vision.icon_detect import IconDetector
from physiclaw.core.vision.ocr import OCRReader, results_to_elements
from physiclaw.core.vision.preprocess import (
    crop_to_phone_screen,
    phone_screen_crop_box,
)
from physiclaw.core.vision.ui_elements import detect_ui_elements, elements_to_json
from physiclaw.core.vision.util import (
    bbox_on_screen,
    find_numpad_digit,
    format_elements,
)
from physiclaw.core.vision.watchdog import Watchdog

log = logging.getLogger(__name__)


class Perception:
    """Detectors + frame acquisition over a borrowed rig."""

    # Delay before a scheduled settle task (exposure re-tune, focus
    # re-lock) contends for the hardware: the view that triggered it
    # still holds the rig lock while its gesture finishes parking.
    RETUNE_DELAY_SECONDS = 2.0

    def __init__(self, rig: HardwareRig):
        self._rig = rig
        # Serialize first-use model construction (the watch route and a
        # concurrent tool op could otherwise double-load a heavy model).
        # One lock per detector so loading one model never blocks a
        # caller that wants the other, already-cached one.
        self._ocr_init_lock = threading.Lock()
        self._icon_init_lock = threading.Lock()
        self._ocr_reader: OCRReader | None = None
        self._icon_detector: IconDetector | None = None
        self._watchdog = Watchdog()
        self._last_tune: exposure.TuneResult | None = None
        self._last_lock: focus.LockResult | None = None
        # Single-flight guard for the background settle tasks, keyed by
        # task name ("exposure re-tune" / "focus re-lock").
        self._pending_lock = threading.Lock()
        self._pending: set[str] = set()

    # ─── Lazy detectors ───────────────────────────────────────

    def ocr_reader(self) -> OCRReader:
        """Lazy-load and cache the OCR reader. Double-checked: the warm
        path returns the cached instance without touching the lock."""
        if self._ocr_reader is None:
            with self._ocr_init_lock:
                if self._ocr_reader is None:
                    self._ocr_reader = OCRReader()
        return self._ocr_reader

    def icon_detector(self) -> IconDetector:
        """Lazy-load and cache the icon detector (see ocr_reader)."""
        if self._icon_detector is None:
            with self._icon_init_lock:
                if self._icon_detector is None:
                    self._icon_detector = IconDetector()
        return self._icon_detector

    # ─── Frame acquisition ────────────────────────────────────

    def camera_view(self):
        """Capture a frame from the overhead camera. Returns BGR numpy array.

        Takes the frame as-is — the stylus may be visible.
        Call rig.park() first if an unobstructed view is needed.
        Frame is already rotated to portrait by the camera.
        """
        frame = self._rig.require_cam().snapshot()
        if frame is None:
            raise RuntimeError("Camera capture failed")
        return frame

    def cropped_view(self):
        """The camera view cropped to the phone screen (0-1 span) — the
        frame shape `detect` expects and the observer grabs."""
        return crop_to_phone_screen(self.camera_view(), self._rig.transforms)

    # ─── Detection ────────────────────────────────────────────

    def detect(self, frame) -> tuple[str, object]:
        """Icon detection + OCR on a frame. Caller holds the lock.

        ``frame`` must already span the phone screen (0-1) — camera
        views need to be cropped via ``cropped_view`` first.
        Returns (formatted element listing, annotated frame).
        """
        elements, annotated = detect_ui_elements(
            frame,
            icon_detector=self.icon_detector(),
            ocr_reader=self.ocr_reader(),
        )
        return format_elements(elements_to_json(elements)), annotated

    def scan_text(self) -> list[dict]:
        """OCR-only pass on the phone-screen region. Caller must hold the lock.

        Fast path for internal polling (e.g. unlock_phone's keypad loop).
        The agent-facing tools go through ``detect`` instead, which also
        runs icon detection.
        """
        self._rig.assert_locked()
        self._rig.park()
        frame = self.camera_view()
        results = self.ocr_reader().read(
            frame, crop_box=phone_screen_crop_box(frame, self._rig.transforms)
        )
        elements = results_to_elements(results, self._rig.transforms)
        return [e for e in elements if bbox_on_screen(e["bbox"])]

    def wait_for_numpad_digit(
        self, digit: str, attempts: int = 8, interval: float = 1.0
    ) -> list[float] | None:
        """Poll ``scan_text`` until the passcode keypad shows `digit`;
        return its bbox, or None after `attempts` polls. Caller must hold
        the lock. Used by unlock: Face ID takes a few seconds to fail
        before the keypad appears, so one scan isn't enough."""
        for _ in range(attempts):
            elements = self.scan_text()
            bbox = find_numpad_digit(elements, digit)
            if bbox is not None:
                return bbox
            time.sleep(interval)
        return None

    # ─── Watchdog ────────────────────────────────────────────

    def watch(self) -> dict:
        """Poll the camera for wake events. Returns ``{"wake": bool, "reason": str}``."""
        with self._rig.locked():
            frame = self._rig.require_cam().peek()
            if frame is None:
                return {"wake": False, "reason": ""}
            return self._watchdog.poll(frame, self._rig.transforms)

    # ─── Camera settle (exposure tune + focus lock) ──────────

    def _screen_crop(self, settle_frames: int):
        """Settled phone-screen crop for the tune/lock meters, or None.

        The one home for the meters' acquisition contract: wait out the
        driver's property-apply latency (`settle_frames` fresh frames,
        bounded), take the latest frame, crop to the phone screen —
        never meter the whole frame, the dark desk around the phone
        would dominate the statistics."""
        cam, t = self._rig.cam, self._rig.transforms
        if cam is None or t is None:
            return None
        if not cam.wait_frames(settle_frames, timeout=5.0):
            return None
        frame = cam.peek()
        if frame is None:
            return None
        return crop_to_phone_screen(frame, t)

    def _with_rig_parked(self, name: str, body) -> bool:
        """Acquire the rig, park the arm (a stylus in frame would
        pollute any meter), run `body`, always release. Returns False
        on a busy skip OR a failure — the rig-lock discipline for every
        settle entry point, defined once. The catch below is what makes
        the settle paths' "fail-open, never raises" claim actually true:
        `park()` is a serial arm move that can raise (timeout, alarm),
        and a settle must never take readiness or a session down with it."""
        try:
            self._rig.acquire()
        except RuntimeError:
            log.info("%s: hardware busy — skipped", name)
            return False
        try:
            self._rig.park()
            body()
            return True
        except Exception:
            log.exception("%s: failed — leaving camera as-is", name)
            return False
        finally:
            self._rig.release()

    def settle_camera(self) -> None:
        """Startup settle: verify/converge exposure, then freeze
        autofocus, under ONE rig hold. The single entry point for every
        ready-flipping path (warm start, the /api/ready flip) — the
        exposure-before-focus order matters (a blown frame's sharpness
        means nothing) and lives here instead of in each caller.
        Fail-open: never raises, never blocks a session.

        A skipped settle (rig busy, or the park/meter failed) seeds a
        DEFERRED lock result so the re-lock policy retries on the next
        sharp view — otherwise a skipped startup would leave the lens
        hunting for the whole session."""
        cam, t = self._rig.cam, self._rig.transforms
        if cam is None or t is None:
            return
        # Probe BEFORE the settle: if the settle fails because the camera
        # channel is momentarily wedged, a post-failure probe could read
        # False and skip the seed, disabling the re-lock policy for the
        # whole session.
        lockable = cam.focus_lockable

        def body() -> None:
            self.tune_now()
            self.lock_focus_now()

        if not self._with_rig_parked("camera settle", body) and lockable:
            self._last_lock = focus.LockResult(
                False, "startup lock skipped (busy or failed)", deferred=True
            )

    def tune_exposure(self) -> None:
        """Standalone verify-and-converge for camera exposure — the
        background re-tune target. Fail-open: never raises; skipped
        when the hardware is busy (the trigger conditions persist, so
        it gets re-scheduled).

        Meters the PHONE-SCREEN crop, never the whole frame (see
        `_screen_crop`). macOS is tunable only when the UVC helper
        controls the camera (AVFoundation ignores exposure props);
        otherwise it exits at `exposure_tunable` and firmware AE stays
        in charge. Worst case a few seconds, bounded by `wait_frames`
        timeouts. A failed tune reverts to auto — the runtime quality
        monitor keeps warning the agent, so nothing is lost."""
        cam, t = self._rig.cam, self._rig.transforms
        if cam is None or t is None or not cam.exposure_tunable:
            return
        self._with_rig_parked("exposure tune", self.tune_now)

    def tune_now(self) -> None:
        """Verify-and-converge with the rig lock ALREADY HELD and the arm
        parked — the synchronous path: the observer calls this mid-grab
        whenever `needs_inline_fix` flags a view (washed out, or the
        first bright reference after a deferred cold-start tune), so the
        corrected frame ships to the agent instead of a mis-exposed one
        with a warning. Fail-open: never raises, no-op when the rig
        can't tune."""
        self._rig.assert_locked()
        cam, t = self._rig.cam, self._rig.transforms
        if cam is None or t is None or not cam.exposure_tunable:
            return
        try:

            def meter():
                crop = self._screen_crop(exposure.SETTLE_FRAMES)
                return None if crop is None else quality.assess(crop)

            result = exposure.converge(
                meter,
                cam.set_auto_exposure,
                cam.set_manual_exposure,
                start=CONFIG.camera.exposure,
                prefer_auto=CONFIG.camera.auto_exposure,
            )
            self._last_tune = result
            log.info("exposure tune: %s", result.detail)
        except Exception:
            log.exception("exposure tune failed — leaving camera as-is")

    def lock_focus(self, *, rerun_af: bool = False) -> None:
        """Freeze autofocus once it has converged — the background
        re-lock target. Fail-open: never raises; skipped when busy.

        The rig is rigid — camera bolted, screen at a fixed distance —
        so the one position AF converges to stays right, and freezing
        it removes the post-gesture AF hunts the arm's pass over the
        lens keeps triggering (each one risks a blurred view and a
        withheld verdict). A dark or hunting scene defers; the re-lock
        policy retries on the first sharp view. With `rerun_af` the
        lens goes back to AF first — the persistent-blur path: the rig
        was bumped or drifted, so the frozen position itself is stale
        and AF must re-converge before the re-freeze."""
        cam, t = self._rig.cam, self._rig.transforms
        if cam is None or t is None or not cam.focus_lockable:
            return

        def body() -> None:
            if rerun_af:
                cam.unlock_focus()
            self.lock_focus_now()

        self._with_rig_parked("focus lock", body)

    def lock_focus_now(self) -> None:
        """Freeze-and-verify with the rig lock ALREADY HELD and the arm
        parked. Fail-open: never raises, no-op when the rig can't lock.
        The meter scores only sharpness (`laplacian_variance`), skipping
        the histogram/blob statistics `assess` would also compute."""
        self._rig.assert_locked()
        cam, t = self._rig.cam, self._rig.transforms
        if cam is None or t is None or not cam.focus_lockable:
            return
        try:

            def meter():
                crop = self._screen_crop(focus.SETTLE_FRAMES)
                return None if crop is None else quality.laplacian_variance(crop)

            result = focus.lock(meter, cam.lock_focus, cam.unlock_focus)
            self._last_lock = result
            log.info("focus lock: %s", result.detail)
        except Exception:
            log.exception("focus lock failed — leaving autofocus on")

    def needs_inline_fix(self, report: quality.QualityReport) -> bool:
        """Should this view's exposure be fixed before it ships?

        The single predicate behind both the observer's inline fix and
        the background safety net. Two evidence-driven cases:
        - a washed-out view: whatever holds exposure (firmware AE or a
          stale manual value) just proved wrong for the current screen;
        - a bright view while a deferred tune is owed its reference:
          startup metered a dark screen (lock screen / asleep) and
          skipped — tuning on the first bright view, before it ships,
          spares the agent the blinded-firmware-auto interim entirely
          (views that are visibly hot yet under the warning threshold)."""
        cam = self._rig.cam
        if cam is None or not cam.exposure_tunable:
            return False
        if report.blown:
            return True
        return (
            self._last_tune is not None
            and self._last_tune.deferred
            and exposure.is_reference(report)
        )

    def on_quality_report(self, report: quality.QualityReport, streak: int) -> None:
        """Background re-tune/re-lock, fed every judged view by the
        GestureObserver.

        The exposure half is the safety net behind the observer's
        inline fix (same `needs_inline_fix` predicate): the inline path
        normally corrects the view before it ships, so it fires only
        when that fix crashed, was skipped, or didn't recover the
        frame. The focus half is `_focus_policy`. Never blocks the view
        that reported: both run on detached threads (single-flight
        each) and skip themselves if the hardware is busy."""
        if self.needs_inline_fix(report):
            if report.blown:
                self._schedule_retune(
                    f"washed-out view ({report.clip_pct:.0%} clip, streak {streak})"
                )
            else:
                self._schedule_retune("bright view arrived after a deferred tune")
        self._focus_policy(report, streak)

    def _focus_policy(self, report: quality.QualityReport, streak: int) -> None:
        """Re-lock policy — the two evidence-driven cases:
        - a sharp view while a deferred lock is owed its reference:
          startup metered a soft/dark scene and skipped — now there is
          a converged focus to freeze;
        - a persistent blur streak under a LOCKED lens: the rig was
          bumped or drifted, so the frozen position is stale — hand the
          lens back to AF, let it re-converge, freeze again. A streak,
          not one view: a single blurry frame is a normal transient the
          grab-retry handles, and an unlocked→relock cycle costs
          seconds of rig time."""
        last = self._last_lock
        if last is None:
            return  # startup lock hasn't run yet (or the rig can't lock)
        if last.deferred and not report.blurry:
            self._schedule_refocus(
                "sharp view arrived after a deferred lock", rerun_af=False
            )
        elif last.locked and report.blurry and streak >= quality.PERSIST_AFTER:
            self._schedule_refocus(
                f"blur streak {streak} under a locked focus", rerun_af=True
            )

    def _schedule_retune(self, reason: str) -> None:
        self._schedule_settle("exposure re-tune", reason, self.tune_exposure)

    def _schedule_refocus(self, reason: str, *, rerun_af: bool) -> None:
        self._schedule_settle(
            "focus re-lock", reason, lambda: self.lock_focus(rerun_af=rerun_af)
        )

    def _schedule_settle(self, name: str, reason: str, task) -> None:
        """Run a settle task on a detached daemon thread, at most one in
        flight per `name`. The delay lets the triggering view's gesture
        release the rig lock first; a still-busy rig makes the task skip
        harmlessly (the trigger conditions persist, so it will be
        re-scheduled)."""
        with self._pending_lock:
            if name in self._pending:
                return
            self._pending.add(name)
        log.info("%s scheduled: %s", name, reason)

        def run() -> None:
            try:
                time.sleep(self.RETUNE_DELAY_SECONDS)
                task()
            finally:
                with self._pending_lock:
                    self._pending.discard(name)

        threading.Thread(target=run, name=name, daemon=True).start()
