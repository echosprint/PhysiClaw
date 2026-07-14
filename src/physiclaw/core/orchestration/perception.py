"""Perception — the seeing side of the orchestration layer.

Owns the lazily-loaded detectors (OCR, icons), the camera-frame
acquisition helpers, the phone-screen watchdog, and the startup
exposure tune. It never gestures: it borrows the rig for devices,
transforms, parking, and the busy lock, and hands frames/element
listings back to whoever asked (the orchestrator's tool ops, the
observer's callbacks, the /api/phone/watch route).

Pixel work itself still lives in physiclaw.core.vision — this class
only coordinates it.
"""

import logging

from physiclaw.common.config import CONFIG
from physiclaw.core.hardware import exposure
from physiclaw.core.orchestration.rig import HardwareRig
from physiclaw.core.vision import quality
from physiclaw.core.vision.icon_detect import IconDetector
from physiclaw.core.vision.ocr import OCRReader, results_to_elements
from physiclaw.core.vision.preprocess import (
    crop_to_phone_screen,
    phone_screen_crop_box,
)
from physiclaw.core.vision.ui_elements import detect_ui_elements, elements_to_json
from physiclaw.core.vision.util import bbox_on_screen, format_elements
from physiclaw.core.vision.watchdog import Watchdog

log = logging.getLogger(__name__)


class Perception:
    """Detectors + frame acquisition over a borrowed rig."""

    def __init__(self, rig: HardwareRig):
        self._rig = rig
        self._ocr_reader: OCRReader | None = None
        self._icon_detector: IconDetector | None = None
        self._watchdog = Watchdog()

    # ─── Lazy detectors ───────────────────────────────────────

    def ocr_reader(self) -> OCRReader:
        """Lazy-load and cache the OCR reader."""
        if self._ocr_reader is None:
            self._ocr_reader = OCRReader()
        return self._ocr_reader

    def icon_detector(self) -> IconDetector:
        """Lazy-load and cache the icon detector."""
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
        self._rig.park()
        frame = self.camera_view()
        results = self.ocr_reader().read(
            frame, crop_box=phone_screen_crop_box(frame, self._rig.transforms)
        )
        elements = results_to_elements(results, self._rig.transforms)
        return [e for e in elements if bbox_on_screen(e["bbox"])]

    # ─── Watchdog ────────────────────────────────────────────

    def watch(self) -> dict:
        """Poll the camera for wake events. Returns ``{"wake": bool, "reason": str}``."""
        with self._rig.locked():
            frame = self._rig.require_cam().peek()
            if frame is None:
                return {"wake": False, "reason": ""}
            return self._watchdog.poll(frame, self._rig.transforms)

    # ─── Exposure tune ───────────────────────────────────────

    def tune_exposure(self) -> None:
        """Startup verify-and-converge for camera exposure. Fail-open:
        never raises, never blocks a session.

        Runs once the transforms exist (setup finished, or a warm-start
        bundle loaded) and the phone is on the home screen — the dark
        scene that exposed the Windows AE failure. Meters the PHONE-SCREEN
        crop, never the whole frame: the dark desk around the phone would
        otherwise dominate the scene and drive AE to blow out the screen.
        macOS is tunable only when the UVC helper controls the camera
        (AVFoundation ignores exposure props); otherwise it exits at
        `exposure_tunable` and firmware AE stays in charge. Worst case a few seconds, bounded
        by `wait_frames` timeouts; skipped entirely if the hardware is
        busy. A failed tune reverts to auto — the runtime quality monitor
        keeps warning the agent, so nothing is lost."""
        cam, t = self._rig.cam, self._rig.transforms
        if cam is None or t is None or not cam.exposure_tunable:
            return
        try:
            self._rig.acquire()
        except RuntimeError:
            log.info("exposure tune: hardware busy — skipped")
            return
        try:
            self._rig.park()  # stylus off the glass — it would pollute the crop

            def meter():
                if not cam.wait_frames(exposure.SETTLE_FRAMES, timeout=5.0):
                    return None
                frame = cam.peek()
                if frame is None:
                    return None
                return quality.assess(crop_to_phone_screen(frame, t))

            result = exposure.converge(
                meter,
                cam.set_auto_exposure,
                cam.set_manual_exposure,
                start=CONFIG.camera.exposure,
                prefer_auto=CONFIG.camera.auto_exposure,
            )
            log.info("exposure tune: %s", result.detail)
        except Exception:
            log.exception("exposure tune failed — leaving camera as-is")
        finally:
            self._rig.release()
