"""
PhysiClaw orchestrator — central hardware lifecycle manager.

Owns the stylus arm, camera, and calibration state.
Creating an instance connects hardware, finds the camera, and runs calibration.
"""

import logging
import threading

import cv2

from physiclaw.camera import Camera
from physiclaw.stylus_arm import StylusArm
from physiclaw.vision import PhoneDetector

log = logging.getLogger(__name__)


class PhysiClaw:
    """Central orchestrator — owns all hardware lifecycle.

    Construction connects the arm, finds the camera, and runs
    the full calibration workflow. Ready to use immediately after.
    """

    def __init__(self):
        self._arm: StylusArm | None = None
        self._cam: Camera | None = None
        self._detector: PhoneDetector | None = None
        self._lock = threading.Lock()
        self._setup()

    def acquire(self):
        """Mark hardware as busy. Raises immediately if already busy."""
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("PhysiClaw is busy — wait for the current operation to finish, then retry.")

    def release(self):
        """Mark hardware as idle."""
        self._lock.release()

    # ─── Setup ────────────────────────────────────────────────

    def _setup(self):
        """Connect arm, find camera, calibrate.

        1. Connect GRBL arm (auto-detect port)
        2. Find the camera that sees the phone
        3. Open camera
        4. Run calibration
        """
        self._arm = StylusArm()
        self._arm.setup()

        self._detector = PhoneDetector()
        self._cam = self._detector.find_camera()

        if self._cam is None:
            raise RuntimeError("Camera not found — is the phone under the camera?")

        input("\nOpen https://www.physiclaw.ai/pen-calib on the phone, "
              "position the stylus above the center orange circle, "
              "then press Enter...")
        self.calibrate()

    # ─── Calibration ───────────────────────────────────────────

    def calibrate(self):
        """Run the full 5-phase calibration workflow.

        Uses the camera to detect green flashes during probing.
        Sets Z depth and axis mapping directly on the arm instance.
        """
        from physiclaw.calibrate import (
            phase1_z, phase2_right, phase3_down,
            phase4_long_press, phase5_swipe,
        )

        arm = self._arm
        cam = self._cam

        # Phase 1 — Z depth (raises on failure)
        z_tap = phase1_z(arm, cam)
        arm.Z_DOWN = z_tap

        # Phase 2 — find phone-right direction (raises on failure)
        right_result = phase2_right(arm, cam, z_tap)

        # Phase 3 — find phone-down direction (raises on failure)
        right_vec = (right_result[0], right_result[1])
        down_result = phase3_down(arm, cam, z_tap, right_vec)

        # Phase 4 — long press verification
        phase4_long_press(arm, cam)

        # Phase 5 — swipe verification
        rax, ray, _ = right_result
        dax, day, _ = down_result
        arm.set_direction_mapping((rax, ray), (dax, day))
        phase5_swipe(arm, cam)

        log.info(f"Calibration complete — Z={z_tap}mm, "
                 f"right=({rax},{ray}), down=({dax},{day})")

    # ─── Properties ────────────────────────────────────────────

    @property
    def arm(self) -> StylusArm:
        return self._arm

    @property
    def cam(self) -> Camera:
        return self._cam

    # ─── Snapshot helpers ──────────────────────────────────────

    PARK_DISTANCE = 100  # mm to move stylus out of frame

    def _ensure_phone_presence(self, frame):
        """Raise if no phone is detected in the frame."""
        detected, conf, _ = self._detector.detect(frame)
        if not detected:
            raise RuntimeError(f"Phone not detected in frame (confidence {conf:.0%}) "
                               "— is the phone still on the platform?")

    def screenshot(self):
        """Capture a frame from the camera. Returns BGR numpy array.

        Takes the frame as-is — the stylus may be visible.
        Call park() first if an unobstructed view is needed.
        """
        frame = self.cam.snapshot()
        if frame is None:
            raise RuntimeError("Camera capture failed")
        self._ensure_phone_presence(frame)
        return frame

    def park(self):
        """Move the stylus 100mm out of the camera frame."""
        arm = self._arm
        if arm.MOVE_DIRECTIONS is None:
            raise RuntimeError("Cannot park — calibration has not been run yet")
        ux, uy = arm.MOVE_DIRECTIONS['top']
        arm._fast_move(ux * self.PARK_DISTANCE, uy * self.PARK_DISTANCE)
        arm.wait_idle()

    @staticmethod
    def frame_to_jpeg(frame, quality=85) -> bytes:
        """Encode a BGR frame to JPEG bytes."""
        _, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
        return jpeg.tobytes()

    # ─── Lifecycle ─────────────────────────────────────────────

    def shutdown(self):
        if self._arm:
            self._arm._pen_up()
            self._arm._fast_move(0, 0)
            self._arm.wait_idle()
            self._arm.close()
        if self._cam:
            self._cam.close()
