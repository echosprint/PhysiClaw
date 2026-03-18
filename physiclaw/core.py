"""
PhysiClaw orchestrator — central hardware lifecycle manager.

Owns the stylus arm, cameras, and calibration state.
Creating an instance connects hardware, identifies cameras, and runs calibration.
"""

import logging

import cv2

from physiclaw.camera import Camera
from physiclaw.stylus_arm import StylusArm
from physiclaw.vision import PhoneDetector

log = logging.getLogger(__name__)


class PhysiClaw:
    """Central orchestrator — owns all hardware lifecycle.

    Construction connects the arm, identifies cameras, and runs
    the full calibration workflow. Ready to use immediately after.
    """

    def __init__(self):
        self._arm: StylusArm | None = None
        self._top_cam: Camera | None = None
        self._side_cam: Camera | None = None
        self._detector: PhoneDetector | None = None
        self._setup()

    # ─── Setup ────────────────────────────────────────────────

    def _setup(self):
        """Connect arm, identify cameras, calibrate.

        1. Connect GRBL arm (auto-detect port)
        2. Identify top vs side cameras via PhoneDetector
        3. Open cameras
        4. Run calibration
        """
        self._arm = StylusArm()
        self._arm.setup()

        self._detector = PhoneDetector()
        cameras = self._detector.identify_cameras()

        if 'top' not in cameras:
            raise RuntimeError("Top camera not found — is the phone under the camera?")

        self._top_cam = Camera(cameras['top'])
        if 'side' in cameras:
            self._side_cam = Camera(cameras['side'])

        self.calibrate()

    # ─── Calibration ───────────────────────────────────────────

    def calibrate(self):
        """Run the full 5-phase calibration workflow.

        Uses the side camera to detect green flashes during probing.
        Sets Z depth and axis mapping directly on the arm instance.
        """
        from physiclaw.calibrate import (
            phase1_z, phase2_right, phase3_down,
            phase4_long_press, phase5_swipe,
        )

        if self._side_cam is None:
            raise RuntimeError("Side camera not found — needed for calibration")

        arm = self._arm
        cam = self._side_cam

        # Phase 1 — Z depth
        z_tap = phase1_z(arm, cam)
        if z_tap is None:
            raise RuntimeError("Phase 1 failed — no Z contact detected")
        arm.Z_DOWN = z_tap

        # Phase 2 — find phone-right direction
        right_result = phase2_right(arm, cam, z_tap)
        if right_result is None:
            raise RuntimeError("Phase 2 failed — could not find phone-right")

        # Phase 3 — find phone-down direction
        right_vec = (right_result[0], right_result[1])
        down_result = phase3_down(arm, cam, z_tap, right_vec)
        if down_result is None:
            raise RuntimeError("Phase 3 failed — could not find phone-down")

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
    def top_cam(self) -> Camera:
        return self._top_cam

    @property
    def side_cam(self) -> Camera:
        if self._side_cam is None:
            raise RuntimeError("Side camera not available")
        return self._side_cam

    # ─── Snapshot helpers ──────────────────────────────────────

    PARK_DISTANCE = 100  # mm to move stylus out of frame

    def _ensure_phone_presence(self, frame):
        """Raise if no phone is detected in the frame."""
        detected, conf, _ = self._detector.detect(frame)
        if not detected:
            raise RuntimeError(f"Phone not detected in frame (confidence {conf:.0%}) "
                               "— is the phone still on the platform?")

    def snapshot_top(self):
        """Capture a frame from the top camera. Returns BGR numpy array.

        Parks the stylus 100mm in the phone-up direction to avoid
        occluding the screen, takes the snapshot, checks that the
        phone is still visible, then returns the stylus.
        """
        arm = self._arm
        saved_x, saved_y = arm.position()

        # Park stylus out of frame
        ux, uy = arm.MOVE_DIRECTIONS['up']
        arm._fast_move(ux * self.PARK_DISTANCE, uy * self.PARK_DISTANCE)

        try:
            frame = self.top_cam.snapshot()
            if frame is None:
                raise RuntimeError("Top camera capture failed")
            self._ensure_phone_presence(frame)
            return frame
        finally:
            arm._fast_move(saved_x, saved_y)

    def snapshot_side(self):
        """Capture a frame from the side camera. Returns BGR numpy array.

        Checks that the phone is still visible in the frame.
        """
        frame = self.side_cam.snapshot()
        if frame is None:
            raise RuntimeError("Side camera capture failed")
        self._ensure_phone_presence(frame)
        return frame

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
            self._arm.close()
        if self._top_cam:
            self._top_cam.close()
        if self._side_cam:
            self._side_cam.close()
