"""
PhysiClaw orchestrator — central hardware lifecycle manager.

Owns the stylus arm, camera, and calibration state.
Creating an instance connects hardware, finds the camera, and runs calibration.
"""

import logging
import threading

import cv2

from physiclaw.camera import Camera
from physiclaw.grid_calibrate import GridCalibration
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
        self._grid_cal: GridCalibration | None = None
        self._pending_bbox: dict | None = None
        self._confirmed_bbox: dict | None = None
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
        """Run the full 6-phase calibration workflow.

        Phases 1-5: Z depth, direction mapping, gesture verification.
        Phase 6: Grid calibration for coordinate-based tapping.
        """
        from physiclaw.calibrate import (
            phase1_z, phase2_right, phase3_down,
            phase4_long_press, phase5_swipe,
        )
        from physiclaw.grid_calibrate import phase6_grid

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

        # Phase 6 — grid calibration for coordinate-based tapping
        # The pen-calib page shows the red dot grid after phases 1-5
        arm._fast_move(0, 0)
        arm.wait_idle()
        self._grid_cal = phase6_grid(arm, cam, (rax, ray), (dax, day))

    # ─── Properties ────────────────────────────────────────────

    @property
    def arm(self) -> StylusArm:
        return self._arm

    @property
    def cam(self) -> Camera:
        return self._cam

    # ─── Bbox state management ────────────────────────────────

    def set_pending_bbox(self, left: float, right: float,
                         top: float, bottom: float):
        """Store a pending bbox from bbox_target. Clears any confirmed bbox."""
        self._pending_bbox = {
            'left': left, 'right': right, 'top': top, 'bottom': bottom,
        }
        self._confirmed_bbox = None

    def confirm_bbox(self):
        """Lock in the pending bbox for the next gesture."""
        if self._pending_bbox is None:
            raise RuntimeError("No pending bbox — call bbox_target first")
        self._confirmed_bbox = self._pending_bbox
        self._pending_bbox = None

    def consume_confirmed_bbox(self) -> dict | None:
        """Return and clear the confirmed bbox. Returns None if none."""
        bbox = self._confirmed_bbox
        self._confirmed_bbox = None
        return bbox

    def move_to_bbox_center(self, bbox: dict):
        """Move arm to the center of a bbox using grid calibration."""
        if self._grid_cal is None:
            raise RuntimeError("Grid calibration not done")
        cx, cy = self._grid_cal.bbox_center_pct(
            bbox['left'], bbox['right'], bbox['top'], bbox['bottom'])
        gx, gy = self._grid_cal.pct_to_grbl_mm(cx, cy)
        self._arm._fast_move(gx, gy)
        self._arm.wait_idle()

    # ─── Snapshot helpers ──────────────────────────────────────

    PARK_DISTANCE = 100  # mm to move stylus out of frame

    def screenshot(self):
        """Capture a frame from the camera. Returns BGR numpy array.

        Takes the frame as-is — the stylus may be visible.
        Call park() first if an unobstructed view is needed.
        Frame is already rotated to portrait by the camera.
        """
        frame = self.cam.snapshot()
        if frame is None:
            raise RuntimeError("Camera capture failed")
        return frame

    def screenshot_with_bbox(self, left: float, right: float,
                             top: float, bottom: float):
        """Take a fresh screenshot and draw a green bbox rectangle on it."""
        frame = self.screenshot()
        if self._grid_cal is None:
            raise RuntimeError("Grid calibration not done")
        tl, br = self._grid_cal.bbox_to_pixel_rect(left, right, top, bottom)
        cv2.rectangle(frame, tl, br, (0, 255, 0), 2)
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
