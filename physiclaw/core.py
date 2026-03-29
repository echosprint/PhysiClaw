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
        self._pending_bboxes: dict[str, dict] = {}
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
        rax, ray, right_dist = right_result
        dax, day, down_dist = down_result
        arm.set_direction_mapping((rax, ray), (dax, day))
        phase5_swipe(arm, cam)

        log.info(f"Calibration complete — Z={z_tap}mm, "
                 f"right=({rax},{ray}), down=({dax},{day})")

        # Phase 6 — grid calibration for coordinate-based tapping
        # The pen-calib page shows the red dot grid after phases 1-5
        arm._fast_move(0, 0)
        arm.wait_idle()
        self._grid_cal = phase6_grid(arm, cam, (rax, ray), (dax, day),
                                     right_dist, down_dist)

    # ─── Properties ────────────────────────────────────────────

    @property
    def arm(self) -> StylusArm:
        return self._arm

    @property
    def cam(self) -> Camera:
        return self._cam

    # ─── Bbox state management ────────────────────────────────

    SMALL_BBOX_THRESHOLD = 15  # % — bbox smaller than this gets candidates

    # Colors for bbox candidates: BGR format. Assigned in order as candidates are added.
    BBOX_COLORS = [
        (0, 255, 0),     # green
        (0, 0, 255),     # red
        (255, 0, 0),     # blue
        (0, 255, 255),   # yellow
        (255, 0, 255),   # magenta
    ]

    def set_pending_bbox(self, left: float, right: float,
                         top: float, bottom: float):
        """Store pending bbox candidates keyed by shift name. Clears any confirmed bbox.

        For small targets, generates shifted candidates along the small
        dimension(s) so the AI agent can pick the best one.
        - Both dimensions small: center + top/bottom/left/right
        - Only width small: center + left/right
        - Only height small: center + top/bottom
        - Both dimensions large: center only
        """
        original = {'left': left, 'right': right, 'top': top, 'bottom': bottom}
        w = right - left
        h = bottom - top
        small_h = w < self.SMALL_BBOX_THRESHOLD
        small_v = h < self.SMALL_BBOX_THRESHOLD

        self._pending_bboxes = {'center': original}

        # Shifted bboxes: 80% size of original, shifted by 70% of that dimension
        scale = 0.8
        shift_ratio = 0.7
        sw = w * scale  # shifted bbox width
        sh = h * scale  # shifted bbox height
        cx = (left + right) / 2
        cy = (top + bottom) / 2

        if small_v:
            sv = h * shift_ratio
            self._pending_bboxes['top'] = {
                'left': max(0, cx - sw / 2), 'right': min(100, cx + sw / 2),
                'top': max(0, cy - sv - sh / 2), 'bottom': max(0, cy - sv + sh / 2),
            }
            self._pending_bboxes['bottom'] = {
                'left': max(0, cx - sw / 2), 'right': min(100, cx + sw / 2),
                'top': min(100, cy + sv - sh / 2), 'bottom': min(100, cy + sv + sh / 2),
            }
        if small_h:
            shh = w * shift_ratio
            self._pending_bboxes['left'] = {
                'left': max(0, cx - shh - sw / 2), 'right': max(0, cx - shh + sw / 2),
                'top': max(0, cy - sh / 2), 'bottom': min(100, cy + sh / 2),
            }
            self._pending_bboxes['right'] = {
                'left': min(100, cx + shh - sw / 2), 'right': min(100, cx + shh + sw / 2),
                'top': max(0, cy - sh / 2), 'bottom': min(100, cy + sh / 2),
            }

        self._confirmed_bbox = None

    def confirm_bbox(self, shift: str = 'center'):
        """Lock in a pending bbox by shift name for the next gesture.

        Valid shift values: "center", "top", "bottom", "left", "right"
        (only those present in the current candidates).
        """
        if not self._pending_bboxes:
            raise RuntimeError("No pending bbox — call bbox_target first")
        if shift not in self._pending_bboxes:
            valid = ', '.join(self._pending_bboxes.keys())
            raise RuntimeError(f"Invalid shift '{shift}' — available: {valid}")
        self._confirmed_bbox = self._pending_bboxes[shift]
        self._pending_bboxes = {}

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

    def screenshot_with_bboxes(self):
        """Take a fresh screenshot with all pending bbox candidates drawn.

        For large targets: one green rectangle labeled "center".
        For small targets: multiple colored rectangles with shift labels.
        Must call set_pending_bbox() first to populate candidates.
        """
        if self._grid_cal is None:
            raise RuntimeError("Grid calibration not done")
        if not self._pending_bboxes:
            raise RuntimeError("No pending bboxes — call set_pending_bbox first")

        # Build list of (tl, br, color, label) for all candidates
        rects = []
        for j, (name, bbox) in enumerate(self._pending_bboxes.items()):
            tl, br = self._grid_cal.bbox_to_pixel_rect(
                bbox['left'], bbox['right'], bbox['top'], bbox['bottom'])
            color = self.BBOX_COLORS[j % len(self.BBOX_COLORS)]
            rects.append((tl, br, color, name))

        frame = self.cam.snapshot()
        if frame is None:
            raise RuntimeError("Camera capture failed")

        for tl, br, color, name in rects:
            cv2.rectangle(frame, tl, br, color, 2)
            # Position label based on shift direction
            if name == 'bottom':
                label_pos = (tl[0] + 2, br[1] + 15)   # below bbox
            elif name == 'right':
                label_pos = (br[0] + 4, (tl[1] + br[1]) // 2)  # right of bbox
            elif name == 'left':
                label_pos = (tl[0] - 30, (tl[1] + br[1]) // 2)  # left of bbox
            else:
                label_pos = (tl[0] + 2, tl[1] - 5)    # above bbox (center, top)
            cv2.putText(frame, name, label_pos,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        # Save annotated frame
        from datetime import datetime
        from physiclaw.camera import SNAPSHOT_DIR
        SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
        cv2.imwrite(str(SNAPSHOT_DIR / f'{ts}_bbox.jpg'), frame)
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
