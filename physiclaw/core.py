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

    SMALL_BBOX_THRESHOLD = 0.15  # bbox smaller than this gets candidates (0-1 scale)

    # Colors for bbox candidates: BGR format. Assigned in order as candidates are added.
    BBOX_COLORS = [
        (0, 255, 0),     # green
        (0, 0, 255),     # red
        (255, 0, 0),     # blue
        (0, 255, 255),   # yellow
        (255, 0, 255),   # magenta
    ]

    def set_pending_bbox(self, left: float, top: float,
                         right: float, bottom: float):
        """Store pending bbox candidates keyed by shift name. Clears any confirmed bbox.

        All coordinates are 0-1 decimals (0=left/top, 1=right/bottom).

        For small targets, generates shifted candidates along the small
        dimension(s) so the AI agent can pick the best one.
        - Both dimensions small: center + top/bottom/left/right
        - Only width small: center + left/right
        - Only height small: center + top/bottom
        - Both dimensions large: center only
        """
        original = {'left': left, 'top': top, 'right': right, 'bottom': bottom}
        w = right - left
        h = bottom - top
        small_h = w < self.SMALL_BBOX_THRESHOLD
        small_v = h < self.SMALL_BBOX_THRESHOLD

        self._pending_bboxes = {'center': original}

        # Shifted bboxes: 0.8x size of original, shifted by 0.7x of that dimension
        scale = 0.8
        shift_ratio = 0.7
        sw = w * scale  # shifted bbox width
        sh = h * scale  # shifted bbox height
        cx = (left + right) / 2
        cy = (top + bottom) / 2

        if small_v:
            sv = h * shift_ratio
            self._pending_bboxes['top'] = {
                'left': max(0, cx - sw / 2), 'top': max(0, cy - sv - sh / 2),
                'right': min(1, cx + sw / 2), 'bottom': max(0, cy - sv + sh / 2),
            }
            self._pending_bboxes['bottom'] = {
                'left': max(0, cx - sw / 2), 'top': min(1, cy + sv - sh / 2),
                'right': min(1, cx + sw / 2), 'bottom': min(1, cy + sv + sh / 2),
            }
        if small_h:
            shh = w * shift_ratio
            self._pending_bboxes['left'] = {
                'left': max(0, cx - shh - sw / 2), 'top': max(0, cy - sh / 2),
                'right': max(0, cx - shh + sw / 2), 'bottom': min(1, cy + sh / 2),
            }
            self._pending_bboxes['right'] = {
                'left': min(1, cx + shh - sw / 2), 'top': max(0, cy - sh / 2),
                'right': min(1, cx + shh + sw / 2), 'bottom': min(1, cy + sh / 2),
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
            bbox['left'], bbox['top'], bbox['right'], bbox['bottom'])
        gx, gy = self._grid_cal.pct_to_grbl_mm(cx, cy)
        self._arm._fast_move(gx, gy)
        self._arm.wait_idle()

    def tap_at_pct(self, x: float, y: float):
        """Move arm to screen coordinate (0-1) and tap. Bypasses bbox workflow."""
        if self._grid_cal is None:
            raise RuntimeError("Grid calibration not done")
        gx, gy = self._grid_cal.pct_to_grbl_mm(x, y)
        self._arm._fast_move(gx, gy)
        self._arm.wait_idle()
        self._arm.tap()

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
                bbox['left'], bbox['top'], bbox['right'], bbox['bottom'])
            color = self.BBOX_COLORS[j % len(self.BBOX_COLORS)]
            rects.append((tl, br, color, name))

        frame = self.cam.snapshot()
        if frame is None:
            raise RuntimeError("Camera capture failed")

        font = cv2.FONT_HERSHEY_SIMPLEX
        pad = 4
        for tl, br, color, name in rects:
            cv2.rectangle(frame, tl, br, color, 2)
            # Position label based on shift direction
            if name == 'bottom':
                lx, ly = tl[0] + 2, br[1] + 15
            elif name == 'right':
                lx, ly = br[0] + 4, (tl[1] + br[1]) // 2
            elif name == 'left':
                lx, ly = tl[0] - 30, (tl[1] + br[1]) // 2
            else:
                lx, ly = tl[0] + 2, tl[1] - 5
            (tw, th), _ = cv2.getTextSize(name, font, 0.8, 2)
            cv2.rectangle(frame, (lx - pad, ly - th - pad),
                          (lx + tw + pad, ly + pad), color, -1)
            cv2.putText(frame, name, (lx, ly), font, 0.8, (255, 255, 255), 2)

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

    # ─── Grid overlay ─────────────────────────────────────────

    def screenshot_with_grid(self, color: str = "green",
                             rows: int = 9, cols: int = 4):
        """Take a screenshot with percentage reference grid lines drawn.

        Draws vertical lines at evenly spaced x-percentages and horizontal
        lines at evenly spaced y-percentages. Labels are drawn outside the
        phone screen area.

        Args:
            color: line color — "green", "red", or "yellow".
            rows: number of horizontal lines (e.g. 9 → lines at 0.10, 0.20, ..., 0.90).
            cols: number of vertical lines (e.g. 4 → lines at 0.20, 0.40, 0.60, 0.80).
        """
        if self._grid_cal is None:
            raise RuntimeError("Grid calibration not done")

        frame = self.cam.snapshot()
        if frame is None:
            raise RuntimeError("Camera capture failed")

        color_map = {
            "green": (0, 255, 0),
            "red": (0, 0, 255),
            "yellow": (0, 255, 255),
        }
        bgr = color_map.get(color, (0, 255, 0))
        cal = self._grid_cal
        font = cv2.FONT_HERSHEY_SIMPLEX

        pad = 4  # padding around label text

        def _draw_label(frame, label, cx, cy):
            """Draw a white-on-color label centered at (cx, cy)."""
            (tw, th), _ = cv2.getTextSize(label, font, 0.8, 2)
            lx, ly = cx - tw // 2, cy + th // 2
            cv2.rectangle(frame, (lx - pad, ly - th - pad),
                          (lx + tw + pad, ly + pad), bgr, -1)
            cv2.putText(frame, label, (lx, ly), font, 0.8, (255, 255, 255), 2)

        # Vertical lines — labels at top and bottom
        for i in range(1, cols + 1):
            x_val = round(i / (cols + 1), 2)
            pt_top = cal.pct_to_cam_pixel(x_val, 0)
            pt_bot = cal.pct_to_cam_pixel(x_val, 1)
            cv2.line(frame, pt_top, pt_bot, bgr, 1)
            label = f"{x_val:.2f}"
            _draw_label(frame, label, pt_top[0], pt_top[1] - 15)
            _draw_label(frame, label, pt_bot[0], pt_bot[1] + 15)

        # Horizontal lines — labels at left and right
        for i in range(1, rows + 1):
            y_val = round(i / (rows + 1), 2)
            pt_left = cal.pct_to_cam_pixel(0, y_val)
            pt_right = cal.pct_to_cam_pixel(1, y_val)
            cv2.line(frame, pt_left, pt_right, bgr, 1)
            label = f"{y_val:.2f}"
            (tw, _), _ = cv2.getTextSize(label, font, 0.8, 2)
            _draw_label(frame, label, pt_left[0] - tw // 2 - 10, pt_left[1])
            _draw_label(frame, label, pt_right[0] + tw // 2 + 10, pt_right[1])

        # Save
        from datetime import datetime
        from physiclaw.camera import SNAPSHOT_DIR
        SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
        cv2.imwrite(str(SNAPSHOT_DIR / f'{ts}_overlay.jpg'), frame)
        return frame

    # ─── Annotation support ──────────────────────────────────────

    # Hex color → BGR tuple for OpenCV drawing
    _COLOR_NAMES = {
        '#ff5252': 'red', '#448aff': 'blue',
        '#69f0ae': 'green', '#ffd740': 'yellow',
        '#e040fb': 'purple', '#00e5ff': 'cyan',
        '#e0e0e0': 'white', '#b2ff59': 'lime',
    }

    @staticmethod
    def _hex_to_bgr(hex_color: str) -> tuple[int, int, int]:
        h = hex_color.lstrip('#')
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        return (b, g, r)

    def process_annotations(self, frame, annotations: list[dict]):
        """Convert pixel-coordinate annotations to 0-1 screen coords.

        Draws colored numbered boxes on the frame and returns a text listing
        with coordinates as 0-1 decimals [left, top, right, bottom] and color.

        Args:
            frame: BGR numpy array (the frozen snapshot)
            annotations: list of {left, top, right, bottom, color?} in image pixels

        Returns:
            (text_listing, annotated_frame) or None if annotations is empty.
        """
        if not annotations:
            return None
        if self._grid_cal is None:
            raise RuntimeError("Grid calibration not done")

        cal = self._grid_cal
        out = frame.copy()
        elements = []
        for i, ann in enumerate(annotations):
            l, t = cal.pixel_to_pct(int(ann['left']), int(ann['top']))
            r, b = cal.pixel_to_pct(int(ann['right']), int(ann['bottom']))
            l = max(0.0, min(1.0, round(l, 3)))
            t = max(0.0, min(1.0, round(t, 3)))
            r = max(0.0, min(1.0, round(r, 3)))
            b = max(0.0, min(1.0, round(b, 3)))
            color = ann.get('color', '#42a5f5')
            elements.append({'id': i + 1, 'left': l, 'top': t,
                             'right': r, 'bottom': b, 'color': color})
            bgr = self._hex_to_bgr(color)
            cv2.rectangle(out,
                          (int(ann['left']), int(ann['top'])),
                          (int(ann['right']), int(ann['bottom'])),
                          bgr, 2)
            cv2.putText(out, str(i + 1),
                        (int(ann['left']) + 4, int(ann['top']) + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, bgr, 2)

        lines = [f"# Pending Annotations ({len(elements)} boxes)\n"]
        for e in elements:
            name = self._COLOR_NAMES.get(e['color'], e['color'])
            lines.append(f"- Box {e['id']} ({name}): [{e['left']}, {e['top']}, "
                         f"{e['right']}, {e['bottom']}]")
        return "\n".join(lines), out

    # ─── Lifecycle ─────────────────────────────────────────────

    def shutdown(self):
        if self._arm:
            self._arm._pen_up()
            self._arm._fast_move(0, 0)
            self._arm.wait_idle()
            self._arm.close()
        if self._cam:
            self._cam.close()
