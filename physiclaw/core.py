"""
PhysiClaw orchestrator — central hardware lifecycle manager.

Owns the stylus arm, camera, and calibration state.
Construction is instant — call connect_arm(), connect_camera(), and
calibrate_z_depth() through calibrate_grid() to set up hardware.
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

    Construction is instant (no hardware). Call connect_arm(),
    connect_camera(), then calibrate_z_depth() through calibrate_grid()
    to set up hardware incrementally.
    """

    def __init__(self):
        self._arm: StylusArm | None = None
        self._cam: Camera | None = None
        self._detector: PhoneDetector | None = None
        self._grid_cal: GridCalibration | None = None
        self._pending_bboxes: dict[str, list[float]] = {}
        self._confirmed_bbox: list[float] | None = None
        self._lock = threading.Lock()
        self._cal: dict = {}  # intermediate calibration state between phases

    @property
    def hardware_ready(self) -> bool:
        """True when arm, camera, and grid calibration are all set."""
        return (self._arm is not None
                and self._cam is not None
                and self._grid_cal is not None)

    def status(self) -> dict:
        """Return current hardware and calibration state."""
        return {
            "arm": self._arm is not None,
            "camera": self._cam is not None,
            "completed_steps": [name for name, check in [
                ("z-depth", 'z_tap' in self._cal),
                ("find-right", 'right_result' in self._cal),
                ("find-down", 'down_result' in self._cal),
                ("long-press", self._cal.get('phase4_done', False)),
                ("swipe", bool(self._arm and self._arm.MOVE_DIRECTIONS)),
                ("grid", self._grid_cal is not None),
            ] if check],
            "calibrated": self.hardware_ready,
        }

    def require_hardware(self):
        """Raise if hardware is not fully set up."""
        if not self.hardware_ready:
            raise RuntimeError(
                "Hardware not set up. Run /setup to connect and calibrate.")

    def acquire(self):
        """Mark hardware as busy. Raises immediately if already busy."""
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("PhysiClaw is busy — wait for the current operation to finish, then retry.")

    def release(self):
        """Mark hardware as idle."""
        self._lock.release()

    # ─── Camera preview ────────────────────────────────────────

    @staticmethod
    def camera_preview(index: int) -> bytes:
        """Capture one frame from a camera, watermark the index, return JPEG bytes.

        Opens the camera, grabs a frame, draws a semi-transparent index
        watermark in the center, closes the camera, and returns JPEG bytes.
        Raises RuntimeError if camera can't be opened or returns no frame.
        """
        cam = Camera(index)
        frame = cam.snapshot()
        cam.close()
        if frame is None:
            raise RuntimeError(f"Camera {index} returned no frame")

        # Watermark the camera index
        h, w = frame.shape[:2]
        label = str(index)
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = h / 150
        thickness = max(2, int(scale * 2))
        (tw, th), _ = cv2.getTextSize(label, font, scale, thickness)
        cx, cy = w // 2, h // 2
        overlay = frame.copy()
        pad = int(scale * 20)
        cv2.rectangle(overlay,
                      (cx - tw // 2 - pad, cy - th // 2 - pad),
                      (cx + tw // 2 + pad, cy + th // 2 + pad),
                      (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
        cv2.putText(frame, label,
                    (cx - tw // 2, cy + th // 2),
                    font, scale, (255, 255, 255), thickness)

        _, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        return jpeg.tobytes()

    # ─── Hardware connection ──────────────────────────────────

    def connect_arm(self):
        """Connect to the GRBL stylus arm (auto-detect USB port).

        Closes any previously connected arm first.
        """
        if self._arm is not None:
            self._arm.close()
            self._arm = None
        self._cal = {}  # reset calibration state
        self._grid_cal = None
        self._arm = StylusArm()
        self._arm.setup()
        log.info("Arm connected")

    def connect_camera(self, index: int | None = None):
        """Open a camera by index, or auto-detect the one seeing the phone.

        Closes any previously connected camera first.

        Args:
            index: specific camera index to use. If None, scans all cameras
                   and picks the one with highest phone detection confidence.
        """
        if self._cam is not None:
            self._cam.close()
            self._cam = None
        if index is not None:
            self._cam = Camera(index)
            log.info(f"Camera {index} connected")
        else:
            self._detector = PhoneDetector()
            self._cam = self._detector.find_camera()
            if self._cam is None:
                raise RuntimeError("Camera not found — is the phone under the camera?")
            log.info(f"Camera {self._cam.index} connected (auto-detected)")

    # ─── Calibration (per-phase) ──────────────────────────────

    def calibrate_z_depth(self) -> dict:
        """Phase 1: Probe Z depth for tap contact (10 greens on center).

        Returns {"z_tap": float}.
        """
        from physiclaw.calibrate import phase1_z
        if self._arm is None or self._cam is None:
            raise RuntimeError("Arm and camera must be connected first")
        z_tap = phase1_z(self._arm, self._cam)
        self._arm.Z_DOWN = z_tap
        self._cal['z_tap'] = z_tap
        return {"z_tap": z_tap}

    def calibrate_find_right(self) -> dict:
        """Phase 2: Probe 4 directions to find phone-right (3 greens).

        Returns {"right_vec": [ax, ay], "right_dist": float}.
        Requires: phase 1 done.
        """
        from physiclaw.calibrate import phase2_right
        z_tap = self._cal.get('z_tap')
        if z_tap is None:
            raise RuntimeError("Phase 1 not done — run calibrate_z_depth() first")
        result = phase2_right(self._arm, self._cam, z_tap)
        self._cal['right_result'] = result
        return {"right_vec": [result[0], result[1]], "right_dist": result[2]}

    def calibrate_find_down(self) -> dict:
        """Phase 3: Probe 2 perpendicular directions for phone-down (3 greens).

        Returns {"down_vec": [ax, ay], "down_dist": float}.
        Requires: phase 2 done.
        """
        from physiclaw.calibrate import phase3_down
        right_result = self._cal.get('right_result')
        if right_result is None:
            raise RuntimeError("Phase 2 not done — run calibrate_find_right() first")
        z_tap = self._cal['z_tap']
        right_vec = (right_result[0], right_result[1])
        result = phase3_down(self._arm, self._cam, z_tap, right_vec)
        self._cal['down_result'] = result
        return {"down_vec": [result[0], result[1]], "down_dist": result[2]}

    def calibrate_long_press(self) -> dict:
        """Phase 4: Verify long press (3 greens, hold 800ms).

        Requires: phase 3 done.
        """
        from physiclaw.calibrate import phase4_long_press
        if 'down_result' not in self._cal:
            raise RuntimeError("Phase 3 not done — run calibrate_find_down() first")
        phase4_long_press(self._arm, self._cam)
        self._cal['phase4_done'] = True
        return {"ok": True}

    def calibrate_swipe(self) -> dict:
        """Phase 5: Verify swipe in 4 directions.

        Sets direction mapping on the arm.
        Requires: phase 4 done.
        """
        from physiclaw.calibrate import phase5_swipe
        if not self._cal.get('phase4_done'):
            raise RuntimeError("Phase 4 not done — run calibrate_long_press() first")
        right_result = self._cal['right_result']
        down_result = self._cal['down_result']
        rax, ray, _ = right_result
        dax, day, _ = down_result
        self._arm.set_direction_mapping((rax, ray), (dax, day))
        phase5_swipe(self._arm, self._cam)
        log.info(f"Calibration phases 1-5 complete — Z={self._cal['z_tap']}mm, "
                 f"right=({rax},{ray}), down=({dax},{day})")
        return {"ok": True}

    def calibrate_grid(self) -> dict:
        """Phase 6: Grid calibration using 15 red dots.

        Computes affine transforms for coordinate-based tapping.
        Requires: phase 5 done.
        """
        from physiclaw.grid_calibrate import phase6_grid
        if not (self._arm and self._arm.MOVE_DIRECTIONS):
            raise RuntimeError("Phase 5 not done — run earlier phases first")
        right_result = self._cal['right_result']
        down_result = self._cal['down_result']
        rax, ray, right_dist = right_result
        dax, day, down_dist = down_result
        self._arm._fast_move(0, 0)
        self._arm.wait_idle()
        self._grid_cal = phase6_grid(
            self._arm, self._cam,
            (rax, ray), (dax, day),
            right_dist, down_dist)
        return {"ok": True}

    def verify_edge_trace(self) -> dict:
        """Trace the phone screen border clockwise for visual verification.

        The arm moves to 8 edge points, pausing 2s at each, then returns
        to center. The user should watch and confirm the arm follows the
        screen edges.
        Requires: phase 6 done.
        """
        from physiclaw.grid_calibrate import trace_screen_edge
        if self._grid_cal is None:
            raise RuntimeError("Phase 6 not done — run calibrate_grid() first")
        trace_screen_edge(self._arm, self._grid_cal)
        return {"ok": True}

    def calibrate(self):
        """Run the full 6-phase calibration workflow (convenience method)."""
        self.calibrate_z_depth()
        self.calibrate_find_right()
        self.calibrate_find_down()
        self.calibrate_long_press()
        self.calibrate_swipe()
        self.calibrate_grid()

    # ─── Properties ────────────────────────────────────────────

    @property
    def arm(self) -> StylusArm:
        return self._arm

    @property
    def cam(self) -> Camera:
        return self._cam

    # ─── Bbox state management ────────────────────────────────

    BBOX_COLOR = (0, 255, 0)  # green, BGR

    def set_pending_bbox(self, bbox: list[float]):
        """Store a pending bbox. Clears any confirmed bbox.

        Args:
            bbox: [left, top, right, bottom] as 0-1 decimals.
        """
        self._pending_bboxes = {'center': list(bbox)}
        self._confirmed_bbox = None

    def confirm_bbox(self):
        """Lock in the pending bbox for the next gesture."""
        if not self._pending_bboxes:
            raise RuntimeError("No pending bbox — call bbox_target first")
        self._confirmed_bbox = self._pending_bboxes['center']
        self._pending_bboxes = {}

    def move_to_bbox_center(self, bbox: list[float]):
        """Move arm to the center of a bbox [left, top, right, bottom] (0-1)."""
        if self._grid_cal is None:
            raise RuntimeError("Grid calibration not done")
        cx, cy = self._grid_cal.bbox_center_pct(bbox)
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
        """Take a fresh screenshot with the pending bbox drawn as a green rectangle.

        Must call set_pending_bbox() first.
        """
        if self._grid_cal is None:
            raise RuntimeError("Grid calibration not done")
        if not self._pending_bboxes:
            raise RuntimeError("No pending bboxes — call set_pending_bbox first")

        bbox = self._pending_bboxes['center']
        tl, br = self._grid_cal.bbox_to_pixel_rect(bbox)

        frame = self.cam.snapshot()
        if frame is None:
            raise RuntimeError("Camera capture failed")

        cv2.rectangle(frame, tl, br, self.BBOX_COLOR, 2)

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

    # ─── Element detection ─────────────────────────────────────

    def detect_elements(self):
        """Detect UI elements using three analysis tools.

        Three complementary detectors run on every frame:
        1. Color segmentation — finds colored buttons, icons, content images
        2. Icon detection — finds all UI elements including gray/colorless ones
        3. OCR — reads all visible text

        Returns (elements_text, color_frame, icon_frame, ocr_frame) where:
        - elements_text: formatted text listing all elements with 0-1 coords
        - color_frame: screenshot annotated with color segmentation boxes
        - icon_frame: screenshot annotated with icon detection boxes
        - ocr_frame: screenshot annotated with OCR text boxes
        """
        if self._grid_cal is None:
            raise RuntimeError("Grid calibration not done")

        frame = self.cam.snapshot()
        if frame is None:
            raise RuntimeError("Camera capture failed")

        from datetime import datetime

        cal = self._grid_cal
        h, w = frame.shape[:2]
        element_id = 0

        # ── Tool 1: Color segmentation ─────────────────────────
        color_table_header = (
            "| id | color | type | bbox [left, top, right, bottom] | h_std |\n"
            "|----|-------|------|------|-------|"
        )
        color_rows = []
        color_frame = frame.copy()
        try:
            from physiclaw.color_segment import detect_color_blocks
            from physiclaw.color_segment import annotate as color_annotate
            blobs = detect_color_blocks(frame)
            color_frame = color_annotate(frame, blobs)
            for blob in blobs:
                element_id += 1
                x1, y1, x2, y2 = blob.bbox
                l, t = cal.pixel_to_pct(int(x1), int(y1))
                r, b = cal.pixel_to_pct(int(x2), int(y2))
                kind = "image" if blob.is_image else "solid" if blob.is_solid else "mixed"
                color_rows.append(
                    f"| {element_id} | {blob.color_name} | {kind} "
                    f"| [{l:.2f}, {t:.2f}, {r:.2f}, {b:.2f}] "
                    f"| {blob.h_std:.1f} |"
                )
        except Exception as ex:
            color_rows.append(f"| — | — | error: {ex} | — | — |")

        # ── Tool 2: Icon detection ─────────────────────────────
        icon_table_header = (
            "| id | bbox [left, top, right, bottom] | conf |\n"
            "|----|------|------|"
        )
        icon_rows = []
        icon_frame = frame.copy()
        try:
            from physiclaw.icon_detect import IconDetector, annotate as icon_annotate
            if not hasattr(self, '_icon_detector'):
                self._icon_detector = IconDetector()
            icons = self._icon_detector.detect(frame, confidence=0.2)
            icon_frame = icon_annotate(frame, icons)
            for e in icons:
                element_id += 1
                x1, y1, x2, y2 = e.bbox
                l, t = cal.pixel_to_pct(x1, y1)
                r, b = cal.pixel_to_pct(x2, y2)
                icon_rows.append(
                    f"| {element_id} "
                    f"| [{l:.2f}, {t:.2f}, {r:.2f}, {b:.2f}] "
                    f"| {e.confidence:.2f} |"
                )
        except (ImportError, FileNotFoundError) as ex:
            icon_rows.append(f"| — | unavailable: {ex} | — |")

        # ── Tool 3: OCR ────────────────────────────────────────
        text_table_header = (
            "| id | label | bbox [left, top, right, bottom] | conf |\n"
            "|----|-------|------|------|"
        )
        ocr_rows = []
        ocr_frame = frame.copy()
        try:
            from physiclaw.ocr import OCRReader, annotate as ocr_annotate
            if not hasattr(self, '_ocr_reader'):
                self._ocr_reader = OCRReader()
            texts = self._ocr_reader.read(frame)
            ocr_frame = ocr_annotate(frame, texts)
            for t in texts:
                element_id += 1
                x1, y1, x2, y2 = t.bbox
                l, tp = cal.pixel_to_pct(x1, y1)
                r, b = cal.pixel_to_pct(x2, y2)
                ocr_rows.append(
                    f"| {element_id} | \"{t.text}\" "
                    f"| [{l:.2f}, {tp:.2f}, {r:.2f}, {b:.2f}] "
                    f"| {t.confidence:.2f} |"
                )
        except ImportError as ex:
            ocr_rows.append(f"| — | unavailable: {ex} | — | — |")

        ts = datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ')
        elements_text = (
            f"# Screen Parse Result\n\n"
            f"- **resolution**: {w}x{h}\n"
            f"- **timestamp**: {ts}\n\n"
            f"## Color Blocks\n\n{color_table_header}\n"
            + "\n".join(color_rows)
            + f"\n\n## Icons\n\n{icon_table_header}\n"
            + "\n".join(icon_rows)
            + f"\n\n## Text\n\n{text_table_header}\n"
            + "\n".join(ocr_rows)
        )

        # Save
        from physiclaw.camera import SNAPSHOT_DIR
        SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
        file_ts = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
        cv2.imwrite(str(SNAPSHOT_DIR / f'{file_ts}_colors.jpg'), color_frame)
        cv2.imwrite(str(SNAPSHOT_DIR / f'{file_ts}_icons.jpg'), icon_frame)
        cv2.imwrite(str(SNAPSHOT_DIR / f'{file_ts}_ocr.jpg'), ocr_frame)

        return elements_text, color_frame, icon_frame, ocr_frame

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
            annotations: list of {left, top, right, bottom, color?, label?, source?}
                         in image pixels

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
            bbox = [
                max(0.0, min(1.0, round(l, 3))),
                max(0.0, min(1.0, round(t, 3))),
                max(0.0, min(1.0, round(r, 3))),
                max(0.0, min(1.0, round(b, 3))),
            ]
            from physiclaw.annotation import classify_bbox
            box_type, coords = classify_bbox(bbox)
            color = ann.get('color', '#42a5f5')
            label = ann.get('label', '')
            source = ann.get('source', 'user')
            elements.append({'id': i + 1, 'type': box_type, 'bbox': coords,
                             'color': color, 'label': label, 'source': source})
            bgr = self._hex_to_bgr(color)
            cv2.rectangle(out,
                          (int(ann['left']), int(ann['top'])),
                          (int(ann['right']), int(ann['bottom'])),
                          bgr, 2)
            cv2.putText(out, str(i + 1),
                        (int(ann['left']) + 4, int(ann['top']) + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, bgr, 2)

        lines = [f"# Pending Annotations ({len(elements)} items)\n"]
        for e in elements:
            name = self._COLOR_NAMES.get(e['color'], e['color'])
            b = e['bbox']
            desc = f" — {e['label']}" if e['label'] else ""
            src = f" [{e['source']}]" if e['source'] != 'user' else ""
            coords = ", ".join(str(v) for v in b)
            type_tag = f" ({e['type']})" if e['type'] != 'box' else ""
            lines.append(f"- {e['id']}{type_tag} ({name}){src}: [{coords}]{desc}")
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
