"""
PhysiClaw orchestrator — central hardware lifecycle manager.

Owns the stylus arm, camera, and calibration state.
Construction is instant — call connect_arm() and connect_camera()
to set up hardware. Calibration is done via /setup skill endpoints.
"""

import logging
import threading
from datetime import datetime

import cv2

from physiclaw.calibration import GridCalibration
from physiclaw.hardware.camera import Camera, SNAPSHOT_DIR
from physiclaw.hardware.screenshot import PhoneScreenshot
from physiclaw.hardware.stylus_arm import StylusArm
from physiclaw.vision.phone_detect import PhoneDetector

from physiclaw.core.rendering import (
    draw_bbox,
    draw_grid_overlay,
    process_annotations as _process_annotations,
    encode_jpeg,
)

log = logging.getLogger(__name__)


class PhysiClaw:
    """Central orchestrator — owns all hardware lifecycle.

    Construction is instant (no hardware). Call connect_arm() and
    connect_camera() to connect hardware. Calibration is handled
    by the /setup skill via HTTP endpoints.
    """

    PARK_DISTANCE = 100  # mm to move stylus out of frame

    def __init__(self):
        self._arm: StylusArm | None = None
        self._cam: Camera | None = None
        self._detector: PhoneDetector | None = None
        self._grid_cal: GridCalibration | None = None
        self._pending_bboxes: dict[str, list[float]] = {}
        self._confirmed_bbox: list[float] | None = None
        self._lock = threading.Lock()
        self._cal: dict = {}  # intermediate calibration state between phases
        self._screenshot = PhoneScreenshot()
        self._icon_detector = None  # cached, lazy
        self._ocr_reader = None     # cached, lazy

    @property
    def hardware_ready(self) -> bool:
        """True when arm, camera, and grid calibration are all set."""
        return (self._arm is not None
                and self._cam is not None
                and self._grid_cal is not None)

    def status(self) -> dict:
        """Return current hardware and calibration state."""
        steps = {}
        z_tap = self._cal.get('z_tap')
        if z_tap is not None:
            steps["z_tap"] = f"{z_tap}mm"
        if 'screenshot_transform' in self._cal:
            t = self._cal['screenshot_transform']
            steps["screenshot_cal"] = f"dpr={t['dpr']}, offset=({t['offset_x']}, {t['offset_y']})"
        if self._arm and self._arm.MOVE_DIRECTIONS:
            steps["alignment"] = "OK"
        if 'rotation' in self._cal:
            names = {-1: "none", 0: "90° CW", 1: "180°", 2: "90° CCW"}
            steps["rotation"] = names.get(self._cal['rotation'], str(self._cal['rotation']))
        if 'screen_to_grbl' in self._cal:
            steps["mapping_a"] = "OK"
        if 'pct_to_cam' in self._cal:
            steps["mapping_b"] = "OK"
        if self._grid_cal is not None:
            steps["validated"] = True
        if self._screenshot.ready:
            sx, sy = self._screenshot.at_screen
            steps["assistive_touch"] = f"({sx:.3f}, {sy:.3f})"
        return {
            "arm": self._arm is not None,
            "camera": self._cam is not None,
            "steps": steps,
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
    def camera_preview(index: int, watermark: bool = False) -> bytes:
        """Capture one frame from a camera, optionally watermark the index.

        Opens the camera, grabs a frame, closes the camera, returns JPEG bytes.
        watermark: if True, draws camera index overlay in the center.
        Raises RuntimeError if camera can't be opened or returns no frame.
        """
        cam = Camera(index)
        frame = cam.snapshot()
        cam.close()
        if frame is None:
            raise RuntimeError(f"Camera {index} returned no frame")

        if not watermark:
            return encode_jpeg(frame, quality=80)

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

        return encode_jpeg(frame, quality=80)

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

    # ─── Verification ─────────────────────────────────────────

    def verify_edge_trace(self) -> dict:
        """Trace the phone screen border clockwise for visual verification.

        The arm moves to 8 edge points, pausing 2s at each, then returns
        to center. The user should watch and confirm the arm follows the
        screen edges.
        Requires: calibration complete (GridCalibration set).
        """
        from physiclaw.calibration.grid_calibrate import trace_screen_edge
        if self._grid_cal is None:
            raise RuntimeError("Not calibrated — run /setup first")
        trace_screen_edge(self._arm, self._grid_cal)
        return {"ok": True}

    # ─── Properties ────────────────────────────────────────────

    @property
    def arm(self) -> StylusArm:
        return self._arm

    @property
    def cam(self) -> Camera:
        return self._cam

    # ─── Bbox state management ────────────────────────────────

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

    def camera_view(self):
        """Capture a frame from the overhead camera. Returns BGR numpy array.

        Takes the frame as-is — the stylus may be visible.
        Call park() first if an unobstructed view is needed.
        Frame is already rotated to portrait by the camera.
        """
        frame = self.cam.snapshot()
        if frame is None:
            raise RuntimeError("Camera capture failed")
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
    def frame_to_jpeg(frame, quality: int = 85) -> bytes:
        """Encode a BGR frame to JPEG bytes."""
        return encode_jpeg(frame, quality)

    # ─── Rendering wrappers ───────────────────────────────────
    #
    # These methods fetch a fresh frame, delegate to pure rendering
    # functions in core.rendering / vision.detect, and save the result.

    def _save_snapshot(self, frame, suffix: str):
        SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
        cv2.imwrite(str(SNAPSHOT_DIR / f'{ts}_{suffix}.jpg'), frame)

    def screenshot_with_bboxes(self):
        """Take a fresh screenshot with the pending bbox drawn as a green rectangle.

        Must call set_pending_bbox() first.
        """
        if self._grid_cal is None:
            raise RuntimeError("Grid calibration not done")
        if not self._pending_bboxes:
            raise RuntimeError("No pending bboxes — call set_pending_bbox first")

        bbox = self._pending_bboxes['center']
        frame = self.cam.snapshot()
        if frame is None:
            raise RuntimeError("Camera capture failed")

        out = draw_bbox(frame, bbox, self._grid_cal)
        self._save_snapshot(out, 'bbox')
        return out

    def screenshot_with_grid(self, color: str = "green",
                             rows: int = 9, cols: int = 4):
        """Take a screenshot with percentage reference grid lines drawn.

        Args:
            color: line color — "green", "red", or "yellow".
            rows: number of horizontal lines.
            cols: number of vertical lines.
        """
        if self._grid_cal is None:
            raise RuntimeError("Grid calibration not done")

        frame = self.cam.snapshot()
        if frame is None:
            raise RuntimeError("Camera capture failed")

        out = draw_grid_overlay(frame, self._grid_cal, color, rows, cols)
        self._save_snapshot(out, 'overlay')
        return out

    def detect_elements(self):
        """Detect UI elements using three analysis tools.

        Three complementary detectors run on every frame:
        1. Color segmentation — finds colored buttons, icons, content images
        2. Icon detection — finds all UI elements including gray/colorless ones
        3. OCR — reads all visible text

        Returns (elements_text, color_frame, icon_frame, ocr_frame).
        """
        if self._grid_cal is None:
            raise RuntimeError("Grid calibration not done")

        frame = self.cam.snapshot()
        if frame is None:
            raise RuntimeError("Camera capture failed")

        # Lazy-create cached detectors on first use
        try:
            from physiclaw.vision.icon_detect import IconDetector
            if self._icon_detector is None:
                self._icon_detector = IconDetector()
        except (ImportError, FileNotFoundError):
            pass  # detect_all_elements handles missing detector gracefully

        try:
            from physiclaw.vision.ocr import OCRReader
            if self._ocr_reader is None:
                self._ocr_reader = OCRReader()
        except ImportError:
            pass

        from physiclaw.vision.detect import detect_all_elements
        elements_text, color_frame, icon_frame, ocr_frame = detect_all_elements(
            frame, self._grid_cal,
            icon_detector=self._icon_detector,
            ocr_reader=self._ocr_reader,
        )

        SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
        file_ts = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
        cv2.imwrite(str(SNAPSHOT_DIR / f'{file_ts}_colors.jpg'), color_frame)
        cv2.imwrite(str(SNAPSHOT_DIR / f'{file_ts}_icons.jpg'), icon_frame)
        cv2.imwrite(str(SNAPSHOT_DIR / f'{file_ts}_ocr.jpg'), ocr_frame)

        return elements_text, color_frame, icon_frame, ocr_frame

    def process_annotations(self, frame, annotations: list[dict]):
        """Convert pixel-coordinate annotations to 0-1 screen coords.

        Thin wrapper around core.rendering.process_annotations.
        """
        if self._grid_cal is None:
            raise RuntimeError("Grid calibration not done")
        return _process_annotations(frame, annotations, self._grid_cal)

    # ─── Lifecycle ─────────────────────────────────────────────

    def shutdown(self):
        if self._arm:
            self._arm._pen_up()
            self._arm._fast_move(0, 0)
            self._arm.wait_idle()
            self._arm.close()
        if self._cam:
            self._cam.close()
