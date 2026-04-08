"""
PhysiClaw orchestrator — central hardware lifecycle manager.

Owns the stylus arm, camera, calibration state, and the bbox workflow
state used by the propose-confirm tap flow. Construction is instant —
call connect_arm() and connect_camera() to set up hardware. Calibration
is done via /setup skill endpoints.

The class deliberately stays narrow: lifecycle, concurrency, hardware
access, primitive movements, and bbox state. All image processing
(rendering, drawing, encoding, vision pipelines) lives in physiclaw.vision.
One-shot utilities live near their callers (e.g. hardware/handler.py).
"""

import logging
import threading

from physiclaw.calibration import ScreenTransforms
from physiclaw.hardware.camera import Camera
from physiclaw.hardware.iphone import AssistiveTouch
from physiclaw.hardware.arm import StylusArm

log = logging.getLogger(__name__)


class PhysiClaw:
    """Central orchestrator — owns hardware lifecycle, the busy lock,
    and the bbox workflow state.

    Construction is instant (no hardware). Call connect_arm() and
    connect_camera() to connect hardware. Calibration is handled
    by the /setup skill via HTTP endpoints.
    """

    PARK_DISTANCE = 100  # mm to move stylus out of frame

    def __init__(self):
        self._arm: StylusArm | None = None
        self._cam: Camera | None = None
        self._transforms: ScreenTransforms | None = None
        self._pending_bboxes: dict[str, list[float]] = {}
        self._confirmed_bbox: list[float] | None = None
        self._lock = threading.Lock()
        self._cal: dict = {}  # intermediate calibration state between phases
        self._assistive_touch = AssistiveTouch()

    # ─── State queries ────────────────────────────────────────

    @property
    def hardware_ready(self) -> bool:
        """True when arm, camera, and grid calibration are all set."""
        return (
            self._arm is not None
            and self._cam is not None
            and self._transforms is not None
        )

    def status(self) -> dict:
        """Return current hardware and calibration state."""
        steps = {}
        z_tap = self._cal.get("z_tap")
        if z_tap is not None:
            steps["z_tap"] = f"{z_tap}mm"
        if "viewport_shift" in self._cal:
            t = self._cal["viewport_shift"]
            steps["viewport_shift"] = (
                f"dpr={t.dpr}, offset=({t.offset_x}, {t.offset_y})"
            )
        if self._arm and self._arm.MOVE_DIRECTIONS:
            steps["alignment"] = "OK"
        if "rotation" in self._cal:
            names = {-1: "none", 0: "90° CW", 1: "180°", 2: "90° CCW"}
            steps["rotation"] = names.get(
                self._cal["rotation"], str(self._cal["rotation"])
            )
        if "screen_to_grbl" in self._cal:
            steps["mapping_a"] = "OK"
        if "pct_to_cam" in self._cal:
            steps["mapping_b"] = "OK"
        if self._transforms is not None:
            steps["validated"] = True
        if self._assistive_touch.ready:
            sx, sy = self._assistive_touch.at_screen
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
                "Hardware not set up. Run /setup to connect and calibrate."
            )

    # ─── Concurrency ──────────────────────────────────────────

    def acquire(self):
        """Mark hardware as busy. Raises immediately if already busy."""
        if not self._lock.acquire(blocking=False):
            raise RuntimeError(
                "PhysiClaw is busy — wait for the current operation to finish, then retry."
            )

    def release(self):
        """Mark hardware as idle."""
        self._lock.release()

    # ─── Hardware connection ──────────────────────────────────

    def connect_arm(self):
        """Connect to the GRBL stylus arm (auto-detect USB port).

        Closes any previously connected arm first.
        """
        if self._arm is not None:
            self._arm.close()
            self._arm = None
        self._cal = {}  # reset calibration state
        self._transforms = None
        self._arm = StylusArm()
        self._arm.setup()
        log.info("Arm connected")

    def connect_camera(self, index: int):
        """Open a camera by index.

        Closes any previously connected camera first. The user picks the
        index after previewing each one via /api/camera-preview/{index}
        during /setup, so we don't try to auto-detect.
        """
        if self._cam is not None:
            self._cam.close()
            self._cam = None
        self._cam = Camera(index)
        log.info(f"Camera {index} connected")

    # ─── Hardware accessors ───────────────────────────────────

    @property
    def arm(self) -> StylusArm:
        return self._arm

    @property
    def cam(self) -> Camera:
        return self._cam

    @property
    def transforms(self) -> ScreenTransforms | None:
        return self._transforms

    @property
    def assistive_touch(self) -> AssistiveTouch:
        return self._assistive_touch

    # ─── Bbox workflow state ─────────────────────────────────

    def set_pending_bbox(self, bbox: list[float]):
        """Store a pending bbox. Clears any confirmed bbox.

        Args:
            bbox: [left, top, right, bottom] as 0-1 decimals.
        """
        self._pending_bboxes = {"center": list(bbox)}
        self._confirmed_bbox = None

    def pending_bbox(self) -> list[float] | None:
        """Read the bbox staged by set_pending_bbox(), or None."""
        return self._pending_bboxes.get("center")

    def confirm_bbox(self):
        """Lock in the pending bbox for the next gesture."""
        if not self._pending_bboxes:
            raise RuntimeError("No pending bbox — call bbox_target first")
        self._confirmed_bbox = self._pending_bboxes["center"]
        self._pending_bboxes = {}

    @property
    def confirmed_bbox(self) -> list[float] | None:
        """The bbox locked in for the next gesture, or None."""
        return self._confirmed_bbox

    # ─── Primitive movements ─────────────────────────────────

    def park(self):
        """Move the stylus 100mm out of the camera frame."""
        arm = self._arm
        if arm.MOVE_DIRECTIONS is None:
            raise RuntimeError("Cannot park — calibration has not been run yet")
        ux, uy = arm.MOVE_DIRECTIONS["top"]
        arm._fast_move(ux * self.PARK_DISTANCE, uy * self.PARK_DISTANCE)
        arm.wait_idle()

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

    def move_to_bbox_center(self, bbox: list[float]):
        """Move arm to the center of a bbox [left, top, right, bottom] (0-1)."""
        if self._transforms is None:
            raise RuntimeError("Screen calibration not done")
        cx, cy = self._transforms.bbox_center_pct(bbox)
        gx, gy = self._transforms.pct_to_grbl_mm(cx, cy)
        self._arm._fast_move(gx, gy)
        self._arm.wait_idle()

    def tap_at_pct(self, x: float, y: float):
        """Move arm to screen coordinate (0-1) and tap. Bypasses bbox workflow."""
        if self._transforms is None:
            raise RuntimeError("Screen calibration not done")
        gx, gy = self._transforms.pct_to_grbl_mm(x, y)
        self._arm._fast_move(gx, gy)
        self._arm.wait_idle()
        self._arm.tap()

    # ─── Lifecycle ─────────────────────────────────────────────

    def shutdown(self):
        if self._arm:
            self._arm._pen_up()
            self._arm._fast_move(0, 0)
            self._arm.wait_idle()
            self._arm.close()
        if self._cam:
            self._cam.close()
