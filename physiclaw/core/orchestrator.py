"""
PhysiClaw orchestrator — central hardware lifecycle manager.

Owns the stylus arm, camera, and calibration state. Construction is
instant — call connect_arm() and connect_camera() to set up hardware.
Calibration is done via /setup skill endpoints.

The class stays narrow: lifecycle, concurrency, hardware access,
primitive movements, and the high-level tool operations invoked by
MCP tools. Image processing (rendering, drawing, encoding, vision
pipelines) lives in physiclaw.vision — the orchestrator only
coordinates sub-modules, it never touches pixels directly.
"""

import logging
import threading
import time
from contextlib import contextmanager
from typing import Literal

from physiclaw.bridge import BridgeState
from physiclaw.calibration import Calibration, ScreenTransforms
from physiclaw.hardware.arm import StylusArm
from physiclaw.hardware.camera import Camera
from physiclaw.hardware.iphone import AssistiveTouch
from physiclaw.vision.icon_detect import IconDetector
from physiclaw.vision.ocr import OCRReader, results_to_elements
from physiclaw.vision.util import (
    bbox_on_screen, decode_image, encode_jpeg, format_elements,
    find_numpad_digit, validate_bbox,
)
from physiclaw.vision.ui_elements import detect_ui_elements, elements_to_json
from physiclaw.vision.watchdog import Watchdog

log = logging.getLogger(__name__)


class PhysiClaw:
    """Central orchestrator — owns hardware lifecycle and the busy lock.

    Construction is instant (no hardware). Call connect_arm() and
    connect_camera() to connect hardware. Calibration is handled
    by the /setup skill via HTTP endpoints.
    """

    def __init__(self):
        self._arm: StylusArm | None = None
        self._cam: Camera | None = None
        self.calibration: Calibration = Calibration()
        self._lock = threading.Lock()
        self._assistive_touch = AssistiveTouch()
        self._bridge: BridgeState | None = None
        self._ocr_reader: OCRReader | None = None
        self._icon_detector: IconDetector | None = None
        self._watchdog = Watchdog()
        self._ready = False  # set True only after /setup finishes its last step

    # ─── Wiring ──────────────────────────────────────────────

    def attach_bridge(self, bridge: BridgeState) -> None:
        """Attach the server-side bridge. Called once from
        ``physiclaw.server.app`` at assembly time; screenshot and
        send_to_clipboard rely on it."""
        self._bridge = bridge

    # ─── Ready state ──────────────────────────────────────────

    @property
    def ready(self) -> bool:
        """True only after setup has fully completed AND hardware is still up."""
        return self._ready and self.hardware_ready

    def mark_ready(self) -> None:
        """Called by /setup after its final step (phone on Home Screen)."""
        self._ready = True

    # ─── State queries ────────────────────────────────────────

    @property
    def hardware_ready(self) -> bool:
        """True when arm, camera, and grid calibration are all set."""
        return (
            self._arm is not None
            and self._cam is not None
            and self.calibration.transforms_ready
        )

    def status(self) -> dict:
        """Return current hardware and calibration state."""
        steps = self.calibration.summary()
        if self._arm and self._arm.MOVE_DIRECTIONS:
            steps["alignment"] = "OK"
        if self._assistive_touch.ready:
            sx, sy = self._assistive_touch.at_screen
            steps["assistive_touch"] = f"({sx:.3f}, {sy:.3f})"
        return {
            "arm": self._arm is not None,
            "camera": self._cam is not None,
            "steps": steps,
            "calibrated": self.hardware_ready,
            "ready": self.ready,
        }

    def require_hardware(self):
        """Raise if hardware isn't connected and calibrated. (Doesn't check
        the `ready` flag — `home_screen()` in setup's final step needs tools
        before `ready` is flipped.)"""
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

    @contextmanager
    def locked(self):
        """Check hardware, acquire lock, auto-park on exit, then release."""
        self.require_hardware()
        self.acquire()
        try:
            yield
        finally:
            try:
                self.park()
            except Exception:
                pass
            self.release()

    # ─── Watchdog ────────────────────────────────────────────

    def watch(self) -> dict:
        """Poll the camera for wake events. Returns ``{"wake": bool, "reason": str}``."""
        with self.locked():
            frame = self.cam.peek()
            if frame is None:
                return {"wake": False, "reason": ""}
            return self._watchdog.poll(frame, self.transforms)


    # ─── Hardware connection ──────────────────────────────────

    def connect_arm(self):
        """Connect to the GRBL stylus arm (auto-detect USB port).

        Closes any previously connected arm first. Keeps the current
        Calibration bundle — on a warm restart we propagate its z_tap
        and direction mapping to the freshly-constructed arm so we can
        fast-path through setup. If the bundle is stale (arm swapped,
        phone moved), the user can delete data/calibration/bundle.json
        to force a fresh calibration.
        """
        if self._arm is not None:
            self._arm.close()
            self._arm = None
        self._arm = StylusArm()
        self._arm.setup()
        self._apply_bundle_to_arm()
        log.info("Arm connected")

    def connect_camera(self, index: int):
        """Open a camera by index.

        Closes any previously connected camera first. The user picks the
        index after previewing each one via /api/camera-preview/{index}
        during /setup, so we don't try to auto-detect. Propagates the
        cached rotation from the Calibration bundle if one is loaded.
        """
        if self._cam is not None:
            self._cam.close()
            self._cam = None
        self._cam = Camera(index)
        if self.calibration.cam_rotation is not None:
            self._cam.rotation = self.calibration.cam_rotation
        log.info(f"Camera {index} connected")

    def _apply_bundle_to_arm(self):
        """Propagate cached calibration into the newly-connected arm."""
        if self._arm is None:
            return
        cal = self.calibration
        if cal.z_tap is not None:
            self._arm.Z_DOWN = cal.z_tap
        if cal.pct_to_grbl is not None:
            p = cal.pct_to_grbl
            right_vec = (float(p[0, 0]), float(p[1, 0]))
            down_vec = (float(p[0, 1]), float(p[1, 1]))
            self._arm.set_direction_mapping(right_vec, down_vec)

    # ─── Hardware accessors ───────────────────────────────────

    @property
    def arm(self) -> StylusArm:
        return self._arm

    @property
    def cam(self) -> Camera:
        return self._cam

    @property
    def transforms(self) -> ScreenTransforms | None:
        return self.calibration.transforms()

    @property
    def assistive_touch(self) -> AssistiveTouch:
        return self._assistive_touch

    # ─── Primitive movements ─────────────────────────────────

    def park(self):
        """Move stylus off-screen to (-0.1, -0.05) — left of the screen, slightly above top edge."""
        gx, gy = self.transforms.pct_to_grbl_mm(-0.1, -0.05)
        self._arm._fast_move(gx, gy)
        self._arm.wait_idle()

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
        t = self.transforms
        if t is None:
            raise RuntimeError("Screen calibration not done")
        cx, cy = t.bbox_center_pct(bbox)
        gx, gy = t.pct_to_grbl_mm(cx, cy)
        self._arm._fast_move(gx, gy)
        self._arm.wait_idle()

    # ─── Tool operations ───────────────────────────────────────


    def _require_assistive_touch(self):
        """Raise if AssistiveTouch isn't calibrated. ``_bridge`` is wired
        into the constructor at server startup, so it's always present."""
        if not self._assistive_touch.ready:
            raise RuntimeError("AssistiveTouch not calibrated — run /setup first")

    def _get_ocr_reader(self) -> OCRReader:
        """Lazy-load and cache the OCR reader."""
        if self._ocr_reader is None:
            self._ocr_reader = OCRReader()
        return self._ocr_reader

    def _get_icon_detector(self) -> IconDetector:
        """Lazy-load and cache the icon detector."""
        if self._icon_detector is None:
            self._icon_detector = IconDetector()
        return self._icon_detector

    def _scan(self) -> list[dict]:
        """OCR the screen → list of element dicts. Caller must hold the lock.

        Filters out elements outside the phone screen — the camera
        also captures the desk, ruler, etc.
        """
        self.park()
        frame = self.camera_view()
        results = self._get_ocr_reader().read(frame)
        elements = results_to_elements(results, self.transforms)
        return [e for e in elements if bbox_on_screen(e["bbox"])]

    def scan(self) -> str:
        """OCR the overhead camera view. Returns JSON list of text elements.

        Same schema as screenshot() but text-only (no icons), and
        bboxes are transformed from camera pixels to screen 0-1.
        """
        with self.locked():
            return format_elements(self._scan())

    def peek(self) -> bytes:
        """Quick camera snapshot. Returns JPEG-encoded bytes."""
        with self.locked():
            self.park()
            return encode_jpeg(self.camera_view())

    def screenshot(self) -> tuple[bytes, str]:
        """Pixel-perfect phone screenshot with UI elements detected.

        Returns JPEG bytes of the annotated image (numbered bboxes)
        and a pretty-printed JSON listing of detected elements.
        """
        with self.locked():
            self._require_assistive_touch()
            data = self._assistive_touch.take_screenshot(
                self._arm, self._bridge, self.transforms.pct_to_grbl, timeout=60.0
            )
            if data is None:
                raise TimeoutError(
                    "Screenshot upload timed out — check the iOS Shortcut"
                )

            frame = decode_image(data)
            elements, annotated = detect_ui_elements(
                frame,
                icon_detector=self._get_icon_detector(),
                ocr_reader=self._get_ocr_reader(),
            )
            return encode_jpeg(annotated), format_elements(elements_to_json(elements))

    # ─── Gesture primitives (no lock) ──────────────────────────

    def _tap(self, bbox: list[float]):
        """Tap at bbox center. Caller must hold the lock."""
        self.move_to_bbox_center(bbox)
        self._arm.tap()
        self._arm.wait_idle()

    def _double_tap(self, bbox: list[float]):
        """Double tap at bbox center. Caller must hold the lock."""
        self.move_to_bbox_center(bbox)
        self._arm.double_tap()
        self._arm.wait_idle()

    def _long_press(self, bbox: list[float]):
        """Long press at bbox center. Caller must hold the lock."""
        self.move_to_bbox_center(bbox)
        self._arm.long_press()
        self._arm.wait_idle()

    _SWIPE_DISTANCES = {"s": 0.1, "m": 0.3, "l": 0.5, "xl": 0.75, "xxl": 0.90}
    _SWIPE_DIRS = ("up", "down", "left", "right")
    _SWIPE_SPEEDS = ("slow", "medium", "fast")

    def _swipe(
        self,
        bbox: list[float],
        direction: Literal["up", "down", "left", "right"],
        size: Literal["s", "m", "l", "xl", "xxl"] = "m",
        speed: Literal["slow", "medium", "fast"] = "medium",
    ):
        """Swipe from bbox center. Caller must hold the lock."""
        t = self.transforms
        ex, ey = t.swipe_end_pct(bbox, direction, self._SWIPE_DISTANCES[size])
        ex_mm, ey_mm = t.pct_to_grbl_mm(ex, ey)
        self.move_to_bbox_center(bbox)
        arm = self._arm
        arm._pen_down()
        arm._linear_move(ex_mm, ey_mm, speed=arm.SWIPE_SPEEDS[speed])
        arm._pen_up()
        arm.wait_idle()

    # ─── Public gestures (with lock) ─────────────────────────

    def tap(self, bbox: list[float]) -> str:
        """Single tap at the center of a bbox."""
        validate_bbox(bbox)
        with self.locked():
            self._tap(bbox)
            return f"Tapped at bbox {bbox}"

    def double_tap(self, bbox: list[float]) -> str:
        """Double tap at the center of a bbox."""
        validate_bbox(bbox)
        with self.locked():
            self._double_tap(bbox)
            return f"Double tapped at bbox {bbox}"

    def long_press(self, bbox: list[float]) -> str:
        """Long press (~1.2s) at the center of a bbox."""
        validate_bbox(bbox)
        with self.locked():
            self._long_press(bbox)
            return f"Long pressed at bbox {bbox}"

    def swipe(
        self,
        bbox: list[float],
        direction: Literal["up", "down", "left", "right"],
        size: Literal["s", "m", "l", "xl", "xxl"] = "m",
        speed: Literal["slow", "medium", "fast"] = "medium",
    ) -> str:
        """Swipe from the bbox center in `direction` by `size` screen fraction."""
        validate_bbox(bbox)
        if direction not in self._SWIPE_DIRS:
            raise ValueError(
                f"direction must be one of {self._SWIPE_DIRS}, got {direction!r}"
            )
        if size not in self._SWIPE_DISTANCES:
            raise ValueError(
                f"size must be one of {list(self._SWIPE_DISTANCES)}, got {size!r}"
            )
        if speed not in self._SWIPE_SPEEDS:
            raise ValueError(
                f"speed must be one of {self._SWIPE_SPEEDS}, got {speed!r}"
            )
        with self.locked():
            self._swipe(bbox, direction, size, speed)
            return f"Swiped {direction} {size} at bbox {bbox}"

    def send_to_clipboard(self, text: str) -> str:
        """Copy text to the phone's clipboard via AT long-press."""
        with self.locked():
            self._require_assistive_touch()
            self._bridge.send_text(text)
            self._assistive_touch.long_press(self._arm, self.transforms.pct_to_grbl)
            if self._bridge.wait_clipboard(timeout=30.0):
                return f"Copied '{text}' to phone clipboard"
            return "AT long-pressed but clipboard not confirmed — check the iOS Shortcut"

    def home_screen(self) -> str:
        """Go to the home screen via bottom-edge swipe up."""
        with self.locked():
            self._swipe([0.4, 0.96, 0.6, 0.98], "up", "xl", speed="fast")
            return "Went to home screen"

    def go_back(self) -> str:
        """Go back one screen via left-edge swipe right."""
        with self.locked():
            self._swipe([0.0, 0.4, 0.04, 0.6], "right", "xxl", speed="fast")
            return "Went back"

    def unlock_phone(self) -> str:
        """Unlock the phone: wake → swipe up → wait for Face ID to fail → enter passcode.

        Fully mechanical — no AI. OCR finds digit "1" on the passcode
        screen, then taps it six times. Passcode is hardcoded to 111111 —
        a dedicated tool-phone passcode, not the user's real password.
        """
        with self.locked():
            self._tap([0.4, 0.4, 0.6, 0.6])
            self._swipe([0.4, 0.96, 0.6, 0.98], "up", "l", speed="fast")
            self.park()
            time.sleep(4)  # Face ID starts

            # Poll for passcode keypad (Face ID fails after a few seconds)
            digit_bbox = None
            for _ in range(8):
                elements = self._scan()
                digit_bbox = find_numpad_digit(elements, "1")
                if digit_bbox is not None:
                    break
                time.sleep(1)

            if digit_bbox is None:
                return "Failed to find passcode keypad — phone may already be unlocked"

            for _ in range(6):
                self._tap(digit_bbox)

            return "Passcode entered"

    # ─── Lifecycle ─────────────────────────────────────────────

    def shutdown(self):
        if self._arm:
            self._arm._pen_up()
            self._arm.return_to_origin()
            self._arm.close()
        if self._cam:
            self._cam.close()
