"""
PhysiClaw orchestrator — central hardware lifecycle manager.

Owns the stylus arm, camera, and calibration state. Construction is
instant — call connect_arm() and connect_camera() to set up hardware.
Calibration is done via /setup skill endpoints.

The class stays narrow: lifecycle, concurrency, hardware access,
primitive movements, and the high-level tool operations invoked by
MCP tools. Image processing (rendering, drawing, encoding, vision
pipelines) lives in physiclaw.core.vision — the orchestrator only
coordinates sub-modules, it never touches pixels directly.
"""

import json
import logging
import threading
import time
from contextlib import contextmanager
from typing import Any, assert_never

from physiclaw.common import paths
from physiclaw.common.config import CONFIG
from physiclaw.common.gesture_vocab import STEP_ARG, STEP_TOOL
from physiclaw.common.text import read_text
from physiclaw.core.bridge import BridgeState
from physiclaw.core.calibration import PARK_PCT, Calibration, ScreenTransforms
from physiclaw.core.hardware import exposure
from physiclaw.core.hardware.arm import StylusArm
from physiclaw.core.hardware.camera import Camera
from physiclaw.core.hardware.iphone import AssistiveTouch
from physiclaw.core.orchestration import gestures
from physiclaw.core.orchestration.clipboard import (
    ClipboardSyncError,
    ClipboardSyncState,
)
from physiclaw.core.orchestration.observation import GestureObserver, GestureResult
from physiclaw.core.vision import quality
from physiclaw.core.vision.icon_detect import IconDetector
from physiclaw.core.vision.ocr import OCRReader, results_to_elements
from physiclaw.core.vision.preprocess import (
    crop_to_phone_screen,
    phone_screen_crop_box,
)
from physiclaw.core.vision.ui_elements import detect_ui_elements, elements_to_json
from physiclaw.core.vision.util import (
    bbox_on_screen,
    decode_image,
    encode_view_jpeg,
    find_numpad_digit,
    format_elements,
    validate_bbox,
)
from physiclaw.core.vision.watchdog import Watchdog

log = logging.getLogger(__name__)


def _layout_learned() -> bool:
    """Read the first-run layout's `learned` marker straight from its JSON.

    The agent-side `screen_layout` module writes the marker (it owns the
    page/field schema); core only reads it, so the orchestrator can report
    first-run progress without importing the agent package."""
    p = paths.screen_layout_json()
    if not p.exists():
        return False
    try:
        data = json.loads(read_text(p))
    except (OSError, json.JSONDecodeError):
        return False
    # "layout_learned" is written by agent-side screen_layout.record() once
    # every box is captured; core only reads it (keep in sync with that writer).
    return bool(isinstance(data, dict) and data.get("layout_learned"))


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
        # Accessor lambdas, not the objects: both are replaced after
        # construction (calibration, tests), and the validator must see
        # the current ones.
        self._validator = gestures.GestureValidator(
            assistive_touch=lambda: self._assistive_touch,
            transforms=lambda: self.transforms,
        )
        self._bridge: BridgeState | None = None
        self._clipboard = ClipboardSyncState()
        self._ocr_reader: OCRReader | None = None
        self._icon_detector: IconDetector | None = None
        self._watchdog = Watchdog()
        # Late-binding lambdas for the same reason as the validator's:
        # `crop_to_phone_screen` / `_detect` must resolve at call time.
        self._observer = GestureObserver(
            park=self.park,
            grab=lambda: crop_to_phone_screen(self.camera_view(), self.transforms),
            detect=lambda frame: self._detect(frame),
        )
        self._ready = False  # set True only after /setup finishes its last step

    # ─── Wiring ──────────────────────────────────────────────

    def attach_bridge(self, bridge: BridgeState) -> None:
        """Attach the server-side bridge. Called once from
        ``physiclaw.core.server.app`` at assembly time; screenshot and
        send_to_clipboard rely on it."""
        self._bridge = bridge

    # ─── Ready state ──────────────────────────────────────────

    @property
    def ready(self) -> bool:
        """True only after setup has fully completed AND hardware is still up."""
        return self._ready and self.hardware_ready

    def mark_ready(self) -> None:
        """Called by /setup after its final step (phone on Home Screen).

        Kept pure (tests pin it). Any NEW path that flips ready should
        also schedule `tune_exposure()` — see the two existing callers
        (`server/watch.py` /api/ready, `server/warm_start.py`)."""
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
            "bridge": (self._bridge.connected if self._bridge is not None else False),
            "steps": steps,
            "calibrated": self.hardware_ready,
            "ready": self.ready,
            "layout_learned": _layout_learned(),
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

    # ─── Exposure tune ───────────────────────────────────────

    def tune_exposure(self) -> None:
        """Startup verify-and-converge for camera exposure. Fail-open:
        never raises, never blocks a session.

        Runs once the transforms exist (setup finished, or a warm-start
        bundle loaded) and the phone is on the home screen — the dark
        scene that exposed the Windows AE failure. Meters the PHONE-SCREEN
        crop, never the whole frame: the dark desk around the phone would
        otherwise dominate the scene and drive AE to blow out the screen.
        macOS exits at `exposure_tunable` (AVFoundation ignores exposure
        props; its firmware AE works). Worst case a few seconds, bounded
        by `wait_frames` timeouts; skipped entirely if the hardware is
        busy. A failed tune reverts to auto — the runtime quality monitor
        keeps warning the agent, so nothing is lost."""
        cam, t = self._cam, self.transforms
        if cam is None or t is None or not cam.exposure_tunable:
            return
        try:
            self.acquire()
        except RuntimeError:
            log.info("exposure tune: hardware busy — skipped")
            return
        try:
            self.park()  # stylus off the glass — it would pollute the crop

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

        Closes any previously connected arm first. ``_apply_bundle_to_arm``
        propagates the cached direction mapping into the freshly-
        constructed arm IF a bundle has been loaded into
        ``self.calibration`` — only true on ``--warm-start``. Plain
        ``physiclaw server`` boots with empty calibration, so the
        propagation is a no-op and step 7 of setup measures fresh.
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
        cached rotation from ``self.calibration`` IF a bundle has been
        loaded — only true on ``--warm-start``. Plain ``physiclaw server``
        boots with empty calibration, so step 8 of setup detects rotation
        fresh.
        """
        if self._cam is not None:
            self._cam.close()
            self._cam = None
        self._cam = Camera(index)
        if self.calibration.cam_rotation is not None:
            self._cam.rotation = self.calibration.cam_rotation
        log.info(f"Camera {index} connected")

    def disconnect_camera(self) -> bool:
        """Release the camera device handle so another app can use it.

        Returned True if a camera was actually closed. Used by `/setup`
        step 8 on Windows so the OS Camera preview app can claim the
        device — Media Foundation enforces exclusive access, so the
        server has to let go before the aim app opens.
        """
        if self._cam is None:
            return False
        self._cam.close()
        self._cam = None
        log.info("Camera disconnected")
        return True

    def restore_park_origin(self) -> bool:
        """Re-pin the GRBL origin assuming the tip rests at the off-screen
        park spot. Warm-start's counterpart to ``_park_for_teardown``.

        ``arm.setup()`` on reconnect issues ``G92 X0 Y0``, declaring the
        arm's *current* physical position to be GRBL ``(0, 0)``. That's only
        the calibrated origin if the tip is sitting there — but clean
        shutdown (and every inter-action ``locked()`` park) leaves it at the
        park spot instead. So re-declare the current position as the park
        coordinate from the loaded bundle, restoring the affine's frame;
        otherwise every subsequent tap is offset by the park vector.

        Returns False (no-op) if the arm or `pct_to_grbl` isn't ready — the
        caller (warm-start) only invokes this with a complete bundle loaded.
        """
        if self._arm is None:
            return False
        park_xy = self.calibration.pct_to_grbl_mm(*PARK_PCT)
        if park_xy is None:
            return False
        self._arm.set_work_position(*park_xy)
        log.info("Re-pinned GRBL origin from park spot %s", PARK_PCT)
        return True

    def _apply_bundle_to_arm(self):
        """Propagate cached calibration into the newly-connected arm."""
        if self._arm is None:
            return
        cal = self.calibration
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
        """Move stylus off-screen to ``PARK_PCT`` — left of the screen,
        slightly above top edge.

        Defensive: no-ops if the arm isn't connected or `pct_to_grbl`
        isn't set yet. This makes parking safe between calibration
        steps (e.g. after step 7 when only the arm-side affine is
        ready, or before step 7 when nothing is). Caller must hold
        the hardware lock.
        """
        if self._arm is None:
            return
        park_xy = self.calibration.pct_to_grbl_mm(*PARK_PCT)
        if park_xy is None:
            return
        self._arm._fast_move(*park_xy)
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

    def _scan_text(self) -> list[dict]:
        """OCR-only pass on the phone-screen region. Caller must hold the lock.

        Fast path for internal polling (e.g. unlock_phone's keypad loop).
        The agent-facing tools go through ``_detect`` instead, which also
        runs icon detection.
        """
        self.park()
        frame = self.camera_view()
        results = self._get_ocr_reader().read(
            frame, crop_box=phone_screen_crop_box(frame, self.transforms)
        )
        elements = results_to_elements(results, self.transforms)
        return [e for e in elements if bbox_on_screen(e["bbox"])]

    def _detect(self, frame) -> tuple[str, Any]:
        """Icon detection + OCR on a frame. Caller holds the lock.

        ``frame`` must already span the phone screen (0-1) — camera
        views need to be cropped via ``crop_to_phone_screen`` first.
        Returns (formatted element listing, annotated frame).
        """
        elements, annotated = detect_ui_elements(
            frame,
            icon_detector=self._get_icon_detector(),
            ocr_reader=self._get_ocr_reader(),
        )
        return format_elements(elements_to_json(elements)), annotated

    def peek(self) -> tuple[bytes, str]:
        """Overhead camera snapshot + icon detection + OCR.

        If the first frame is too blurry, waits 2s and re-grabs once
        (``GestureObserver.peek_frame`` owns that acquisition policy).

        Returns an annotated JPEG (icon bboxes drawn on the cropped
        camera view) and the matching element listing — same shape as
        screenshot(), but from the camera rather than the phone's own
        screenshot.
        """
        with self.locked():
            cropped = self._observer.peek_frame()
            listing, annotated = self._detect(cropped)
            warning = self._observer.observe_quality("peek", cropped)
            if warning is not None:
                listing = f"{listing}\n{warning}"
            return encode_view_jpeg(annotated), listing

    def screenshot(self) -> tuple[bytes, str]:
        """Pixel-perfect phone screenshot + icon detection + OCR.

        Returns an annotated JPEG (icon bboxes drawn) and the matching
        element listing — same shape as peek(), but sourced from the
        phone's own screenshot instead of the camera.
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
            # Detection runs on the native-resolution screenshot;
            # encode_view_jpeg caps the image the agent sees.
            listing, annotated = self._detect(frame)
            return encode_view_jpeg(annotated), listing

    # ─── Gesture primitives ────────────────────────────────────

    def _tap(self, bbox: list[float]):
        """Tap at bbox center. Caller must hold the lock."""
        self._validator.require_no_at_overlap(bbox, "tap")
        self.move_to_bbox_center(bbox)
        self._arm.tap()
        self._arm.wait_idle()

    def _double_tap(self, bbox: list[float]):
        """Double tap at bbox center. Caller must hold the lock."""
        self._validator.require_no_at_overlap(bbox, "double_tap")
        self.move_to_bbox_center(bbox)
        self._arm.double_tap()
        self._arm.wait_idle()

    def _long_press(self, bbox: list[float]):
        """Long press at bbox center. Caller must hold the lock."""
        self._validator.require_no_at_overlap(bbox, "long_press")
        self.move_to_bbox_center(bbox)
        self._arm.long_press()
        self._arm.wait_idle()

    def _swipe(
        self,
        bbox: list[float],
        direction: gestures.Direction,
        size: gestures.Size = "m",
        speed: gestures.Speed = "medium",
        start_dwell: float = 0.0,
    ):
        """Swipe from bbox center. Caller must hold the lock. `start_dwell` (s)
        anchors the touch-down at the start before sliding (see arm.swipe_to)."""
        self._validator.require_no_at_crossing(bbox, direction)
        t = self.transforms
        ex, ey = t.swipe_end_pct(bbox, direction, gestures.SWIPE_DISTANCES[size])
        ex_mm, ey_mm = t.pct_to_grbl_mm(ex, ey)
        self.move_to_bbox_center(bbox)
        self._arm.swipe_to(ex_mm, ey_mm, speed, start_dwell=start_dwell)

    # ─── Screen-change verdict ────────────────────────────────

    def _observed(self, act) -> "GestureResult":
        """Lock, run `act` (which returns its action text) bracketed by
        the observer's verdict/view frames — the single lock +
        observation bracket every mutating tool goes through."""
        with self.locked():
            return self._observer.with_view(act)

    def _execute(self, step: gestures.Gesture) -> str:
        """Run one typed, already-validated gesture and compose its
        action text — the single dispatch point shared by the public
        gesture methods and `sequence` steps. Caller must hold the lock."""
        match step:
            case gestures.Tap(bbox):
                self._tap(bbox)
                return f"Tapped at bbox {bbox}"
            case gestures.DoubleTap(bbox):
                self._double_tap(bbox)
                return f"Double tapped at bbox {bbox}"
            case gestures.LongPress(bbox):
                self._long_press(bbox)
                return f"Long pressed at bbox {bbox}"
            case gestures.Swipe(bbox, direction, size, speed):
                self._swipe(bbox, direction, size, speed)
                return f"Swiped {direction} {size} at bbox {bbox}"
            case gestures.SendToClipboard(text):
                return self._send_to_clipboard(text)
            case _:
                assert_never(step)

    def _run_gesture(self, step: gestures.Gesture) -> "GestureResult":
        """Run one typed gesture under the observation bracket."""
        return self._observed(lambda: self._execute(step))

    def _run_macro(self, gesture, result: str) -> "GestureResult":
        """Lock, run the multi-step `gesture` body bracketed by
        verdict/view frames, and return the fixed `result` text fused
        with the post-gesture view — the shared shape of the macro
        gestures (home_screen / go_back / force_quit)."""

        def act() -> str:
            gesture()
            return result

        return self._observed(act)

    # ─── Public gestures (with lock) ─────────────────────────

    def tap(self, bbox: list[float]) -> "GestureResult":
        """Single tap at the center of a bbox."""
        validate_bbox(bbox)
        return self._run_gesture(gestures.Tap(bbox))

    def double_tap(self, bbox: list[float]) -> "GestureResult":
        """Double tap at the center of a bbox."""
        validate_bbox(bbox)
        return self._run_gesture(gestures.DoubleTap(bbox))

    def long_press(self, bbox: list[float]) -> "GestureResult":
        """Long press (~1.2s) at the center of a bbox."""
        validate_bbox(bbox)
        return self._run_gesture(gestures.LongPress(bbox))

    def swipe(
        self,
        bbox: list[float],
        direction: gestures.Direction,
        size: gestures.Size = "m",
        speed: gestures.Speed = "medium",
    ) -> "GestureResult":
        """Swipe from the bbox center in `direction` by `size` screen fraction.

        `no visible change` in the verdict after a scroll swipe = end of
        the list."""
        self._validator.validate_swipe(bbox, direction, size, speed)
        return self._run_gesture(gestures.Swipe(bbox, direction, size, speed))

    def _send_to_clipboard(self, text: str) -> str:
        """Copy text via AssistiveTouch long-press. Caller must hold the lock.

        Raises ``ClipboardSyncError`` when the phone never fetches the
        text — the phone's clipboard still holds the PREVIOUS content,
        so any follow-up paste would insert stale text. Raising (not
        returning) makes a ``sequence`` abort before its paste/send
        steps; the standalone tool surfaces it as a tool error.

        The messages deliberately do NOT echo `text`: the agent already
        sees it in its own tool_call args, and inside a `sequence` this
        string joins the action text the engine scans for the screen
        verdict — echoed free-form content (often quoted from an IM or
        the screen) could carry marker-like text (`physiclaw.common.verdict`)."""
        self._require_assistive_touch()
        timeout = self._clipboard.begin()
        self._bridge.send_text(text)
        self._assistive_touch.long_press(self._arm, self.transforms.pct_to_grbl)
        if self._bridge.wait_clipboard(timeout=timeout):
            self._clipboard.confirm()
            return f"Copied {len(text)} chars to phone clipboard"
        # Un-queue the text so a LATE Shortcut run can't fetch it after
        # this timeout — the error below promises the phone clipboard
        # still holds the previous content; keep that true, not racy.
        self._bridge.clear_text()
        raise ClipboardSyncError(self._clipboard.record_miss())

    def send_to_clipboard(self, text: str) -> str:
        """Copy text to the phone's clipboard via AssistiveTouch long-press."""
        with self.locked():
            return self._send_to_clipboard(text)

    def _run_step(self, tool: str, arg) -> str:
        """Dispatch one sequence step. Caller must hold the lock.

        `parse_step` resolves the string-keyed arg into a typed gesture
        (raising the agent-facing ValueError on bad shape/range);
        `_execute` runs it — the same executor the public gesture
        methods use.
        """
        return self._execute(self._validator.parse_step(tool, arg))

    def sequence(self, steps: list[dict]) -> "GestureResult":
        """Run multiple gestures atomically — one lock held across all steps.

        Each step: {"tool_name": str, "arg": ...}. Stops on first failure.
        Lock is acquired once; no park between steps. Per-step camera
        checks are deliberately absent: the stylus occludes the screen
        mid-batch, and the sanctioned fast paths (paste popovers) EXPECT
        the screen to change between steps — so safety comes from the
        grounding policy (CONVENTION § Sequence bundling) plus the
        whole-batch verdict and fused final view returned here.
        """

        def act() -> str:
            lines = []
            for i, s in enumerate(steps, 1):
                tool = s[STEP_TOOL]
                try:
                    result = self._run_step(tool, s.get(STEP_ARG))
                    lines.append(f"{i} {tool} ok — {result}")
                except Exception as e:
                    lines.append(f"{i} {tool} FAIL ({e})")
                    break
            return "\n".join(lines)

        return self._observed(act)

    def home_screen(self) -> "GestureResult":
        """Go to the home screen via bottom-edge swipe up."""
        return self._run_macro(
            lambda: self._swipe([0.4, 0.96, 0.6, 0.98], "up", "xl", speed="fast"),
            "Went to home screen",
        )

    # Hold the tip at the very edge this long after contact, before the slide,
    # so iOS registers a touch-down inside the narrow interactive-pop zone —
    # without it, a fast slide starting on contact skips past the edge and the
    # gesture reads as a content scroll (no back). ~9 display frames; 0.08
    # (~1.5 frames) was too tight and intermittently missed the arming window.
    BACK_EDGE_DWELL_SECONDS = 0.15

    def go_back(self) -> "GestureResult":
        """Go back one screen via left-edge swipe right.

        Starts at the true left edge (x≈0) and dwells briefly on contact so the
        touch is seen inside iOS's edge-pan zone before the slide arms the
        interactive pop. `no visible change` in the verdict = the edge swipe
        didn't pop (modal, root tab, image viewer) — the cue to try another
        exit."""
        return self._run_macro(
            lambda: self._swipe(
                [0.0, 0.4, 0.01, 0.6],
                "right",
                "xxl",
                speed="fast",
                start_dwell=self.BACK_EDGE_DWELL_SECONDS,
            ),
            "Went back",
        )

    def force_quit(self) -> "GestureResult":
        """Force-quit current app via the iOS app-switcher gesture.

        Step 1's swipe is short on purpose — a longer upward swipe from
        the bottom edge would go home instead of opening the switcher.
        """

        def steps() -> None:
            self._swipe([0.4, 0.96, 0.6, 0.98], "up", "s", speed="slow")
            self._swipe([0.4, 0.45, 0.6, 0.55], "left", "m", speed="medium")
            self._swipe([0.4, 0.70, 0.6, 0.80], "up", "xl", speed="fast")
            self._tap([0.4, 0.92, 0.6, 0.96])

        return self._run_macro(steps, "Force-quit current app")

    def unlock_phone(self) -> "GestureResult":
        """Unlock the phone: wake → swipe up → wait for Face ID to fail → enter passcode.

        Fully mechanical — no AI. OCR finds digit "1" on the passcode
        screen, then taps it six times. Passcode is hardcoded to 111111 —
        a dedicated tool-phone passcode, not the user's real password.
        """

        def act() -> str:
            # Warm the OCR model before waking the phone — a cold RapidOCR
            # load (seconds) inside the keypad's ~8s lifetime lets the
            # numpad sleep again before the first tap lands.
            self._get_ocr_reader()
            self._tap([0.4, 0.4, 0.6, 0.6])
            self._swipe([0.4, 0.96, 0.6, 0.98], "up", "l", speed="fast")
            self.park()
            time.sleep(2)  # Face ID starts; the poll below absorbs the rest

            # Poll for passcode keypad (Face ID fails after a few seconds)
            digit_bbox = None
            for _ in range(8):
                elements = self._scan_text()
                digit_bbox = find_numpad_digit(elements, "1")
                if digit_bbox is not None:
                    break
                time.sleep(1)

            if digit_bbox is None:
                return "Failed to find passcode keypad — phone may already be unlocked"

            for _ in range(6):
                self._tap(digit_bbox)

            return "Passcode entered"

        return self._observed(act)

    # ─── Lifecycle ─────────────────────────────────────────────

    def _park_for_teardown(self):
        """Rest the stylus at the same off-screen park spot used between
        taps and swipes, so the user can place or lift the phone without
        the tip sitting over the glass.

        Falls back to homing (0, 0) only when calibration isn't loaded —
        ``park()`` has no target to compute then, and leaving the tip
        mid-travel would be worse than a known machine origin.
        """
        if self.calibration.pct_to_grbl is not None:
            self.park()
        else:
            self._arm.return_to_origin()

    def shutdown(self):
        """Release the coil, park the arm off-screen, and close every device
        handle.

        Teardown is best-effort: each step is guarded so a failure in one (a
        serial timeout, a GRBL alarm, an already-disconnected device) can't
        skip the rest and leak the serial/camera handle or strand the coil.
        Steps run in safety order — release the coil first; ``arm.close`` also
        re-attempts the release as a backstop, and the firmware drops the PWM
        on alarm. Failures are logged, never raised, so callers (atexit /
        signal handlers) can rely on shutdown completing.

        The arm parks at the same off-screen spot used between taps (homing
        only if uncalibrated) so the resting position is consistent and the
        phone stays clear for placement / removal.
        """

        def _safe(action, desc):
            try:
                action()
            except Exception:
                log.exception("shutdown: %s failed", desc)

        if self._arm:
            _safe(self._arm.lift_stylus, "lift stylus")
            _safe(self._park_for_teardown, "park")
            _safe(self._arm.close, "arm close")
        if self._cam:
            _safe(self._cam.close, "camera close")
