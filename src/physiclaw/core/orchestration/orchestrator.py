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
from dataclasses import dataclass
from typing import Any, Literal

from physiclaw import paths, verdict
from physiclaw.text import read_text
from physiclaw.core.bridge import BridgeState
from physiclaw.core.calibration import PARK_PCT, Calibration, ScreenTransforms
from physiclaw.core.hardware.arm import StylusArm
from physiclaw.core.hardware.camera import Camera
from physiclaw.core.hardware.iphone import AssistiveTouch
from physiclaw.core.vision import quality
from physiclaw.core.vision.change import frames_changed
from physiclaw.core.vision.icon_detect import IconDetector
from physiclaw.core.vision.ocr import OCRReader, results_to_elements
from physiclaw.core.vision.util import (
    bbox_on_screen,
    crop_to_phone_screen,
    decode_image,
    encode_jpeg,
    format_elements,
    find_numpad_digit,
    laplacian_variance,
    phone_screen_crop_box,
    validate_bbox,
)
from physiclaw.core.vision.ui_elements import detect_ui_elements, elements_to_json
from physiclaw.core.vision.watchdog import Watchdog

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class GestureResult:
    """What a screen-mutating gesture hands to the MCP tool layer: the
    action text (screen-change verdict attached) plus the fused
    post-gesture view — annotated JPEG + element listing detected on the
    same parked after-frame the verdict used. `jpeg`/`listing` are None
    when the camera or detector hiccupped; the tool layer then falls
    back to a text-only result telling the agent to `peek`."""
    text: str
    jpeg: bytes | None = None
    listing: str | None = None


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


class ClipboardSyncError(RuntimeError):
    """AT long-press fired but the phone never fetched the queued text.

    Raised (not returned) so the `sequence` path ABORTS before any
    paste/send step runs against the phone's stale clipboard — returning
    a message here once let batches paste the previous query into an IM
    and hit send, repeatedly messaging garbage to a real person."""


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
        # Consecutive unconfirmed clipboard syncs (guarded by the gesture
        # lock). Drives the shorter retry timeout + escalating error text
        # in _send_to_clipboard; reset by any confirmed sync, or by
        # CLIPBOARD_MISS_DECAY_SECONDS elapsing since the last miss.
        self._clipboard_misses = 0
        self._clipboard_miss_at = 0.0  # time.monotonic() of the last miss
        self._ocr_reader: OCRReader | None = None
        self._icon_detector: IconDetector | None = None
        self._watchdog = Watchdog()
        # Judges every camera view for AF/AE failure (blur / blown
        # highlights); shared across peeks and gesture views so a
        # persistent rig problem escalates to "tell your user".
        self._quality = quality.QualityMonitor()
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
            "bridge": (
                self._bridge.connected if self._bridge is not None else False
            ),
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

    def _observe_quality(self, source: str, frame) -> str | None:
        """Judge a camera view for AF/AE failure; on a bad one, warn in the
        log and return the agent-facing ⚠ line for the caller to attach.

        Runs AFTER the blur-retry grabs, so a frame still failing here means
        the retry didn't recover it. Fail-open: a crash in the check never
        costs the view."""
        try:
            warning = self._quality.observe(quality.assess(frame))
        except Exception:
            log.exception("camera-quality check failed — skipping")
            return None
        if warning is not None:
            log.warning("%s: %s", source, warning)
        return warning

    # Blur-retry gate, same measurement and scale as the quality monitor's
    # verdict (laplacian_variance normalizes internally) — one number,
    # defined once.
    PEEK_BLUR_THRESHOLD = quality.BLUR_THRESHOLD

    def peek(self) -> tuple[bytes, str]:
        """Overhead camera snapshot + icon detection + OCR.

        If the first frame is too blurry, waits 2s and re-grabs once.

        Returns an annotated JPEG (icon bboxes drawn on the cropped
        camera view) and the matching element listing — same shape as
        screenshot(), but from the camera rather than the phone's own
        screenshot.
        """
        with self.locked():
            self.park()
            cropped = crop_to_phone_screen(self.camera_view(), self.transforms)
            sharpness = laplacian_variance(cropped)
            if sharpness < self.PEEK_BLUR_THRESHOLD:
                log.warning(
                    "peek: blurry frame (laplacian var=%.1f < %.0f) — retrying",
                    sharpness, self.PEEK_BLUR_THRESHOLD,
                )
                time.sleep(2.0)
                cropped = crop_to_phone_screen(self.camera_view(), self.transforms)
            listing, annotated = self._detect(cropped)
            warning = self._observe_quality("peek", cropped)
            if warning is not None:
                listing = f"{listing}\n{warning}"
            return encode_jpeg(annotated), listing

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
            listing, annotated = self._detect(frame)
            return encode_jpeg(annotated), listing

    # ─── AssistiveTouch guards ─────────────────────────────────

    def _require_no_at_overlap(self, bbox: list[float], gesture: str):
        """Raise if the bbox center would hit the AssistiveTouch button."""
        cx, cy = self.transforms.bbox_center_pct(bbox)
        if self._assistive_touch.overlaps_at(cx, cy):
            raise ValueError(
                f"{gesture} target {bbox} overlaps AssistiveTouch button — aim aside"
            )

    def _require_no_at_crossing(self, bbox: list[float], direction: str):
        """Raise if a swipe from bbox center in `direction` would cross AssistiveTouch."""
        cx, cy = self.transforms.bbox_center_pct(bbox)
        if self._assistive_touch.swipe_crosses_at(cx, cy, direction):
            raise ValueError(
                f"swipe {direction} at {bbox} crosses AssistiveTouch button — aim aside"
            )

    # ─── Gesture primitives ────────────────────────────────────

    def _tap(self, bbox: list[float]):
        """Tap at bbox center. Caller must hold the lock."""
        self._require_no_at_overlap(bbox, "tap")
        self.move_to_bbox_center(bbox)
        self._arm.tap()
        self._arm.wait_idle()

    def _double_tap(self, bbox: list[float]):
        """Double tap at bbox center. Caller must hold the lock."""
        self._require_no_at_overlap(bbox, "double_tap")
        self.move_to_bbox_center(bbox)
        self._arm.double_tap()
        self._arm.wait_idle()

    def _long_press(self, bbox: list[float]):
        """Long press at bbox center. Caller must hold the lock."""
        self._require_no_at_overlap(bbox, "long_press")
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
        start_dwell: float = 0.0,
    ):
        """Swipe from bbox center. Caller must hold the lock. `start_dwell` (s)
        anchors the touch-down at the start before sliding (see arm.swipe_to)."""
        self._require_no_at_crossing(bbox, direction)
        t = self.transforms
        ex, ey = t.swipe_end_pct(bbox, direction, self._SWIPE_DISTANCES[size])
        ex_mm, ey_mm = t.pct_to_grbl_mm(ex, ey)
        self.move_to_bbox_center(bbox)
        self._arm.swipe_to(ex_mm, ey_mm, speed, start_dwell=start_dwell)

    # ─── Screen-change verdict ────────────────────────────────

    # Post-gesture settle before the fused view is captured: added to the
    # ~1s of stylus retract + park, total ≈ 2s — enough for most page
    # transitions (Anthropic's computer-use reference uses a 2.0s delay).
    GESTURE_SETTLE_SECONDS = 1.0
    # The arm crossing the lens re-triggers autofocus; a frame captured
    # mid-hunt is blurry and poisons both the verdict and the fused view.
    # Below this Laplacian variance, wait and re-grab once. Same number
    # and scale as the quality monitor's blur verdict.
    GRAB_BLUR_THRESHOLD = quality.BLUR_THRESHOLD
    GRAB_BLUR_RETRY_SECONDS = 1.5

    # A healthy clipboard Shortcut confirms in ~1s of the long-press; the
    # generous first window covers a cold Shortcuts app. After a miss the
    # Shortcut is almost certainly broken (moved AT button, permission
    # dialog, phone off the LAN) — don't burn the full window per retry.
    CLIPBOARD_CONFIRM_SECONDS = 30.0
    CLIPBOARD_RETRY_CONFIRM_SECONDS = 8.0
    # Miss state older than this is forgotten: the server process spans
    # sessions, and a "Miss #2 in a row" escalation (or the short retry
    # window) hours after an unrelated miss would be misleading — the
    # phone may have been fixed or rebooted in between.
    CLIPBOARD_MISS_DECAY_SECONDS = 600.0

    def _grab_screen(self, settle: float = 0.0):
        """Park (clearing the arm from the lens), let the screen and
        autofocus settle for `settle`s, then capture the cropped
        phone-screen frame for the gesture diff and fused view. Parking
        first means the settle covers the AF hunt the arm's move
        triggered, so the capture is usually sharp; a still-blurry frame
        is re-grabbed once as a fallback.

        Returns `(frame, sharp)`: frame is None on any failure (the view
        is best-effort, never a reason to fail a gesture); sharp is False
        when even the retry stayed below GRAB_BLUR_THRESHOLD — such a
        frame still serves the fused view, but blur erases edges so it
        diffs as "changed everywhere": the caller must skip the verdict
        (None, the fail-open direction) rather than emit a false
        `changed` that would reset the stuck guard. Caller must hold the
        lock."""
        try:
            self.park()
            if settle:
                time.sleep(settle)
            frame = crop_to_phone_screen(self.camera_view(), self.transforms)
            if laplacian_variance(frame) < self.GRAB_BLUR_THRESHOLD:
                log.debug("gesture frame blurry — waiting for autofocus")
                time.sleep(self.GRAB_BLUR_RETRY_SECONDS)
                frame = crop_to_phone_screen(self.camera_view(), self.transforms)
                if laplacian_variance(frame) < self.GRAB_BLUR_THRESHOLD:
                    log.warning("gesture frame still blurry after retry — verdict withheld")
                    return frame, False
            return frame, True
        except Exception:
            log.warning("screen frame grab failed", exc_info=True)
            return None, False

    def _with_view(self, action) -> GestureResult:
        """Run a gesture body bracketed by before/after frames; return
        the action text with the screen-change verdict attached PLUS the
        fused post-gesture view (annotated JPEG + listing) detected on
        the same after-frame. One capture serves both:

          - the verdict is the agent's only evidence for SILENT REFUSALS
            — a toast vanishes long before the arm parks, so "the screen
            looks exactly the same" distinguishes a refused tap from a
            landed one;
          - the view replaces the act-then-peek turn pair — the agent
            reads the new screen from the gesture result itself.

        Caller must hold the lock. The before-grab needs no settle (the
        arm was already parked from the previous op, screen static); the
        after-grab settles so the transition completes and AF recovers
        from the gesture's arm move before capture.
        """
        before, before_sharp = self._grab_screen()
        result = action()
        after, after_sharp = self._grab_screen(settle=self.GESTURE_SETTLE_SECONDS)
        changed = None
        jpeg = listing = None
        if after is not None:
            # Verdict only from two sharp frames — a blurry side diffs as
            # "changed everywhere" (false `changed`, the harmful
            # direction). The view below is best-effort either way.
            if before is not None and before_sharp and after_sharp:
                try:
                    changed = frames_changed(before, after)
                except Exception:
                    log.debug("screen-verdict diff failed", exc_info=True)
            try:
                listing, annotated = self._detect(after)
                jpeg = encode_jpeg(annotated)
            except Exception:
                log.warning("post-gesture view failed", exc_info=True)
            warning = self._observe_quality("gesture view", after)
        else:
            warning = None
        text = verdict.attach(result, changed)
        if warning is not None:
            if listing is not None:
                listing = f"{listing}\n{warning}"
            else:
                # Detection failed, so there's no listing to carry the line —
                # ride the action text: a blurry/blown frame is a plausible
                # CAUSE of the failed view, and the agent needs to hear it
                # before blindly re-peeking.
                text = f"{text}\n{warning}"
        return GestureResult(text=text, jpeg=jpeg, listing=listing)

    def _run_gesture(self, gesture, result: str) -> "GestureResult":
        """Lock, run `gesture` bracketed by verdict/view frames, and
        return `result` fused with the post-gesture view — the shared
        shape of every public gesture below."""

        def act() -> str:
            gesture()
            return result

        with self.locked():
            return self._with_view(act)

    # ─── Public gestures (with lock) ─────────────────────────

    def tap(self, bbox: list[float]) -> "GestureResult":
        """Single tap at the center of a bbox."""
        validate_bbox(bbox)
        return self._run_gesture(
            lambda: self._tap(bbox), f"Tapped at bbox {bbox}"
        )

    def double_tap(self, bbox: list[float]) -> "GestureResult":
        """Double tap at the center of a bbox."""
        validate_bbox(bbox)
        return self._run_gesture(
            lambda: self._double_tap(bbox), f"Double tapped at bbox {bbox}"
        )

    def long_press(self, bbox: list[float]) -> "GestureResult":
        """Long press (~1.2s) at the center of a bbox."""
        validate_bbox(bbox)
        return self._run_gesture(
            lambda: self._long_press(bbox), f"Long pressed at bbox {bbox}"
        )

    def _validate_swipe(self, bbox, direction, size, speed):
        """Raises ValueError if any swipe arg is out of range."""
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

    def swipe(
        self,
        bbox: list[float],
        direction: Literal["up", "down", "left", "right"],
        size: Literal["s", "m", "l", "xl", "xxl"] = "m",
        speed: Literal["slow", "medium", "fast"] = "medium",
    ) -> "GestureResult":
        """Swipe from the bbox center in `direction` by `size` screen fraction.

        `no visible change` in the verdict after a scroll swipe = end of
        the list."""
        self._validate_swipe(bbox, direction, size, speed)
        return self._run_gesture(
            lambda: self._swipe(bbox, direction, size, speed),
            f"Swiped {direction} {size} at bbox {bbox}",
        )

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
        the screen) could carry marker-like text (`physiclaw.verdict`)."""
        self._require_assistive_touch()
        if (
            self._clipboard_misses
            and time.monotonic() - self._clipboard_miss_at
            > self.CLIPBOARD_MISS_DECAY_SECONDS
        ):
            self._clipboard_misses = 0
        self._bridge.send_text(text)
        self._assistive_touch.long_press(self._arm, self.transforms.pct_to_grbl)
        timeout = (
            self.CLIPBOARD_CONFIRM_SECONDS
            if self._clipboard_misses == 0
            else self.CLIPBOARD_RETRY_CONFIRM_SECONDS
        )
        if self._bridge.wait_clipboard(timeout=timeout):
            self._clipboard_misses = 0
            return f"Copied {len(text)} chars to phone clipboard"
        # Un-queue the text so a LATE Shortcut run can't fetch it after
        # this timeout — the error below promises the phone clipboard
        # still holds the previous content; keep that true, not racy.
        self._bridge.clear_text()
        self._clipboard_misses += 1
        self._clipboard_miss_at = time.monotonic()
        msg = (
            "clipboard sync FAILED — AssistiveTouch long-pressed but the "
            "phone never fetched the text; its clipboard still holds the "
            "PREVIOUS content, so do NOT paste"
        )
        if self._clipboard_misses >= 2:
            msg += (
                f". Miss #{self._clipboard_misses} in a row — the clipboard "
                "Shortcut is broken (moved AssistiveTouch button, a "
                "Shortcuts permission dialog, or the phone lost the LAN). "
                "STOP retrying this path: type with the keyboard instead, "
                "or report the problem to your user"
            )
        raise ClipboardSyncError(msg)

    def send_to_clipboard(self, text: str) -> str:
        """Copy text to the phone's clipboard via AssistiveTouch long-press."""
        with self.locked():
            return self._send_to_clipboard(text)

    def _run_step(self, tool: str, arg) -> str:
        """Dispatch one sequence step. Caller must hold the lock.

        Used only by `sequence` — public gesture methods call their `_tap`/
        `_swipe`/etc. directly.
        """
        if tool == "tap":
            validate_bbox(arg)
            self._tap(arg)
            return f"Tapped at bbox {arg}"
        if tool == "double_tap":
            validate_bbox(arg)
            self._double_tap(arg)
            return f"Double tapped at bbox {arg}"
        if tool == "long_press":
            validate_bbox(arg)
            self._long_press(arg)
            return f"Long pressed at bbox {arg}"
        if tool == "swipe":
            # Error messages here avoid repr-ing the raw arg — they join
            # the batch action text the engine scans for the verdict
            # marker (see _send_to_clipboard).
            if not isinstance(arg, dict) or "bbox" not in arg or "direction" not in arg:
                raise ValueError(
                    f"swipe arg needs a dict with bbox + direction, got "
                    f"{type(arg).__name__}"
                )
            bbox, direction = arg["bbox"], arg["direction"]
            size, speed = arg.get("size", "m"), arg.get("speed", "medium")
            self._validate_swipe(bbox, direction, size, speed)
            self._swipe(bbox, direction, size, speed)
            return f"Swiped {direction} {size} at bbox {bbox}"
        if tool == "send_to_clipboard":
            if not isinstance(arg, str):
                raise ValueError(
                    f"send_to_clipboard arg must be a string, got "
                    f"{type(arg).__name__}"
                )
            return self._send_to_clipboard(arg)
        raise ValueError(f"tool {tool!r} not allowed in sequence")

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
                tool = s["tool_name"]
                try:
                    result = self._run_step(tool, s.get("arg"))
                    lines.append(f"{i} {tool} ok — {result}")
                except Exception as e:
                    lines.append(f"{i} {tool} FAIL ({e})")
                    break
            return "\n".join(lines)

        with self.locked():
            return self._with_view(act)

    def home_screen(self) -> "GestureResult":
        """Go to the home screen via bottom-edge swipe up."""
        return self._run_gesture(
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
        return self._run_gesture(
            lambda: self._swipe(
                [0.0, 0.4, 0.01, 0.6], "right", "xxl", speed="fast",
                start_dwell=self.BACK_EDGE_DWELL_SECONDS,
            ),
            "Went back",
        )

    def force_quit(self) -> "GestureResult":
        """Force-quit current app via the iOS app-switcher gesture.

        Step 1's swipe is short on purpose — a longer upward swipe from
        the bottom edge would go home instead of opening the switcher.
        """

        def gestures() -> None:
            self._swipe([0.4, 0.96, 0.6, 0.98], "up", "s", speed="slow")
            self._swipe([0.4, 0.45, 0.6, 0.55], "left", "m", speed="medium")
            self._swipe([0.4, 0.70, 0.6, 0.80], "up", "xl", speed="fast")
            self._tap([0.4, 0.92, 0.6, 0.96])

        return self._run_gesture(gestures, "Force-quit current app")

    def unlock_phone(self) -> "GestureResult":
        """Unlock the phone: wake → swipe up → wait for Face ID to fail → enter passcode.

        Fully mechanical — no AI. OCR finds digit "1" on the passcode
        screen, then taps it six times. Passcode is hardcoded to 111111 —
        a dedicated tool-phone passcode, not the user's real password.
        """

        def act() -> str:
            self._tap([0.4, 0.4, 0.6, 0.6])
            self._swipe([0.4, 0.96, 0.6, 0.98], "up", "l", speed="fast")
            self.park()
            time.sleep(4)  # Face ID starts

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

        with self.locked():
            return self._with_view(act)

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
