"""
PhysiClaw orchestrator — the agent's action surface.

A thin facade that composes the two orchestration halves and keeps only
the tool operations the MCP layer exposes:

- ``rig`` (:class:`HardwareRig`) — devices, calibration, the busy lock,
  lifecycle, and primitive positioning moves;
- ``perception`` (:class:`Perception`) — detectors, frame acquisition,
  the watchdog, and the exposure tune.

Everything gesture-shaped funnels through the single validated dispatch
(``_execute``) under the single lock + observation bracket
(``_observed``). Consumers that don't gesture (calibration and
hardware-setup handlers) take the rig directly and never see this
surface. Image processing lives in physiclaw.core.vision — the
orchestrator never touches pixels.
"""

import threading
from typing import TYPE_CHECKING, assert_never, cast

if TYPE_CHECKING:
    import numpy as np

from physiclaw.common.gesture_vocab import STEP_ARG, STEP_TOOL
from physiclaw.core.orchestration import gestures, unlock
from physiclaw.core.orchestration.clipboard import (
    ClipboardSyncError,
    ClipboardSyncState,
)
from physiclaw.core.orchestration.observation import GestureObserver, GestureResult
from physiclaw.core.orchestration.perception import Perception
from physiclaw.core.orchestration.rig import HardwareRig
from physiclaw.core.vision.util import (
    decode_image,
    encode_view_jpeg,
    validate_bbox,
)


class PhysiClaw:
    """Facade over the rig + perception: the tool operations MCP exposes."""

    def __init__(self):
        self.rig = HardwareRig()
        self.perception = Perception(self.rig)
        # Accessor lambdas, not the objects: both are replaced after
        # construction (calibration, tests), and the validator must see
        # the current ones.
        self._validator = gestures.GestureValidator(
            assistive_touch=lambda: self.rig.assistive_touch,
            transforms=lambda: self.rig.require_transforms(),
        )
        self._clipboard = ClipboardSyncState()
        # Long-lived so the located "1" bbox caches across unlocks.
        self._unlock = unlock.PhoneUnlock()
        self._become_ready_lock = threading.Lock()  # see become_ready
        # Lambdas, not bound methods: they look `grab`/`detect` up on the
        # perception object at call time, so a replaced (or test-patched)
        # implementation is honored for the observer too.
        self._observer = GestureObserver(
            park=self.rig.park,
            grab=lambda: self.perception.cropped_view(),
            detect=lambda frame: self.perception.detect(frame),
            on_quality=lambda report, streak: self.perception.on_quality_report(
                report, streak
            ),
            # Synchronous: grab paths run inside rig.locked(), so the
            # inline fix must not re-acquire — tune_now assumes the lock.
            fix_exposure=lambda: self.perception.tune_now(),
            needs_fix=lambda report: self.perception.needs_inline_fix(report),
        )

    # ─── Lifecycle ─────────────────────────────────────────────

    def become_ready(self):
        """Settle the camera (exposure tune), THEN flip the
        ready flag — the single ready-flipping path. Ordering matters:
        the agent runtime polls `ready` and peeks immediately, so
        flipping first would let the first view of a session ship from
        an unsettled camera. `settle_camera` is fail-open; the finally
        is the backstop that keeps "ready always flips" true even if
        that claim ever grows a hole — a failed settle must cost the
        settle, never readiness.

        Single-flight: a duplicate call (wizard page refreshed during
        the settle window re-fires /api/ready) waits for the in-flight
        settle instead of busy-skipping its own and flipping ready on a
        still-unsettled camera; once ready, later calls no-op.
        Callers: warm-start, /api/ready."""
        with self._become_ready_lock:
            if self.rig.ready:
                return
            try:
                self.perception.settle_camera()
            finally:
                self.rig.mark_ready()

    def shutdown(self):
        """Release the coil, park, and close every device handle.
        Kept on the facade so teardown callers (`server/app.py`, atexit)
        hold one object; the work is `HardwareRig.shutdown`."""
        self.rig.shutdown()

    # ─── Screen views ─────────────────────────────────────────

    def peek(self) -> tuple[bytes, str]:
        """Overhead camera snapshot + icon detection + OCR.

        A blurry or blown first frame waits 2s and re-grabs once; a
        persistently blown one — or the first bright reference after a
        deferred cold-start tune — gets an inline exposure re-tune
        before the view ships (``GestureObserver.peek_frame`` owns that
        acquisition policy).

        Returns an annotated JPEG (icon bboxes drawn on the cropped
        camera view) and the matching element listing — same shape as
        screenshot(), but from the camera rather than the phone's own
        screenshot.
        """
        with self.rig.locked():
            cropped, report, retuned = self._observer.peek_frame()
            listing, annotated = self.perception.detect(cropped)
            warning = self._observer.observe_quality("peek", report, retuned=retuned)
            if warning is not None:
                listing = f"{listing}\n{warning}"
            # detect() types the annotated frame as `object`; it is always
            # the bbox-drawn ndarray from detect_ui_elements.
            return encode_view_jpeg(cast("np.ndarray", annotated)), listing

    def screenshot(self) -> tuple[bytes, str]:
        """Pixel-perfect phone screenshot + icon detection + OCR.

        Returns an annotated JPEG (icon bboxes drawn) and the matching
        element listing — same shape as peek(), but sourced from the
        phone's own screenshot instead of the camera.
        """
        with self.rig.locked():
            data = self.rig.take_screenshot(timeout=60.0)
            if data is None:
                hint = self.rig.require_bridge().connectivity_hint()
                raise TimeoutError(
                    f"Screenshot upload timed out — {hint or 'check the iOS Shortcut'}"
                )

            frame = decode_image(data)
            # Detection runs on the native-resolution screenshot;
            # encode_view_jpeg caps the image the agent sees.
            listing, annotated = self.perception.detect(frame)
            # detect() types the annotated frame as `object`; it is always
            # the bbox-drawn ndarray from detect_ui_elements.
            return encode_view_jpeg(cast("np.ndarray", annotated)), listing

    # ─── Gesture primitives ────────────────────────────────────

    def _press(self, bbox: list[float], gesture: str):
        """Shared body of the press gestures: AT guard, move to the bbox
        center, fire the arm method named `gesture`, wait idle. The
        gesture vocabulary and StylusArm deliberately share the names
        `tap` / `double_tap` / `long_press`. Caller must hold the lock."""
        self._validator.require_no_at_overlap(bbox, gesture)
        self.rig.move_to_bbox_center(bbox)
        arm = self.rig.require_arm()
        getattr(arm, gesture)()
        arm.wait_idle()

    def _swipe(
        self,
        bbox: list[float],
        direction: gestures.Direction,
        size: gestures.Size = "m",
        speed: gestures.Speed = "medium",
        start_dwell: float = 0.0,
        end_dwell: float = 0.0,
    ):
        """Swipe from bbox center: AT-crossing guard, then the rig's swipe
        primitive — resolving the size keyword to a screen-fraction
        distance is the facade's only geometry job here. `start_dwell` /
        `end_dwell` (s) anchor the touch-down / hold the endpoint (see
        arm.swipe_to). Caller must hold the lock."""
        self._validator.require_no_at_crossing(bbox, direction, size)
        self.rig.swipe_from_bbox(
            bbox,
            direction,
            gestures.SWIPE_DISTANCES[size],
            speed,
            start_dwell=start_dwell,
            end_dwell=end_dwell,
        )

    # ─── Screen-change verdict ────────────────────────────────

    def _observed(self, act) -> "GestureResult":
        """Lock, run `act` (which returns its action text) bracketed by
        the observer's verdict/view frames — the single lock +
        observation bracket every mutating tool goes through."""
        with self.rig.locked():
            return self._observer.with_view(act)

    def _execute(self, step: gestures.Gesture) -> str:
        """Run one typed, already-validated gesture and compose its
        action text — the single dispatch point shared by the public
        gesture methods, `sequence` steps, and the macro recipes.
        Caller must hold the lock."""
        self.rig.assert_locked()
        match step:
            case gestures.Tap(bbox):
                self._press(bbox, "tap")
                return f"Tapped at bbox {bbox}"
            case gestures.DoubleTap(bbox):
                self._press(bbox, "double_tap")
                return f"Double tapped at bbox {bbox}"
            case gestures.LongPress(bbox):
                self._press(bbox, "long_press")
                return f"Long pressed at bbox {bbox}"
            case gestures.Swipe(bbox, direction, size, speed, start_dwell, end_dwell):
                self._swipe(bbox, direction, size, speed, start_dwell, end_dwell)
                return f"Swiped {direction} {size} at bbox {bbox}"
            case gestures.SendToClipboard(text):
                return self._send_to_clipboard(text)
            case _:
                assert_never(step)

    def _run_gesture(self, step: gestures.Gesture) -> "GestureResult":
        """Run one typed gesture under the observation bracket."""
        return self._observed(lambda: self._execute(step))

    def _run_macro(
        self, steps: tuple[gestures.Gesture, ...], result: str
    ) -> "GestureResult":
        """Lock, replay a recipe's steps through the shared dispatch
        bracketed by verdict/view frames, and return the fixed `result`
        text fused with the post-gesture view — the shared shape of the
        macro gestures (home_screen / go_back / force_quit)."""

        def act() -> str:
            for step in steps:
                self._execute(step)
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
        # Require BEFORE queueing: text queued and then raised on a missing
        # AT setup would sit on the bridge for a late Shortcut run to fetch.
        self.rig.require_assistive_touch()
        timeout = self._clipboard.begin()
        # The queue/long-press/wait/un-queue choreography is the rig's
        # (`sync_clipboard`); the facade owns only the retry/miss policy.
        if self.rig.sync_clipboard(text, timeout):
            self._clipboard.confirm()
            return f"Copied {len(text)} chars to phone clipboard"
        raise ClipboardSyncError(
            self._clipboard.record_miss(self.rig.require_bridge().connectivity_hint())
        )

    def send_to_clipboard(self, text: str) -> str:
        """Copy text to the phone's clipboard via AssistiveTouch long-press."""
        with self.rig.locked():
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
                # Read the tool name INSIDE the try: the schema is only
                # list[dict], so a malformed step (missing "tool_name",
                # non-dict junk) must surface as a per-step FAIL line
                # (keeping the results of steps already run + the fused
                # view), not a bare error that discards the whole batch's
                # report. The "?" seed covers the case where the read
                # itself is what failed.
                tool = "?"
                try:
                    tool = s[STEP_TOOL]
                    result = self._run_step(tool, s.get(STEP_ARG))
                    lines.append(f"{i} {tool} ok — {result}")
                except Exception as e:
                    lines.append(f"{i} {tool} FAIL ({e})")
                    break
            return "\n".join(lines)

        return self._observed(act)

    # ─── Macro gestures (iOS recipes in gestures.py) ─────────

    def home_screen(self) -> "GestureResult":
        """Go to the home screen via bottom-edge swipe up."""
        return self._run_macro(gestures.HOME_SCREEN, "Went to home screen")

    def go_back(self) -> "GestureResult":
        """Go back one screen via left-edge swipe right.

        `no visible change` in the verdict = the edge swipe didn't pop
        (modal, root tab, image viewer) — the cue to try another exit."""
        return self._run_macro(gestures.GO_BACK, "Went back")

    def force_quit(self) -> "GestureResult":
        """Force-quit current app via the iOS app-switcher recipe
        (gestures.FORCE_QUIT): open with swipe-up-and-hold, drag the
        clipped current-app card to center, fling it, dismiss the
        switcher. The attached view shows where you landed."""
        return self._run_macro(gestures.FORCE_QUIT, "Force-quit current app")

    def unlock_phone(self) -> "GestureResult":
        """Unlock the phone with passcode 111111 — wake, swipe up, find the
        keypad "1", tap it six times.

        Fully mechanical (no AI) with a dedicated throwaway passcode. The
        imperative timing dance (Face-ID wait, keypad OCR poll, stale-keypad
        re-arm) lives in ``unlock.PhoneUnlock``, which caches the located
        "1" so later unlocks skip OCR; the facade only supplies the lock +
        observation bracket and the two collaborators it needs. Single-shot
        on purpose: no pixel verify — the attached view shows whether the
        phone unlocked, and the AGENT retries if not.
        """
        return self._observed(
            lambda: self._unlock.unlock_phone(
                self._execute, self.perception, self.rig.park
            )
        )
