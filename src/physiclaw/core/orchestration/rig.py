"""HardwareRig — device ownership, the busy lock, and lifecycle.

The rig owns the physical stack (arm, camera, AssistiveTouch, bridge,
calibration state) and everything that guards it: connect/disconnect,
the single hardware lock, parking, warm-start origin re-pinning, ready
state, and teardown. It performs positioning moves but no gestures —
taps, swipes, and their observation bracket live in the orchestrator.

Consumers that only need devices (the calibration and hardware-setup
HTTP handlers) take a HardwareRig directly, so they never see the
agent-facing gesture surface.
"""

import json
import logging
import threading
from contextlib import contextmanager

from physiclaw.common import paths
from physiclaw.common.text import read_text
from physiclaw.core.bridge import BridgeState
from physiclaw.core.calibration import PARK_PCT, Calibration, ScreenTransforms
from physiclaw.core.hardware.arm import StylusArm
from physiclaw.core.hardware.camera import Camera
from physiclaw.core.hardware.iphone import AssistiveTouch

log = logging.getLogger(__name__)

# How long a TOOL gesture waits for the hardware lock before reporting
# busy. Background work (watchdog frames, exposure retune) holds the
# lock for O(seconds); a session's tap must outwait that, not lose to it.
TOOL_LOCK_WAIT_SECONDS = 10.0


def _layout_learned() -> bool:
    """Read the first-run layout's `learned` marker straight from its JSON.

    The agent-side `agent.layout` module writes the marker (it owns the
    page/field schema); core only reads it, so the rig can report
    first-run progress without importing the agent package."""
    p = paths.screen_layout_json()
    if not p.exists():
        return False
    try:
        data = json.loads(read_text(p))
    except (OSError, ValueError):  # ValueError covers JSON + UTF-8 decode errors
        return False
    # "layout_learned" is written by agent-side layout.record() once
    # every box is captured; core only reads it (keep in sync with that writer).
    return bool(isinstance(data, dict) and data.get("layout_learned"))


class HardwareRig:
    """Devices + calibration + the busy lock.

    Construction is instant (no hardware). Call connect_arm() and
    connect_camera() to connect hardware. Calibration is handled
    by the /setup skill via HTTP endpoints.
    """

    def __init__(self):
        self._arm: StylusArm | None = None
        self._cam: Camera | None = None
        self.calibration: Calibration = Calibration()
        self._lock = threading.Lock()
        self._lock_owner: int | None = None  # thread ident; see assert_locked
        self._assistive_touch = AssistiveTouch()
        self._bridge: BridgeState | None = None
        self._ready = False  # set True only after /setup finishes its last step
        # Does the GRBL work frame match `calibration`'s affine? See park().
        self._origin_pinned = False

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
        """Flip the ready flag. Kept pure (tests pin it).

        Don't call this directly from a new path — go through
        ``PhysiClaw.become_ready()``, which settles the camera FIRST so
        the agent can never observe ``ready`` on an unsettled camera."""
        self._ready = True

    def unmark_ready(self) -> None:
        """Drop the ready flag so the next ``become_ready`` re-settles.
        Called when the camera is replaced mid-session (a live-server
        setup re-run): the fresh Camera boots with default exposure (its
        lens is already pinned from the bundle seed), but
        ``become_ready`` early-returns while ``ready`` is set — without
        this it would skip the re-settle for the rest of the process,
        leaving the first views of the new camera mis-exposed."""
        self._ready = False

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
        at_pos = self._assistive_touch.at_screen
        if self._assistive_touch.ready and at_pos is not None:  # ready ⇒ at_pos
            sx, sy = at_pos
            steps["assistive_touch"] = f"({sx:.3f}, {sy:.3f})"
        return {
            "arm": self._arm is not None,
            "camera": self._cam is not None,
            "bridge": (self._bridge.connected if self._bridge is not None else False),
            "steps": steps,
            "calibrated": self.hardware_ready,
            "ready": self.ready,
            "origin_pinned": self._origin_pinned,
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
        # A loaded affine with an un-pinned frame (a mid-session arm
        # reconnect that never re-pinned) would land every gesture off by
        # the park vector — fail loudly instead of tapping a stale frame.
        if self.calibration.pct_to_grbl is not None and not self._origin_pinned:
            raise RuntimeError(
                "Arm work frame is not pinned to the calibration — restart "
                "the server (--hot-start with the arm at its park spot), or "
                "recalibrate."
            )

    # ─── Concurrency ──────────────────────────────────────────

    def acquire(self, wait_seconds: float = 0):
        """Mark hardware as busy. Raises if not acquired within
        `wait_seconds` (default: immediately). Background work keeps the
        immediate form and fails open on contention; the TOOL path waits
        a bounded moment instead — a session gesture must not die because
        the exposure retune held the lock for two seconds (field-measured:
        that race killed a conductor walk mid-gate)."""
        # timeout=0 IS the immediate non-blocking poll — one call form
        # covers every wait.
        if not self._lock.acquire(timeout=wait_seconds):
            raise RuntimeError(
                "PhysiClaw is busy — wait for the current operation to finish, then retry."
            )
        self._lock_owner = threading.get_ident()

    def release(self):
        """Mark hardware as idle."""
        self._lock_owner = None
        self._lock.release()

    def _held_by_this_thread(self) -> bool:
        """OWNERSHIP, not mere held-ness — another thread legitimately
        holding the lock is exactly when an unserialized caller would
        interleave G-code with a gesture in flight, so held-by-anyone
        would pass right when the distinction matters most."""
        return self._lock_owner == threading.get_ident()

    def assert_locked(self):
        """Loud guard for caller-must-hold-the-lock methods: raise instead
        of silently touching hardware unserialized."""
        if not self._held_by_this_thread():
            raise RuntimeError(
                "hardware lock not held by this thread — wrap the call in "
                "rig.locked() or acquire()"
            )

    @contextmanager
    def try_locked(self, timeout: float):
        """Bounded, non-raising lock bracket for teardown-grade callers
        that must proceed either way. Yields whether the lock was won;
        already-owning threads pass straight through. Owner bookkeeping
        stays here, beside acquire/release, so the protocol has one home."""
        if self._held_by_this_thread():
            yield True
            return
        try:
            got = self._lock.acquire(timeout=timeout)
        except KeyboardInterrupt:
            # A second Ctrl-C during the bounded wait means "stop waiting",
            # not "abandon teardown" — this bracket exists for callers that
            # must proceed either way.
            got = False
        if got:
            self._lock_owner = threading.get_ident()
        try:
            yield got
        finally:
            if got:
                self.release()

    @contextmanager
    def locked(self, wait_seconds: float = 0):
        """Hold the hardware lock for the block, releasing on any exit.

        The bare acquire→release bracket with no parking and no ready
        gate — the shared primitive under ``engaged()`` and under the
        connect/calibration/warm-start handlers that must serialize
        hardware work *before* the rig is ready (so ``engaged()``'s
        ``require_hardware()`` gate would wrongly reject them). Each of
        those callers layers its own parking policy on top, because the
        park timing genuinely differs (on exit, on failure, or not at
        all); this owns only the part that never differs. Raises the busy
        error if the lock is already held — callers that must fail open on
        contention (the background settle) use ``acquire()`` directly and
        catch it."""
        self.acquire(wait_seconds)
        try:
            yield
        finally:
            self.release()

    @contextmanager
    def engaged(self, wait_seconds: float = 0):
        """Check hardware, acquire lock, auto-park on exit, then release.
        Tool entries pass TOOL_LOCK_WAIT_SECONDS to wait out short
        background contention; fail-open callers (the watchdog poll)
        keep the immediate default and skip on busy."""
        self.require_hardware()
        with self.locked(wait_seconds=wait_seconds):
            try:
                yield
            finally:
                try:
                    self.park()
                except Exception:
                    # Best-effort like shutdown's _safe, but never silent: a
                    # persistently failing inter-gesture park leaves the tip
                    # over the glass while gestures keep running.
                    log.exception("engaged(): auto-park failed")

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
        arm = StylusArm()
        try:
            arm.setup()
        except BaseException:
            # BaseException, matching StylusArm.__init__: a Ctrl-C mid-setup
            # must still close the port (releasing the coil via the DTR
            # drop). A half-configured arm (serial open DTR-reset the board,
            # so no G92 work origin, no G21/G90, unconfigured solenoid) must
            # not stay attached: hardware_ready would report it connected and
            # gestures would land machine-origin-relative — wrong-place taps.
            try:
                arm.close()
            except Exception:
                log.exception("connect_arm: close after failed setup also failed")
            raise
        self._arm = arm
        self._origin_pinned = False  # arm.setup() re-zeroed the work frame
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

        The lens position is a calibration constant — see
        ``hardware/focus.py``. Seeding it into the Camera pins the lens
        from the very first open; empty calibration (fresh setup) leaves
        AF live until the mapping step pins it.
        """
        if self._cam is not None:
            self._cam.close()
            self._cam = None
        self._cam = Camera(index, focus_value=self.calibration.cam_focus)
        if self.calibration.cam_rotation is not None:
            self._cam.rotation = self.calibration.cam_rotation
        # A fresh Camera starts with default exposure — force the next
        # become_ready to re-settle it (see unmark_ready).
        self.unmark_ready()
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
        # No camera → not ready; a later reconnect re-settles via become_ready.
        self.unmark_ready()
        log.info("Camera disconnected")
        return True

    def restore_park_origin(self) -> bool:
        """Re-pin the GRBL origin assuming the tip rests at the off-screen
        park spot. Warm-start's counterpart to ``_park_for_teardown``.

        ``arm.setup()`` on reconnect issues ``G92 X0 Y0``, declaring the
        arm's *current* physical position to be GRBL ``(0, 0)``. That's only
        the calibrated origin if the tip is sitting there — but clean
        shutdown (and every inter-action ``engaged()`` park) leaves it at the
        park spot instead. So re-declare the current position as the park
        coordinate from the loaded bundle, restoring the affine's frame;
        otherwise every subsequent tap is offset by the park vector.

        Returns False (no-op) if the arm or `pct_to_grbl` isn't ready.
        On success the frame counts as pinned again — `park()` and its
        callers may trust absolute moves from here on.
        """
        if self._arm is None:
            return False
        park_xy = self.calibration.pct_to_grbl_mm(*PARK_PCT)
        if park_xy is None:
            return False
        self._arm.set_work_position(*park_xy)
        self._origin_pinned = True  # frame and affine agree again
        log.info("Re-pinned GRBL origin from park spot %s", PARK_PCT)
        return True

    @property
    def origin_pinned(self) -> bool:
        """Whether the GRBL work frame matches `calibration`'s affine."""
        return self._origin_pinned

    def install_arm_calibration(self, pct_to_grbl) -> None:
        """The one way a probe-CONFIRMED affine is installed: sets it,
        propagates the direction mapping to the arm, and pins the frame —
        `calibrate_arm` ends `set_origin`'d at screen center, the exact
        frame the fresh affine is rebased to, so the two agree by
        construction. (`borrow_arm_calibration` is the provisional
        sibling; `uninstall_arm_calibration` the undo.)"""
        self.calibration.pct_to_grbl = pct_to_grbl
        self._apply_bundle_to_arm()
        self._origin_pinned = True

    def borrow_arm_calibration(self, pct_to_grbl) -> None:
        """`install_arm_calibration`'s provisional sibling: seats a SAVED
        affine so a from_park calibration can reach the screen. No pin
        (the caller earns that from `restore_park_origin` at the physical
        park spot) and no direction-mapping propagation (the arm keeps
        what connect applied) — nothing here claims a probe confirmed
        the affine. Undone by `uninstall_arm_calibration` if the probe
        then fails."""
        self.calibration.pct_to_grbl = pct_to_grbl

    def uninstall_arm_calibration(self) -> None:
        """The borrow/install undo: clears the affine and the pin
        together (a pin without an affine models nothing). Callers park
        BEFORE this — park needs both."""
        self.calibration.pct_to_grbl = None
        self._origin_pinned = False

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
    def arm(self) -> StylusArm | None:
        """The connected arm, or None — honest about pre-setup state.
        Callers that must actuate use ``require_arm()``."""
        return self._arm

    @property
    def cam(self) -> Camera | None:
        """The connected camera, or None. See ``require_cam()``."""
        return self._cam

    @property
    def bridge(self) -> BridgeState | None:
        """The attached server-side bridge, or None before assembly."""
        return self._bridge

    def require_bridge(self) -> BridgeState:
        """The attached bridge, or raise — for callers that must reach the
        phone's /bridge page."""
        if self._bridge is None:
            raise RuntimeError("Bridge not attached — server assembly incomplete")
        return self._bridge

    def require_arm(self) -> StylusArm:
        """The connected arm, or raise — for callers that must actuate."""
        if self._arm is None:
            raise RuntimeError("Arm not connected. Run /setup to connect it.")
        return self._arm

    def require_cam(self) -> Camera:
        """The connected camera, or raise — for callers that must see."""
        if self._cam is None:
            raise RuntimeError("Camera not connected. Run /setup to connect it.")
        return self._cam

    @property
    def transforms(self) -> ScreenTransforms | None:
        return self.calibration.transforms()

    def require_transforms(self) -> ScreenTransforms:
        """The calibrated transforms, or raise — for callers that must map
        screen pct to hardware coordinates."""
        t = self.calibration.transforms()
        if t is None:
            raise RuntimeError("Screen calibration not done")
        return t

    @property
    def assistive_touch(self) -> AssistiveTouch:
        return self._assistive_touch

    def require_assistive_touch(self) -> AssistiveTouch:
        """The calibrated AssistiveTouch, or raise — for callers about to
        drive one of its iOS Shortcuts."""
        if not self._assistive_touch.ready:
            raise RuntimeError("AssistiveTouch not calibrated — run /setup first")
        return self._assistive_touch

    # ─── AssistiveTouch operations ────────────────────────────
    # AT gestures are rig plumbing, not agent gestures: AssistiveTouch
    # needs the arm, bridge, and transforms — all rig-owned — so the
    # wiring lives here and callers never unpack it. Caller must hold
    # the hardware lock.

    def take_screenshot(self, timeout: float = 60.0) -> bytes | None:
        """Trigger the iOS screenshot + upload Shortcuts via AssistiveTouch;
        return the uploaded image bytes, or None on upload timeout."""
        self.assert_locked()
        at = self.require_assistive_touch()
        return at.take_screenshot(
            self.require_arm(),
            self.require_bridge(),
            self.require_transforms().pct_to_grbl,
            timeout=timeout,
        )

    def at_long_press(self) -> None:
        """Long-press the AssistiveTouch button (fires the clipboard Shortcut)."""
        self.assert_locked()
        at = self.require_assistive_touch()
        at.long_press(self.require_arm(), self.require_transforms().pct_to_grbl)

    def sync_clipboard(self, text: str, timeout: float) -> bool:
        """Queue `text` on the bridge, long-press AT (fires the clipboard
        Shortcut), wait for the phone's fetch confirmation. Returns True
        on confirm. On timeout the text is retired via
        ``BridgeState.expire_text()``, which is atomic with the Shortcut's
        fetch: a late run either already got the text (counted here as a
        late success) or can never get it afterwards — a plain
        ``clear_text()`` left a gap where the phone received the text
        while we returned False, so callers' "the phone clipboard still
        holds the previous content" stays true, not racy. Caller must
        hold the lock; the retry/miss policy lives in the orchestrator's
        ClipboardSyncState."""
        self.assert_locked()
        bridge = self.require_bridge()
        bridge.send_text(text)
        try:
            self.at_long_press()
        except Exception:
            # Arm failure after the text is queued: retire it before
            # re-raising, or a later Shortcut run / /bridge tap could
            # fetch THIS run's text — the residue the expire path exists
            # to prevent, and a violation of ClipboardSyncError's "the
            # phone clipboard still holds the previous content" contract.
            bridge.expire_text()
            raise
        if bridge.wait_clipboard(timeout=timeout):
            return True
        return bridge.expire_text()

    # ─── Primitive movements ─────────────────────────────────

    def park(self):
        """Move stylus off-screen to ``PARK_PCT`` — left of the screen,
        slightly above top edge.

        Defensive: no-ops if the arm isn't connected, `pct_to_grbl` isn't
        set yet, or the work frame isn't pinned to that affine. The first
        two make parking safe between calibration steps; the third is the
        crash guard: ``arm.setup()`` zeroes the frame at the resting
        position on every connect, and an absolute park move in that
        un-pinned frame travels the full park vector PAST the resting
        spot — into the frame top. Un-pinned means the frame's relation
        to the affine is UNKNOWN, so no absolute move can be modeled —
        refusing to move is the only safe park. Caller must hold the
        hardware lock.
        """
        if self._arm is None:
            return
        park_xy = self.calibration.pct_to_grbl_mm(*PARK_PCT)
        if park_xy is None:
            return
        if not self._origin_pinned:
            log.warning("park skipped: work frame not pinned to the affine")
            return
        self._arm.rapid_to(*park_xy)
        self._arm.wait_idle()

    def move_to_bbox_center(self, bbox: list[float]):
        """Move arm to the center of a bbox [left, top, right, bottom] (0-1).
        Caller must hold the hardware lock."""
        self.assert_locked()
        t = self.transforms
        if t is None:
            raise RuntimeError("Screen calibration not done")
        cx, cy = t.bbox_center_pct(bbox)
        gx, gy = t.pct_to_grbl_mm(cx, cy)
        arm = self.require_arm()
        arm.rapid_to(gx, gy)
        arm.wait_idle()

    def swipe_from_bbox(
        self,
        bbox: list[float],
        direction: str,
        dist: float,
        speed: str,
        start_dwell: float = 0.0,
        end_dwell: float = 0.0,
    ):
        """Swipe from the bbox center by `dist` screen-fraction in
        `direction` — the swipe sibling of ``move_to_bbox_center``, so
        both press and swipe primitives resolve their coordinates here.
        `start_dwell`/`end_dwell` (s) anchor the touch-down / hold the
        endpoint (see ``StylusArm.swipe_to``). Caller must hold the
        hardware lock."""
        self.assert_locked()
        t = self.transforms
        if t is None:
            raise RuntimeError("Screen calibration not done")
        ex, ey = t.swipe_end_pct(bbox, direction, dist)
        ex_mm, ey_mm = t.pct_to_grbl_mm(ex, ey)
        self.move_to_bbox_center(bbox)
        self.require_arm().swipe_to(
            ex_mm, ey_mm, speed, start_dwell=start_dwell, end_dwell=end_dwell
        )

    # ─── Lifecycle ─────────────────────────────────────────────

    def _park_for_teardown(self):
        """Rest the stylus at the same off-screen park spot used between
        taps and swipes, so the user can place or lift the phone without
        the tip sitting over the glass.

        ``park()`` self-guards (no affine, or a frame not pinned to it →
        no move; see its docstring). Falls back to homing (0, 0) only
        when calibration isn't loaded — the frame ``arm.setup()`` declared
        at connect, a known BELIEVED position; leaving the tip mid-travel
        would be worse. Caveat: believed, not measured — steppers idle off
        and there are no encoders, so a hand-moved gantry desyncs this.
        After moving the arm by hand, reconnect before relying on it.
        """
        if self._arm is None:
            return
        if self.calibration.pct_to_grbl is not None:
            self.park()
        else:
            self._arm.return_to_origin()

    def shutdown(self, lock_timeout: float = 5.0):
        """Release the coil, park the arm off-screen, and close every device
        handle.

        Teardown is best-effort: each step is guarded so a failure in one (a
        serial timeout, a GRBL alarm, an already-disconnected device) can't
        skip the rest and leak the serial/camera handle or strand the coil.
        Failures are logged, never raised — including a second Ctrl-C during
        the bounded lock wait (``try_locked`` absorbs it) — so callers
        (atexit / signal handlers) can rely on shutdown completing.

        Two paths, split by whether the hardware lock is won within
        ``lock_timeout``:

        - Lock won (the normal exit): safety order — release the coil,
          park at the same off-screen spot used between taps (homing only
          if uncalibrated), close. ``arm.close`` re-attempts the release
          as a backstop, and the firmware drops the PWM on alarm.
        - Timeout (a live or abandoned holder): motion-free — ``arm.close``
          only. Its bounded M5 releases the coil when the wire is free,
          and the port close is the true backstop either way (it unblocks
          a stuck reader; the DTR drop resets the board, coil included).
        """

        def _safe(action, desc):
            try:
                action()
            except Exception:
                log.exception("shutdown: %s failed", desc)

        # Order teardown after any in-flight holder (a gesture mid-move,
        # the resume thread mid-re-pin), but never hang an atexit handler.
        # The default wait is long enough for a gesture to land, far
        # shorter than a warm-start bridge wait.
        with self.try_locked(lock_timeout) as got:
            if got:
                if self._arm:
                    _safe(self._arm.lift_stylus, "lift stylus")
                    _safe(self._park_for_teardown, "park")
                    _safe(self._arm.close, "arm close")
            elif self._arm:
                # The lock is owned by a live or abandoned holder: any
                # motion here would be unserialized, and the pin flag can't
                # be trusted from outside the lock (the holder may be
                # mutating the frame right now). No park, no homing —
                # arm.close's bounded release plus the port close (DTR
                # resets the board) handle the coil.
                log.warning(
                    "shutdown: hardware lock still held; closing without parking"
                )
                _safe(self._arm.close, "arm close")
            if self._cam:
                _safe(self._cam.close, "camera close")
