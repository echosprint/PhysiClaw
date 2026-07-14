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


def _layout_learned() -> bool:
    """Read the first-run layout's `learned` marker straight from its JSON.

    The agent-side `screen_layout` module writes the marker (it owns the
    page/field schema); core only reads it, so the rig can report
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
        self._assistive_touch = AssistiveTouch()
        self._bridge: BridgeState | None = None
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
        also schedule `Perception.tune_exposure()` — see the two existing
        callers (`server/watch.py` /api/ready, `server/warm_start.py`)."""
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
        at = self.require_assistive_touch()
        return at.take_screenshot(
            self._arm, self._bridge, self.transforms.pct_to_grbl, timeout=timeout
        )

    def at_long_press(self) -> None:
        """Long-press the AssistiveTouch button (fires the clipboard Shortcut)."""
        at = self.require_assistive_touch()
        at.long_press(self._arm, self.transforms.pct_to_grbl)

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
        self._arm.rapid_to(*park_xy)
        self._arm.wait_idle()

    def move_to_bbox_center(self, bbox: list[float]):
        """Move arm to the center of a bbox [left, top, right, bottom] (0-1)."""
        t = self.transforms
        if t is None:
            raise RuntimeError("Screen calibration not done")
        cx, cy = t.bbox_center_pct(bbox)
        gx, gy = t.pct_to_grbl_mm(cx, cy)
        arm = self.require_arm()
        arm.rapid_to(gx, gy)
        arm.wait_idle()

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
