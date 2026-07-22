"""
GRBL stylus arm controller for phone touch automation.

The arm owns the GRBL serial transport and the X/Y gantry, and exposes the
high-level touch gestures (tap, double-tap, long-press, swipe, move). The
touch (Z) itself is a solenoid driven through the spindle PWM pin — that
actuator lives in :class:`physiclaw.core.hardware.solenoid.Solenoid`, which
this class composes and drives. There is no stepper Z axis and no Z depth to
calibrate — the solenoid stroke is mechanical.

IMPORTANT: The arm must be calibrated before use (`physiclaw setup hardware`).
Calibration determines only the X/Y mapping: which arm axis maps to
which phone axis (e.g. arm X+ = phone right, arm Y+ = phone down).
Place the phone aligned with the arm axes, no rotation — portrait or
landscape both work.

During calibration, the user manually positions the stylus right above
the center orange circle on the phone — this becomes arm position (0, 0).
After calibration, phone directions (right/left/up/down) are mapped
to arm axes automatically.
"""

import logging
import re
import time

import serial

from physiclaw.core.hardware.device import (
    DeviceNotFound,
    DeviceTimeout,
    ProtocolError,
)
from physiclaw.core.hardware.grbl import candidate_ports, detect_grbl
from physiclaw.core.hardware.solenoid import Solenoid
from physiclaw.core.hardware.transport import SerialTransport

log = logging.getLogger(__name__)


# ─── G-code templates ────────────────────────────────────────
# Transport + XY motion only. Solenoid (M3/M5) G-code lives in solenoid.py.

GCODE_SET_ORIGIN = "G92 X0.0 Y0.0 Z0"
GCODE_SET_WORK_POS = "G92 X{x:.3f} Y{y:.3f} Z0"  # declare current pos = (x, y)
GCODE_MM_UNITS = "G21"
GCODE_ABSOLUTE = "G90"
GCODE_DEFAULT_F = "F8000"
GCODE_IDLE_DELAY = "$1=250"
GCODE_UNLOCK = "$X"
GCODE_VERSION = "$I"
GCODE_FAST_MOVE = "G0 X{x:.3f}Y{y:.3f}F{f}"  # rapid XY (G0)
GCODE_LINEAR_MOVE = "G1 X{x:.3f}Y{y:.3f}F{f}"  # controlled XY (G1)
GCODE_REL_FAST = "G91G0 X{x:.3f}Y{y:.3f}"  # relative rapid
GCODE_REL_LINEAR = "G91G1 X{x:.3f}Y{y:.3f}F{f}"  # relative linear
GCODE_DWELL = "G4 P{s}"  # dwell for s seconds (planner-side)

# ─── Main class ──────────────────────────────────────────────


class StylusArm:
    # Gesture timing (seconds) — how long the solenoid stays on the glass for
    # each gesture. The solenoid's own electrical/mechanical constants (strike
    # duty, hold duty, rebound) live in solenoid.Solenoid.
    TAP_DURATION = 0.08  # phone threshold ~50ms, 80ms has margin
    LONG_PRESS_DURATION = 1.2  # iOS/Android threshold ~500ms, 800~1000ms is safe
    SWIPE_DISTANCE = 15  # mm, default swipe length
    MOVE_DIRECTIONS = (
        None  # set by set_direction_mapping() — maps phone directions to arm (x, y)
    )
    MOVE_DISTANCES = {
        "large": 20,  # half the screen away
        "medium": 8,  # a few icons away
        "small": 3,  # one icon away
        "nudge": 1,  # fine-tune
    }
    SWIPE_SPEEDS = {
        "slow": 3000,  # scroll, careful drag
        "medium": 6000,  # normal swipe (~100 mm/s)
        "fast": 10000,  # fling, page switch
    }

    def __init__(self, port: str | None = None, baudrate: int = 115200) -> None:
        if port is None:
            port = detect_grbl()
        if port is None:
            # No ports at all ≠ ports that didn't answer as GRBL: the
            # first almost always means the board isn't plugged in or
            # isn't powered, and "specify port manually" would send the
            # user hunting for a port that doesn't exist.
            if not candidate_ports():
                raise DeviceNotFound(
                    "GRBL device not found — no serial ports detected; "
                    "is the control board plugged into USB and its 12V "
                    "power on?"
                )
            raise DeviceNotFound("GRBL device not found, please specify port manually")

        self.ser = serial.Serial(port, baudrate, timeout=3)
        try:
            self.transport = SerialTransport(self.ser)
            self.port = port
            # The Z executor: a solenoid driven over this arm's GRBL channel.
            self.solenoid = Solenoid(send=self._send, dwell=self._dwell)
            time.sleep(2)
            self.transport.reset_input_buffer()
        except BaseException:
            # Same rule Camera.__init__ follows: a failure after the open
            # (cable yanked during the settle) must not leave the port
            # handle claimed until GC — an immediate reconnect would find
            # it still held (exclusive on Windows COM).
            self.ser.close()
            raise
        log.info(f"Arm connected: {port}")

    # ─── Low-level communication ─────────────────────────────
    # Wire mechanics (send/ok handshake, status query, the port lock)
    # live in SerialTransport; these thin delegates keep the arm's
    # internal call sites and the solenoid binding unchanged.

    def _send(self, cmd: str, wait_ok: bool = True, optional: bool = False) -> None:
        self.transport.send(cmd, wait_ok=wait_ok, optional=optional)

    def _query_status(self) -> str:
        return self.transport.query_status()

    def health(self) -> bool:
        """Live check: does the board answer a status query right now?
        Distinguishes a cable pulled mid-session from 'was connected once'
        (see the Device protocol in ``device.py``)."""
        try:
            return self._query_status() != ""
        except Exception:
            return False

    def wait_idle(self, timeout: float = 10) -> None:
        """Poll until GRBL reports Idle status."""
        time.sleep(0.01)  # let GRBL transition from Idle→Run after buffering
        deadline = time.time() + timeout
        while time.time() < deadline:
            status = self._query_status()
            if "Idle" in status:
                return
            time.sleep(0.1)
        raise DeviceTimeout(f"Arm not idle after {timeout}s")

    def position(self) -> tuple[float, float]:
        """Return current (x, y) in work coordinates (WPos).

        GRBL reports MPos (machine position) and/or WPos (work position).
        All G-code moves use WPos (set by G92), so we must return WPos.
        If GRBL reports WPos directly, use it. Otherwise compute from
        MPos - WCO (work coordinate offset).
        """
        status = self._query_status()

        # Try WPos first (GRBL $10=0)
        m = re.search(r"WPos:([-\d.]+),([-\d.]+)", status)
        if m:
            return float(m.group(1)), float(m.group(2))

        # Fall back to MPos - WCO
        m_mpos = re.search(r"MPos:([-\d.]+),([-\d.]+)", status)
        m_wco = re.search(r"WCO:([-\d.]+),([-\d.]+)", status)
        if m_mpos and m_wco:
            mx, my = float(m_mpos.group(1)), float(m_mpos.group(2))
            wx, wy = float(m_wco.group(1)), float(m_wco.group(2))
            return mx - wx, my - wy

        if m_mpos:
            # No WCO available — configure GRBL to report WPos with $10=0
            log.warning(
                "[GRBL] board not reporting WPos or WCO — position may be "
                "wrong; set $10=0 for WPos reporting"
            )
            return float(m_mpos.group(1)), float(m_mpos.group(2))

        raise ProtocolError(f"Cannot parse position from: {status}")

    # ─── Initialization ──────────────────────────────────────

    def setup(self) -> None:
        """
        1. Wait for startup message
        2. Query version to confirm connection
        3. Set origin, units, coordinate mode
        """
        # Wait and read startup message
        time.sleep(0.5)
        startup = self.transport.read_startup()
        if startup.strip():
            log.debug(f"[GRBL] startup banner: {startup.strip()}")

        self._send(GCODE_VERSION)

        status = self._query_status()
        if "Alarm" in status:
            log.debug("[GRBL] alarm detected, unlocking...")
            self.unlock()

        self._send(GCODE_SET_ORIGIN)
        self._send(GCODE_MM_UNITS)
        self._send(GCODE_ABSOLUTE)
        self._send(GCODE_DEFAULT_F)
        # $-settings are firmware-owned: FluidNC takes them from YAML and
        # rejects live writes (error:3 "Invalid $ statement" / error:162
        # "Read-only"), while bare GRBL accepts them. Send best-effort so both
        # work — same rationale as the solenoid's $32/$30 in configure().
        self._send(GCODE_IDLE_DELAY, optional=True)  # $1=250 motor idle delay
        self._send("$10=0", optional=True)  # report WPos in status (not MPos)

        self.solenoid.configure()  # spindle PWM mode + range, coil off

        log.info("Arm setup complete")

    def set_direction_mapping(self, right_vec: tuple, down_vec: tuple) -> None:
        """Build MOVE_DIRECTIONS from calibrated right/down vectors."""
        rx, ry = right_vec
        dx, dy = down_vec
        self.MOVE_DIRECTIONS = {
            "right": (rx, ry),
            "left": (-rx, -ry),
            "bottom": (dx, dy),
            "top": (-dx, -dy),
            "top-left": (-rx - dx, -ry - dy),
            "top-right": (rx - dx, ry - dy),
            "bottom-left": (-rx + dx, -ry + dy),
            "bottom-right": (rx + dx, ry + dy),
        }

    def unlock(self) -> None:
        """Clear alarm lock.

        Uses $X (kill alarm) instead of $H (homing cycle) because
        this pen plotter has no limit switches — $H would run the
        axes into the frame and stall.
        """
        self._send(GCODE_UNLOCK)

    def set_origin(self) -> None:
        """Set current position as coordinate origin (move stylus to target first)."""
        self._send(GCODE_SET_ORIGIN)
        log.debug("Origin set to current position")

    def set_work_position(self, x: float, y: float) -> None:
        """Declare the current physical position to be work coordinate (x, y).

        G92 shifts the work coordinate system without moving the arm. Used on
        warm-start reconnect to re-pin the calibrated frame from the known
        park-spot resting position (see ``HardwareRig.restore_park_origin``).
        """
        self._send(GCODE_SET_WORK_POS.format(x=x, y=y))
        log.debug("Work position set to (%.3f, %.3f)", x, y)

    def return_to_origin(self) -> None:
        """Fast-move back to (0, 0) and wait for motion to settle."""
        self.rapid_to(0, 0)
        self.wait_idle()

    # ─── Basic motions ──

    def _dwell(self, seconds: float) -> None:
        """Hold position for duration (GRBL-side timing, 50ms granularity).
        G4 is a sync barrier: drains the planner first, then dwells.
        _send() blocks until the dwell completes.
        """
        self._send(GCODE_DWELL.format(s=seconds))

    def rapid_to(self, x: float, y: float, speed: int = 8000) -> None:
        """Rapid move to a work coordinate without touching the screen (G0).
        Pen must be up first. Public: the orchestration and calibration
        layers drive every calibrated-coordinate positioning move with it."""
        self._send(GCODE_FAST_MOVE.format(x=x, y=y, f=speed))

    def _linear_move(self, x: float, y: float, speed: int = 8000) -> None:
        """Linear move at controlled speed (G1) — used for a swipe slide while
        the solenoid holds the tip down."""
        self._send(GCODE_LINEAR_MOVE.format(x=x, y=y, f=speed))

    # ─── Public API (for AI agent) ─────────────────────────

    def move(self, direction: str, distance: str = "medium") -> None:
        """Move stylus relative to current position.
        direction: 'top', 'bottom', 'left', 'right',
                   'top-left', 'top-right', 'bottom-left', 'bottom-right'
        distance: 'large', 'medium', 'small', 'nudge'
        """
        if self.MOVE_DIRECTIONS is None:
            raise RuntimeError("MOVE_DIRECTIONS not set — run calibration first")
        mx, my = self.MOVE_DIRECTIONS[direction]
        d = self.MOVE_DISTANCES[distance]
        # Normalize diagonal vectors so actual distance matches intended distance
        mag = (mx**2 + my**2) ** 0.5 or 1
        self._send(GCODE_REL_FAST.format(x=mx / mag * d, y=my / mag * d))
        self._send(GCODE_ABSOLUTE)
        self.wait_idle()

    def tap(self) -> None:
        """Single tap at current position."""
        self.solenoid.tap(self.TAP_DURATION)
        self.wait_idle()

    def double_tap(self) -> None:
        """Double tap at current position.

        One solenoid double-tap: two strikes separated only by a brief
        contact-breaking lift (not the full spring-clear release), so
        down-to-down stays well under the iOS ~300ms double-tap window and the
        pair registers as one double-tap instead of two single taps.
        """
        self.solenoid.double_tap(self.TAP_DURATION)
        self.wait_idle()

    def long_press(self) -> None:
        """Long press at current position."""
        self.solenoid.press_and_hold(self.LONG_PRESS_DURATION)
        self.wait_idle()

    def swipe(self, direction: str, speed: str = "medium") -> None:
        """Swipe from current position in a cardinal direction.
        direction: 'top', 'bottom', 'left', 'right'
        speed: 'slow', 'medium', 'fast'
        """
        if self.MOVE_DIRECTIONS is None:
            raise RuntimeError("MOVE_DIRECTIONS not set — run calibration first")
        mx, my = self.MOVE_DIRECTIONS[direction]
        d = self.SWIPE_DISTANCE
        mag = (mx**2 + my**2) ** 0.5 or 1
        dx = mx / mag * d
        dy = my / mag * d
        f = self.SWIPE_SPEEDS[speed]
        # pressed() holds the tip down (at HOLD_S) for the slide and guarantees
        # release even on error, so a failed slide can't leave the coil hot.
        with self.solenoid.held():
            self._send(GCODE_REL_LINEAR.format(x=dx, y=dy, f=f))  # XY slide
            self._send(GCODE_ABSOLUTE)  # restore absolute mode

    def swipe_to(
        self,
        x: float,
        y: float,
        speed: str = "medium",
        start_dwell: float = 0.0,
        end_dwell: float = 0.0,
    ) -> None:
        """Swipe to an absolute work-coordinate (x, y) in mm: press, slide,
        release. Unlike :meth:`swipe` (relative cardinal direction), this
        slides to a caller-computed endpoint — used by the orchestrator with
        calibrated screen→arm mm coordinates.

        `start_dwell` (s) holds the tip stationary at the start AFTER contact,
        before the slide — so iOS registers a touch-down at that point before
        the drag. Needed for the left-edge back gesture: the interactive-pop
        recognizer only arms if the touch is seen at the edge first; a fast
        slide that starts moving on contact skips past the narrow edge zone.

        `end_dwell` (s) holds the tip pressed at the endpoint AFTER the slide,
        before release — iOS reads a bottom swipe-up-and-hold as "open the app
        switcher", while releasing on arrival reads as "go home". G4 is a sync
        barrier, so the dwell runs after the slide finishes and the release
        M-code queues after the dwell."""
        with self.solenoid.held():
            if start_dwell:
                self._dwell(start_dwell)
            self._linear_move(x, y, speed=self.SWIPE_SPEEDS[speed])
            if end_dwell:
                self._dwell(end_dwell)
        self.wait_idle()

    def lift_stylus(self) -> None:
        """Lift the stylus tip off the screen (release the Z actuator).

        The arm-level name for "release contact" — callers don't need to know
        Z is a solenoid. Used by teardown to clear the glass before homing.
        """
        self.solenoid.release()

    def close(self) -> None:
        """Release the solenoid and close serial port."""
        try:
            self.solenoid.release()
        except Exception:
            pass
        self.transport.close()
