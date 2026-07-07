"""
GRBL stylus arm controller for phone touch automation.

The arm owns the GRBL serial transport and the X/Y gantry, and exposes the
high-level touch gestures (tap, double-tap, long-press, swipe, move). The
touch (Z) itself is a solenoid driven through the spindle PWM pin — that
actuator lives in :class:`physiclaw.core.hardware.solenoid.Solenoid`, which
this class composes and drives. There is no stepper Z axis and no Z depth to
calibrate — the solenoid stroke is mechanical.

IMPORTANT: The arm must be calibrated before use (run calibrate.py).
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
import serial
import time

from physiclaw.core.hardware.grbl import detect_grbl
from physiclaw.core.hardware.solenoid import Solenoid

log = logging.getLogger(__name__)


# Error codes that mean "this firmware doesn't take this setting at
# runtime" — the value lives in YAML config (FluidNC) or isn't supported.
# Swallowed only when a command is sent with optional=True. Matches
# scripts/grbl_solenoid_test.py.
_OPTIONAL_ERRORS = frozenset({"error:3", "error:162"})


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

    def __init__(self, port=None, baudrate=115200):
        if port is None:
            port = detect_grbl()
        if port is None:
            raise Exception("GRBL device not found, please specify port manually")

        self.ser = serial.Serial(port, baudrate, timeout=3)
        self.port = port
        # The Z executor: a solenoid driven over this arm's GRBL channel.
        self.solenoid = Solenoid(send=self._send, dwell=self._dwell)
        time.sleep(2)
        self.ser.reset_input_buffer()
        log.info(f"Arm connected: {port}")

    # ─── Low-level communication ─────────────────────────────

    def _send(self, cmd, wait_ok=True, optional=False):
        """Send a single command, block until 'ok'.

        optional=True swallows the "setting not accepted at runtime" error
        codes (error:3 / error:162) so PWM-config writes ($32/$30) work on
        FluidNC (which takes them from YAML) and a bare GRBL board alike.
        """
        log.debug(f">>> {cmd}")
        # LF only — NOT CRLF. FluidNC v4 treats the trailing `\r` + `\n` as two
        # line endings (the command + an empty line) and acks BOTH with `ok`;
        # that extra `ok` lags every reply by one and desyncs the stream (a
        # later command then reads a stale error). `\n` is the canonical GRBL
        # terminator and yields exactly one reply per command.
        self.ser.write((cmd + "\n").encode())

        if not wait_ok:
            return

        retries = 0
        while True:
            line = self.ser.readline().decode("utf-8", errors="ignore").strip()
            if line:
                retries = 0
                log.debug(f"<<< {line}")
            else:
                retries += 1
                if retries > 3:
                    raise Exception(f"GRBL not responding, command: {cmd}")
                continue
            if line == "ok":
                break
            if line.startswith("error"):
                if optional and line.replace(" ", "") in _OPTIONAL_ERRORS:
                    # FluidNC follows a rejected setting with a trailing
                    # `[MSG:ERR: ...]` line (no `ok`). It's a non-terminator,
                    # so the next command's read loop skips it harmlessly — no
                    # draining needed now that we send LF (single reply each).
                    log.debug(f"{cmd}: {line} not supported at runtime — skipping")
                    break
                raise Exception(f"GRBL error: {line}  command: {cmd}")
            if line.startswith("ALARM"):
                raise Exception(f"GRBL alarm: {line}, call unlock() first")

    def _query_status(self):
        """Query current status, return status string."""
        self.ser.write(b"?")
        time.sleep(0.1)
        resp = self.ser.read(self.ser.in_waiting or 64).decode("utf-8", errors="ignore")
        for line in resp.splitlines():
            if line.startswith("<"):
                log.debug(f"<<< {line}")
                return line
        return ""

    def wait_idle(self, timeout=10):
        """Poll until GRBL reports Idle status."""
        time.sleep(0.01)  # let GRBL transition from Idle→Run after buffering
        deadline = time.time() + timeout
        while time.time() < deadline:
            status = self._query_status()
            if "Idle" in status:
                return
            time.sleep(0.1)
        raise RuntimeError(f"Arm not idle after {timeout}s")

    def position(self) -> tuple[float, float]:
        """Return current (x, y) in work coordinates (WPos).

        GRBL reports MPos (machine position) and/or WPos (work position).
        All G-code moves use WPos (set by G92), so we must return WPos.
        If GRBL reports WPos directly, use it. Otherwise compute from
        MPos - WCO (work coordinate offset).
        """
        import re

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
                "GRBL not reporting WPos or WCO — position may be wrong. "
                "Set $10=0 for WPos reporting."
            )
            return float(m_mpos.group(1)), float(m_mpos.group(2))

        raise RuntimeError(f"Cannot parse position from: {status}")

    # ─── Initialization ──────────────────────────────────────

    def setup(self):
        """
        1. Wait for startup message
        2. Query version to confirm connection
        3. Set origin, units, coordinate mode
        """
        # Wait and read startup message
        time.sleep(0.5)
        startup = self.ser.read(self.ser.in_waiting or 256).decode(
            "utf-8", errors="ignore"
        )
        if startup.strip():
            log.debug(f"<<< {startup.strip()}")

        self._send(GCODE_VERSION)

        status = self._query_status()
        if "Alarm" in status:
            log.debug("Alarm detected, unlocking...")
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

    def set_direction_mapping(self, right_vec: tuple, down_vec: tuple):
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

    def unlock(self):
        """Clear alarm lock.

        Uses $X (kill alarm) instead of $H (homing cycle) because
        this pen plotter has no limit switches — $H would run the
        axes into the frame and stall.
        """
        self._send(GCODE_UNLOCK)

    def set_origin(self):
        """Set current position as coordinate origin (move stylus to target first)."""
        self._send(GCODE_SET_ORIGIN)
        log.debug("Origin set to current position")

    def set_work_position(self, x, y):
        """Declare the current physical position to be work coordinate (x, y).

        G92 shifts the work coordinate system without moving the arm. Used on
        warm-start reconnect to re-pin the calibrated frame from the known
        park-spot resting position (see ``PhysiClaw.restore_park_origin``).
        """
        self._send(GCODE_SET_WORK_POS.format(x=x, y=y))
        log.debug("Work position set to (%.3f, %.3f)", x, y)

    def return_to_origin(self):
        """Fast-move back to (0, 0) and wait for motion to settle."""
        self._fast_move(0, 0)
        self.wait_idle()

    # ─── Basic motions ──

    def _dwell(self, seconds):
        """Hold position for duration (GRBL-side timing, 50ms granularity).
        G4 is a sync barrier: drains the planner first, then dwells.
        _send() blocks until the dwell completes.
        """
        self._send(GCODE_DWELL.format(s=seconds))

    def _fast_move(self, x, y, speed=8000):
        """Rapid move without touching screen (G0). Pen must be up first."""
        self._send(GCODE_FAST_MOVE.format(x=x, y=y, f=speed))

    def _linear_move(self, x, y, speed=8000):
        """Linear move at controlled speed (G1) — used for a swipe slide while
        the solenoid holds the tip down."""
        self._send(GCODE_LINEAR_MOVE.format(x=x, y=y, f=speed))

    # ─── Public API (for AI agent) ─────────────────────────

    def move(self, direction, distance="medium"):
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

    def tap(self):
        """Single tap at current position."""
        self.solenoid.tap(self.TAP_DURATION)
        self.wait_idle()

    def double_tap(self):
        """Double tap at current position.

        One solenoid double-tap: two strikes separated only by a brief
        contact-breaking lift (not the full spring-clear release), so
        down-to-down stays well under the iOS ~300ms double-tap window and the
        pair registers as one double-tap instead of two single taps.
        """
        self.solenoid.double_tap(self.TAP_DURATION)
        self.wait_idle()

    def long_press(self):
        """Long press at current position."""
        self.solenoid.press_and_hold(self.LONG_PRESS_DURATION)
        self.wait_idle()

    def swipe(self, direction, speed="medium"):
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

    def swipe_to(self, x, y, speed="medium", start_dwell=0.0):
        """Swipe to an absolute work-coordinate (x, y) in mm: press, slide,
        release. Unlike :meth:`swipe` (relative cardinal direction), this
        slides to a caller-computed endpoint — used by the orchestrator with
        calibrated screen→arm mm coordinates.

        `start_dwell` (s) holds the tip stationary at the start AFTER contact,
        before the slide — so iOS registers a touch-down at that point before
        the drag. Needed for the left-edge back gesture: the interactive-pop
        recognizer only arms if the touch is seen at the edge first; a fast
        slide that starts moving on contact skips past the narrow edge zone."""
        with self.solenoid.held():
            if start_dwell:
                self._dwell(start_dwell)
            self._linear_move(x, y, speed=self.SWIPE_SPEEDS[speed])
        self.wait_idle()

    def lift_stylus(self):
        """Lift the stylus tip off the screen (release the Z actuator).

        The arm-level name for "release contact" — callers don't need to know
        Z is a solenoid. Used by teardown to clear the glass before homing.
        """
        self.solenoid.release()

    def close(self):
        """Release the solenoid and close serial port."""
        try:
            self.solenoid.release()
        except Exception:
            pass
        self.ser.close()
        log.debug("Serial port closed")
