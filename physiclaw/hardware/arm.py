"""
GRBL stylus arm controller for phone touch automation.

IMPORTANT: The arm must be calibrated before use (run calibrate.py).
Calibration determines:
  - Z depth: how far the stylus tip must descend to touch the screen
    (too far will break the phone screen)
  - X/Y mapping: which arm axis maps to which phone axis
    (e.g. arm X+ = phone right, arm Y+ = phone down)
    (place the phone aligned with the arm axes, no rotation — portrait or landscape both work)

During calibration, the user manually positions the stylus right above
the center orange circle on the phone — this becomes arm position (0, 0).
After calibration, phone directions (right/left/up/down) are mapped
to arm axes automatically. Z = Z_DOWN touches screen, Z = 0 lifts off.
"""

import logging
import serial
import time

from physiclaw.hardware.grbl import detect_grbl

log = logging.getLogger(__name__)


# ─── G-code templates ────────────────────────────────────────
# All G-code strings in one place for easy audit and modification.

GCODE_SET_ORIGIN = "G92 X0.0 Y0.0 Z0"
GCODE_MM_UNITS = "G21"
GCODE_ABSOLUTE = "G90"
GCODE_DEFAULT_F = "F8000"
GCODE_IDLE_DELAY = "$1=250"
GCODE_UNLOCK = "$X"
GCODE_VERSION = "$I"
GCODE_PEN_DOWN = "G1G90 Z{z}F{f}"  # absolute Z down
GCODE_PEN_UP = "G1G90 Z{z}F{f}"  # absolute Z up
GCODE_FAST_MOVE = "G0 X{x:.3f}Y{y:.3f}F{f}"  # rapid XY (G0)
GCODE_LINEAR_MOVE = "G1 X{x:.3f}Y{y:.3f}F{f}"  # controlled XY (G1)
GCODE_REL_FAST = "G91G0 X{x:.3f}Y{y:.3f}"  # relative rapid
GCODE_REL_LINEAR = "G91G1 X{x:.3f}Y{y:.3f}F{f}"  # relative linear
GCODE_DWELL = "G4 P{s}"  # dwell for s seconds (planner-side)

# ─── Main class ──────────────────────────────────────────────


class StylusArm:
    # Z-axis parameters
    Z_DOWN = None  # pen down position — must be set by calibration (calibrate.py)
    Z_UP = 0.0  # pen up position (spring rebound)
    # Z-axis speed — matches human finger tap (~100 mm/s).
    # F6000 is realistic and avoids slamming the screen.
    Z_SPEED = 6000

    # Gesture timing (seconds)
    TAP_DURATION = 0.08  # phone threshold ~50ms, 80ms has margin
    DOUBLE_TAP_GAP = 0  # no dwell gap; pen travel alone provides ~70ms gap
    LONG_PRESS_DURATION = 1.2  # iOS/Android threshold ~500ms, 800~1000ms is safe
    LONG_PRESS_ADVANCE = 0.25  # mm extra Z to travel during long press hold
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
        time.sleep(2)
        self.ser.reset_input_buffer()
        log.info(f"Arm connected: {port}")

    # ─── Low-level communication ─────────────────────────────

    def _send(self, cmd, wait_ok=True):
        """Send a single command."""
        log.debug(f">>> {cmd}")
        self.ser.write((cmd + "\r\n").encode())

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
        self._send(GCODE_IDLE_DELAY)  # 250ms motor idle delay
        self._send("$10=0")  # report WPos in status (not MPos)
        # 80ms tap is well within range
        # auto power-off after 250ms idle, safer than $1=255

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

    def return_to_origin(self):
        """Fast-move back to (0, 0) and wait for motion to settle."""
        self._fast_move(0, 0)
        self.wait_idle()

    # ─── Basic motions ──

    def _pen_down(self, z=None, speed=None):
        """Lower stylus. G1G90: always reassert absolute mode to prevent
        Z-axis crushing the screen due to mode errors.
        Buffers the command only — caller must use _dwell() or wait_idle()
        to ensure contact before proceeding.
        z: override Z depth (used by calibration probing). Defaults to Z_DOWN.
        speed: override Z speed. Defaults to Z_SPEED.
        """
        z = z if z is not None else self.Z_DOWN
        if z is None:
            raise RuntimeError("Z_DOWN not set — run calibration first")
        f = speed or self.Z_SPEED
        self._send(GCODE_PEN_DOWN.format(z=z, f=f))

    def _pen_up(self):
        """Raise stylus. Actively drive Z back to 0 instead of relying on spring,
        keeps GRBL coordinate tracking in sync.
        """
        self._send(GCODE_PEN_UP.format(z=self.Z_UP, f=self.Z_SPEED))

    def _dwell(self, seconds):
        """Hold position for duration (GRBL-side timing, 50ms granularity).
        G4 is a sync barrier: drains the planner first, then dwells.
        _send() blocks until the dwell completes. The $1 idle timer
        runs during the dwell — caller must set $1=255 for long holds.
        """
        self._send(GCODE_DWELL.format(s=seconds))

    def _fast_move(self, x, y, speed=8000):
        """Rapid move without touching screen (G0). Pen must be up first."""
        self._send(GCODE_FAST_MOVE.format(x=x, y=y, f=speed))

    def _linear_move(self, x, y, speed=8000):
        """
        Linear move at controlled speed (G1) — used for swipe while pen is down.
        Continuous XY motion keeps resetting $1 timer, Z motor stays powered,
        spring cannot rebound.
        """
        self._send(GCODE_LINEAR_MOVE.format(x=x, y=y, f=speed))

    # ─── Tap mechanics ───────────────────────────────────────

    def _hold_contact(self, duration):
        """Hold stylus on screen for duration seconds (GRBL-timed via G4).

        G4 is a sync barrier: it drains the planner, then dwells. During
        the dwell, the $1 idle timer runs. For durations > $1 timeout
        (250ms), set $1=255 to keep the Z motor powered and prevent
        spring rebound.

        G4 has 50ms granularity — actual dwell is rounded up to the
        next multiple of 50ms (e.g. 80ms → 100ms). Acceptable for
        phone touch thresholds.
        """
        needs_hold = duration > 0.2
        if needs_hold:
            self._set_motors_always_on(True)
        try:
            self._pen_down()
            self._dwell(duration)
            self._pen_up()
            self.wait_idle()
        finally:
            if needs_hold:
                self._set_motors_always_on(False)

    def _set_motors_always_on(self, always_on):
        """Keep stepper motors powered ($1=255) or restore normal idle timeout ($1=250).
        Retries on failure to prevent stuck state.
        """
        ms = 255 if always_on else 250
        for _ in range(3):
            try:
                self._send(f"$1={ms}")
                return
            except Exception:
                self.unlock()

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
        self._hold_contact(self.TAP_DURATION)

    def double_tap(self):
        """Double tap at current position.
        G4 P0 sync barrier forces pen_up to complete before pen_down,
        preventing GRBL motion planner from blending the two moves.
        Total gap = pen_up travel + pen_down travel ≈ 70ms
        (well under 300ms iOS double-tap threshold).
        """
        self._pen_down()
        self._dwell(self.TAP_DURATION)
        self._pen_up()
        self._dwell(self.DOUBLE_TAP_GAP)  # sync barrier — ensure pen fully rises
        self._pen_down()
        self._dwell(self.TAP_DURATION)
        self._pen_up()
        self.wait_idle()

    def long_press(self):
        """Long press at current position."""
        self._hold_contact(self.LONG_PRESS_DURATION)

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
        self._pen_down()  # Z down (queued)
        self._send(
            GCODE_REL_LINEAR.format(x=dx, y=dy, f=f)
        )  # XY slide (queued after Z)
        self._send(GCODE_ABSOLUTE)  # restore absolute mode
        self._pen_up()  # Z up (queued after slide)

    def close(self):
        """Restore safe defaults and close serial port."""
        try:
            self._set_motors_always_on(False)
        except Exception:
            pass
        self.ser.close()
        log.debug("Serial port closed")
