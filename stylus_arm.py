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

import json
import serial
import time

from serial_probe import detect_grbl


# ─── G-code templates ────────────────────────────────────────
# All G-code strings in one place for easy audit and modification.

GCODE_SET_ORIGIN  = 'G92 X0.0 Y0.0 Z0'
GCODE_MM_UNITS    = 'G21'
GCODE_ABSOLUTE    = 'G90'
GCODE_DEFAULT_F   = 'F8000'
GCODE_IDLE_DELAY  = '$1=250'
GCODE_UNLOCK      = '$X'
GCODE_VERSION     = '$I'
GCODE_PEN_DOWN    = 'G1G90 Z{z}F{f}'       # absolute Z down
GCODE_PEN_UP      = 'G1G90 Z{z}F{f}'       # absolute Z up
GCODE_FAST_MOVE   = 'G0 X{x:.3f}Y{y:.3f}F{f}'        # rapid XY (G0)
GCODE_LINEAR_MOVE = 'G1 X{x:.3f}Y{y:.3f}F{f}'        # controlled XY (G1)
GCODE_REL_FAST    = 'G91G0 X{x:.3f}Y{y:.3f}'          # relative rapid
GCODE_REL_LINEAR  = 'G91G1 X{x:.3f}Y{y:.3f}F{f}'     # relative linear
GCODE_VIBRATE_A   = 'G1 Z{z:.2f} F500'     # vibration oscillation

# ─── Main class ──────────────────────────────────────────────

class GrblDevice:

    # Z-axis parameters
    Z_DOWN = None   # pen down position — must be set by calibration (calibrate.py)
    Z_UP   = 0.0    # pen up position (spring rebound)
    # Z-axis speed — matches human finger tap (~100 mm/s).
    # F6000 is realistic and avoids slamming the screen.
    Z_SPEED = 6000

    # Gesture timing (seconds)
    TAP_DURATION = 0.08        # phone threshold ~50ms, 80ms has margin
    DOUBLE_TAP_GAP = 0.1      # gap between two taps, must be < 300ms
    LONG_PRESS_DURATION = 0.8  # iOS/Android threshold ~500ms, 800ms is safe
    SWIPE_DISTANCE = 15        # mm, default swipe length
    MOVE_DIRECTIONS = None     # set by load_calibration() — maps phone directions to arm (x, y)
    MOVE_DISTANCES = {
        'large':  20,           # half the screen away
        'medium': 8,            # a few icons away
        'small':  3,            # one icon away
        'nudge':  1,            # fine-tune
    }
    SWIPE_SPEEDS = {
        'slow':   3000,         # scroll, careful drag
        'medium': 6000,         # normal swipe (~100 mm/s)
        'fast':   10000,        # fling, page switch
    }

    def __init__(self, port=None, baudrate=115200):
        if port is None:
            port = detect_grbl()
        if port is None:
            raise Exception('GRBL device not found, please specify port manually')

        self.ser = serial.Serial(port, baudrate, timeout=3)
        self.port = port
        time.sleep(2)
        self.ser.reset_input_buffer()
        print(f'Connected: {port}')

    # ─── Low-level communication ─────────────────────────────

    def _send(self, cmd, wait_ok=True):
        """Send a single command."""
        print(f'>>> {cmd}')
        self.ser.write((cmd + '\r\n').encode())

        if not wait_ok:
            return

        retries = 0
        while True:
            line = self.ser.readline().decode('utf-8', errors='ignore').strip()
            if line:
                retries = 0
                print(f'<<< {line}')
            else:
                retries += 1
                if retries > 3:
                    raise Exception(f'GRBL not responding, command: {cmd}')
                continue
            if line == 'ok':
                break
            if line.startswith('error'):
                raise Exception(f'GRBL error: {line}  command: {cmd}')
            if line.startswith('ALARM'):
                raise Exception(f'GRBL alarm: {line}, call unlock() first')

    def _query_status(self):
        """Query current status, return status string."""
        self.ser.write(b'?')
        time.sleep(0.1)
        resp = self.ser.read(self.ser.in_waiting or 64).decode('utf-8', errors='ignore')
        for line in resp.splitlines():
            if line.startswith('<'):
                print(f'<<< {line}')
                return line
        return ''

    # ─── Initialization ──────────────────────────────────────

    def setup(self):
        """
        1. Wait for startup message
        2. Query version to confirm connection
        3. Set origin, units, coordinate mode
        """
        print('\n=== Setup ===')

        # Wait and read startup message
        time.sleep(0.5)
        startup = self.ser.read(self.ser.in_waiting or 256).decode('utf-8', errors='ignore')
        if startup.strip():
            print(f'<<< {startup.strip()}')

        self._send(GCODE_VERSION)

        status = self._query_status()
        if 'Alarm' in status:
            print('Alarm detected, unlocking...')
            self.unlock()

        self._send(GCODE_SET_ORIGIN)
        self._send(GCODE_MM_UNITS)
        self._send(GCODE_ABSOLUTE)
        self._send(GCODE_DEFAULT_F)
        self._send(GCODE_IDLE_DELAY)      # 250ms motor idle delay
                                          # 80ms tap is well within range
                                          # auto power-off after 250ms idle, safer than $1=255

        print('=== Setup complete ===\n')

    def load_calibration(self, path='calibration.json'):
        """Load calibration data and build direction mapping.
        Must be called after setup() and before move()/tap()/swipe().
        """
        with open(path) as f:
            cal = json.load(f)

        self.Z_DOWN = cal['z_tap_mm']

        # Build MOVE_DIRECTIONS from calibrated right/down vectors
        rx, ry = cal['right_vec']
        dx, dy = cal['down_vec']
        self.MOVE_DIRECTIONS = {
            'right':      ( rx,      ry),
            'left':       (-rx,     -ry),
            'down':       ( dx,      dy),
            'up':         (-dx,     -dy),
            'up-left':    (-rx - dx, -ry - dy),
            'up-right':   ( rx - dx,  ry - dy),
            'down-left':  (-rx + dx, -ry + dy),
            'down-right': ( rx + dx,  ry + dy),
        }

        print(f'Calibration loaded: Z={self.Z_DOWN} mm, '
              f'right=({rx},{ry}), down=({dx},{dy})')

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
        print('Origin set to current position')

    # ─── Basic motions ──

    def _pen_down(self, z=None, speed=None):
        """Lower stylus. G1G90: always reassert absolute mode to prevent
        Z-axis crushing the screen due to mode errors.
        z: override Z depth (used by calibration probing). Defaults to Z_DOWN.
        speed: override Z speed. Defaults to Z_SPEED.
        """
        z = z if z is not None else self.Z_DOWN
        if z is None:
            raise RuntimeError('Z_DOWN not set — run calibration first')
        f = speed or self.Z_SPEED
        self._send(GCODE_PEN_DOWN.format(z=z, f=f))

    def _pen_up(self):
        """Raise stylus. Actively drive Z back to 0 instead of relying on spring,
        keeps GRBL coordinate tracking in sync.
        """
        self._send(GCODE_PEN_UP.format(z=self.Z_UP, f=self.Z_SPEED))

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

    def _tap_with_vibration(self, duration=0.08):
        """
        Micro-vibration method: after pen down, continuously oscillate Z-axis
        by a tiny amount to keep the motor powered.
        $1=250 covers short taps (80ms), but long press (800ms) needs this
        to prevent spring rebound.
        Principle: Z-axis always has pending commands, $1 timer keeps resetting
        Amplitude: 0.02mm (Z-axis min step size), imperceptible to the screen
        """
        self._pen_down()
        steps = max(1, int(duration / 0.02))
        for _ in range(steps):
            self._send(GCODE_VIBRATE_A.format(z=self.Z_DOWN - 0.02))
            self._send(GCODE_VIBRATE_A.format(z=self.Z_DOWN))
        self._pen_up()


    # ─── Public API (for AI agent) ─────────────────────────

    def move(self, direction, distance='medium'):
        """Move stylus relative to current position.
        direction: 'up', 'down', 'left', 'right',
                   'up-left', 'up-right', 'down-left', 'down-right'
        distance: 'large', 'medium', 'small', 'nudge'
        """
        if self.MOVE_DIRECTIONS is None:
            raise RuntimeError('MOVE_DIRECTIONS not set — run load_calibration() first')
        mx, my = self.MOVE_DIRECTIONS[direction]
        d = self.MOVE_DISTANCES[distance]
        self._send(GCODE_REL_FAST.format(x=mx * d, y=my * d))
        self._send(GCODE_ABSOLUTE)

    def tap(self):
        """Single tap at current position."""
        self._tap_with_vibration(self.TAP_DURATION)

    def double_tap(self):
        """Double tap at current position."""
        self._tap_with_vibration(self.TAP_DURATION)
        time.sleep(self.DOUBLE_TAP_GAP)
        self._tap_with_vibration(self.TAP_DURATION)

    def long_press(self):
        """Long press at current position."""
        self._tap_with_vibration(self.LONG_PRESS_DURATION)

    def swipe(self, direction, speed='medium'):
        """Swipe from current position in a cardinal direction.
        direction: 'up', 'down', 'left', 'right'
        speed: 'slow', 'medium', 'fast'
        """
        d = self.SWIPE_DISTANCE
        offsets = {
            'up':    (0, -d),
            'down':  (0,  d),
            'left':  (-d, 0),
            'right': ( d, 0),
        }
        dx, dy = offsets[direction]
        f = self.SWIPE_SPEEDS[speed]
        self._pen_down()
        self._send(GCODE_REL_LINEAR.format(x=dx, y=dy, f=f))
        self._send(GCODE_ABSOLUTE)
        self._pen_up()

    def close(self):
        """Close serial port."""
        self.ser.close()
        print('Serial port closed')
