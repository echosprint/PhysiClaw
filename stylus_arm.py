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

import serial
import time

from serial_probe import detect_grbl


# ─── Main class ──────────────────────────────────────────────

class GrblDevice:

    # Z-axis parameters
    Z_DOWN = None   # pen down position — must be set by calibration (calibrate.py)
    Z_UP   = 0.0    # pen up position (spring rebound)
    # Z-axis speed — matches human finger tap (~100 mm/s).
    # F6000 is realistic and avoids slamming the screen.
    Z_SPEED = 6000

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

        # Query version
        self._send('$I')

        # Check status
        status = self._query_status()
        if 'Alarm' in status:
            print('Alarm detected, unlocking...')
            self.unlock()

        # Follow Kuixiang initialization order
        self._send('G92 X0.0 Y0.0 Z0')  # set current position as origin
        self._send('G21')                 # millimeter units
        self._send('G90')                 # absolute coordinates
        self._send('F8000')               # default rapid speed
        self._send('$1=250')              # motor idle delay 250ms
                                          # 80ms tap is well within range
                                          # auto power-off after 250ms idle, safer than $1=255

        print('=== Setup complete ===\n')

    def unlock(self):
        """Clear alarm lock.

        Uses $X (kill alarm) instead of $H (homing cycle) because
        this pen plotter has no limit switches — $H would run the
        axes into the frame and stall.
        """
        self._send('$X')

    def set_origin(self):
        """Set current position as coordinate origin (move stylus to target first)."""
        self._send('G92 X0.0 Y0.0 Z0')
        print('Origin set to current position')

    # ─── Basic motions ──

    def _pen_down(self):
        """
        Lower stylus. G1G90: always reassert absolute mode to prevent
        Z-axis crushing the screen due to mode errors.
        """
        if self.Z_DOWN is None:
            raise RuntimeError('Z_DOWN not set — run calibration first')
        self._send(f'G1G90 Z{self.Z_DOWN}F{self.Z_SPEED}')

    def _pen_up(self):
        """
        Raise stylus → Kuixiang equivalent: G1G90 Z0.0F20000
        Actively drive Z back to 0 instead of relying on spring,
        keeps GRBL coordinate tracking in sync
        """
        self._send(f'G1G90 Z{self.Z_UP}F{self.Z_SPEED}')

    def _fast_move(self, x, y, speed=8000):
        """
        Rapid move without touching screen (G0). Pen must be up first.
        """
        self._send(f'G0 X{x:.3f}Y{y:.3f}F{speed}')

    def _linear_move(self, x, y, speed=8000):
        """
        Linear move at controlled speed (G1) — used for swipe while pen is down.
        Continuous XY motion keeps resetting $1 timer, Z motor stays powered,
        spring cannot rebound.
        """
        self._send(f'G1 X{x:.3f}Y{y:.3f}F{speed}')


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
            self._send(f'G1 Z{self.Z_DOWN - 0.02:.2f} F500')
            self._send(f'G1 Z{self.Z_DOWN:.2f} F500')
        self._pen_up()


    # ─── Gestures ────────────────────────────────────────────

    def tap(self, duration=0.08):
        """Single tap at current position.
        duration: contact time in seconds. Phone threshold ~50ms, 80ms has margin.
        """
        self._tap_with_vibration(duration)

    def double_tap(self):
        """Double tap at current position (100ms gap between taps)."""
        self._tap_with_vibration(0.08)
        time.sleep(0.1)
        self._tap_with_vibration(0.08)

    def long_press(self, duration=0.8):
        """Long press at current position.
        iOS/Android long press threshold ~500ms, 800ms is safe.
        """
        self._tap_with_vibration(duration)

    SWIPE_DISTANCE = 15  # mm, default swipe length

    def swipe(self, direction, distance=None, speed=6000):
        """Swipe from current position in a cardinal direction.
        direction: 'up', 'down', 'left', 'right'
        distance: mm (defaults to SWIPE_DISTANCE)
        speed: too fast = fling, too slow = long press. 6000mm/min ~ 100mm/s.
        """
        d = distance or self.SWIPE_DISTANCE
        offsets = {
            'up':    (0, -d),
            'down':  (0,  d),
            'left':  (-d, 0),
            'right': ( d, 0),
        }
        dx, dy = offsets[direction]
        self._pen_down()
        self._send(f'G91 G1 X{dx:.3f}Y{dy:.3f}F{speed}')
        self._send('G90')  # restore absolute mode
        self._pen_up()

    def close(self):
        """Close serial port."""
        self.ser.close()
        print('Serial port closed')
