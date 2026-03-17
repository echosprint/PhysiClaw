"""
GRBL pen plotter → phone touch robot
Reverse-engineered from Kuixiang engraving communication logs

Coordinate system:
  - Before powering on, manually align the stylus to the phone's top-left corner,
    then call init() to set the origin
  - X positive = right
  - Y positive = down
  - Z=0  pen up (spring rebound position)
  - Z=5  pen down (touching screen)

Machine parameters (from $$):
  $100=100  X-axis 100 steps/mm, min resolution 0.01mm
  $101=100  Y-axis 100 steps/mm, min resolution 0.01mm
  $102=50   Z-axis 50 steps/mm,  min resolution 0.02mm
  $1=250    motor idle delay 250ms (covers 80ms tap, auto power-off 250ms after idle)
  $110=12000 X-axis max speed
  $111=12000 Y-axis max speed
  $112=10000 Z-axis max speed (Kuixiang sets 20000, truncated to this)

Key findings (from Kuixiang log analysis):
  1. Pen down uses G1G90 Z5.0F20000, pen up uses G1G90 Z0.0F20000
  2. During writing, XY commands are continuous, $1 timer keeps resetting, Z stays powered
  3. Tapping requires a pause; during pause Z powers off after 25ms, spring rebounds
  4. Swiping is same as writing — continuous XY motion, no $1 issue
  5. Kuixiang never uses G4, relies on continuous XY motion to keep Z powered
"""

import serial
import time

from serial_probe import detect_grbl


# ─── Main class ──────────────────────────────────────────────

class GrblDevice:

    # Z-axis parameters
    Z_DOWN = None   # pen down position — must be set by calibration (calibrate.py)
    Z_UP   = 0.0    # pen up position (spring rebound)
    Z_SPEED = 6000  # Z-axis speed — matches human finger tap (~100 mm/s)
                    # Kuixiang used F20000 but GRBL caps at $112=10000;
                    # F6000 is realistic and avoids slamming the screen.

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
        Follows Kuixiang initialization sequence:
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
        """Clear alarm lock."""
        self._send('$X')

    def set_origin(self):
        """Set current position as coordinate origin (move stylus to target first)."""
        self._send('G92 X0.0 Y0.0 Z0')
        print('Origin set to current position')

    # ─── Basic motions (directly mapped to Kuixiang G-code) ──

    def pen_down(self):
        """
        Lower stylus. G1G90: always reassert absolute mode to prevent
        Z-axis crushing the screen due to mode errors.
        """
        if self.Z_DOWN is None:
            raise RuntimeError('Z_DOWN not set — run calibration first')
        self._send(f'G1G90 Z{self.Z_DOWN}F{self.Z_SPEED}')

    def pen_up(self):
        """
        Raise stylus → Kuixiang equivalent: G1G90 Z0.0F20000
        Actively drive Z back to 0 instead of relying on spring,
        keeps GRBL coordinate tracking in sync
        """
        self._send(f'G1G90 Z{self.Z_UP}F{self.Z_SPEED}')

    def move(self, x, y, speed=8000):
        """
        Rapid move without touching screen → Kuixiang: G0 X...Y...F8000
        Must call pen_up() before move(), otherwise it drags across screen
        """
        self._send(f'G0 X{x:.3f}Y{y:.3f}F{speed}')

    def draw(self, x, y, speed=8000):
        """
        Move while touching screen (swipe/write) → Kuixiang: G1 X...Y...F8000
        Continuous XY motion keeps resetting $1 timer, Z motor stays powered,
        spring cannot rebound
        """
        self._send(f'G1 X{x:.3f}Y{y:.3f}F{speed}')

    def home(self):
        """Return to origin → Kuixiang: G90G0 X0Y0"""
        self.pen_up()
        self._send('G90G0 X0Y0')

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
        self.pen_down()
        steps = max(1, int(duration / 0.02))
        for _ in range(steps):
            self._send(f'G1 Z{self.Z_DOWN - 0.02:.2f} F500')
            self._send(f'G1 Z{self.Z_DOWN:.2f} F500')
        self.pen_up()


    # ─── Gestures ────────────────────────────────────────────

    def tap(self, x, y, duration=0.08):
        """
        Single tap.
        duration: contact time in seconds. Phone threshold ~50ms, 80ms has margin.
        """
        self.move(x, y)
        self._tap_with_vibration(duration)

    def double_tap(self, x, y):
        """Double tap (100ms gap between taps)."""
        self.move(x, y)
        self._tap_with_vibration(0.08)
        time.sleep(0.1)
        self._tap_with_vibration(0.08)

    def long_press(self, x, y, duration=0.8):
        """
        Long press (triggers context menu, text selection, etc.)
        iOS/Android long press threshold ~500ms, 800ms is safe.
        """
        self.move(x, y)
        self._tap_with_vibration(duration)

    def swipe(self, x1, y1, x2, y2, speed=6000):
        """
        Swipe gesture.
        Continuous XY motion keeps $1 timer resetting, no spring rebound issue.
        speed: too fast = fling, too slow = long press. 6000mm/min ≈ 100mm/s.
        """
        self.move(x1, y1)
        self.pen_down()
        self.draw(x2, y2, speed=speed)
        self.pen_up()

    def scroll_up(self, x, y, distance=30, speed=3000):
        """
        Scroll up (content moves up, finger swipes bottom to top).
        distance: swipe distance in mm, larger = more scrolling.
        speed: slow to trigger scroll instead of page switch.
        """
        self.swipe(x, y + distance/2, x, y - distance/2, speed=speed)

    def scroll_down(self, x, y, distance=30, speed=3000):
        """Scroll down (content moves down, finger swipes top to bottom)."""
        self.swipe(x, y - distance/2, x, y + distance/2, speed=speed)

    def swipe_left(self, x, y, distance=50, speed=8000):
        """Swipe left (next page / switch app)."""
        self.swipe(x + distance/2, y, x - distance/2, y, speed=speed)

    def swipe_right(self, x, y, distance=50, speed=8000):
        """Swipe right (go back)."""
        self.swipe(x - distance/2, y, x + distance/2, y, speed=speed)

    def swipe_up_from_bottom(self, screen_w, screen_h, speed=6000):
        """Swipe up from bottom (home / open multitask)."""
        x = screen_w / 2
        self.swipe(x, screen_h - 5, x, screen_h * 0.3, speed=speed)

    def swipe_down_from_top(self, screen_w, speed=4000):
        """Swipe down from top (open notification panel)."""
        x = screen_w / 2
        self.swipe(x, 2, x, 40, speed=speed)

    def close(self):
        """Close serial port."""
        self.ser.close()
        print('Serial port closed')
