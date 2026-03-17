"""
PhysiClaw stylus calibration.

Open /pen-calib on the phone, position the stylus just above the center
orange circle, then run:

    uv run python calibrate.py [--camera INDEX]

Phases:
  1. Probe Z depth for tap contact  (center x10)
  2. Scan X offset for right circle  (right  x3)
  3. Scan Y offset for down circle   (down   x3)
  4. Verify long press               (center x3, 800 ms hold)
  5. Verify swipe                    (up / down / right / left)

Results are saved to calibration.json.
"""

import argparse
import json
import sys
import time

import cv2
import numpy as np

from grbl_device import GrblDevice


# ─── Camera ──────────────────────────────────────────────────────

class TopCamera:
    """Keep the top camera open for fast green-screen detection."""

    def __init__(self, index=0):
        self.cap = cv2.VideoCapture(index)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera index {index}")
        # Warmup: discard initial auto-exposure frames
        for _ in range(15):
            self.cap.read()
        print(f"Camera {index} ready")

    def _fresh_frame(self):
        """Flush buffered frames and return the latest one."""
        for _ in range(4):
            self.cap.grab()
        ret, frame = self.cap.read()
        return frame if ret else None

    def is_green(self):
        """Check if the phone screen is showing the green success flash."""
        frame = self._fresh_frame()
        if frame is None:
            return False
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # #22c55e ≈ HSV(145°, 83%, 77%) → OpenCV scale H=72, S=212, V=196
        lower = np.array([35, 50, 50])
        upper = np.array([90, 255, 255])
        mask = cv2.inRange(hsv, lower, upper)
        ratio = np.count_nonzero(mask) / mask.size
        return ratio > 0.15

    def wait_for_green(self, timeout=1.5):
        """Poll for green screen within timeout."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.is_green():
                return True
            time.sleep(0.05)
        return False

    def wait_for_white(self, timeout=3.0):
        """Wait until the green flash clears."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            if not self.is_green():
                return True
            time.sleep(0.1)
        return False

    def close(self):
        self.cap.release()


# ─── Motion helpers ──────────────────────────────────────────────

def tap_once(bot, z_tap, z_speed=5000):
    """Pen down then up at current XY position."""
    bot._send(f'G1G90 Z{z_tap:.2f} F{z_speed}')
    time.sleep(0.15)
    bot._send(f'G1G90 Z0.00 F{z_speed}')


def move_xy(bot, x, y, speed=8000):
    """Rapid move to absolute XY (pen must be up)."""
    bot._send(f'G0 X{x:.2f} Y{y:.2f} F{speed}')
    time.sleep(0.3)


# ─── Phase 1: Z-axis probing ────────────────────────────────────

def phase1_z(bot, cam):
    """Probe Z depth for tap registration, then complete 10 center taps."""
    print("\n" + "=" * 50)
    print("PHASE 1: Z-axis calibration  (tap center ×10)")
    print("=" * 50)

    z_tap = None

    # Probe from 0.5 mm downward in 0.3 mm steps — cautious
    for z_raw in np.arange(0.5, 8.0, 0.3):
        z = round(float(z_raw), 2)
        print(f"  Probe Z={z:.2f} mm …", end=" ", flush=True)
        tap_once(bot, z, z_speed=3000)
        time.sleep(0.3)

        if cam.wait_for_green(timeout=1.0):
            z_tap = round(z + 0.2, 2)          # small margin for reliability
            print(f"CONTACT → using Z={z_tap:.2f} mm")
            break
        else:
            print("no tap")

    if z_tap is None:
        print("\n  FAILED — no contact up to 8 mm. Check stylus alignment.")
        return None

    # Complete remaining 9 taps
    successes = 1
    attempts = 0
    while successes < 10 and attempts < 25:
        cam.wait_for_white()
        time.sleep(0.3)
        print(f"  Tap {successes + 1}/10 …", end=" ", flush=True)
        tap_once(bot, z_tap)
        time.sleep(0.3)
        if cam.wait_for_green(timeout=1.0):
            successes += 1
            print("ok")
        else:
            print("miss")
        attempts += 1

    print(f"\n  Phase 1 done — z_tap = {z_tap} mm  ({successes}/10)")
    return z_tap


# ─── Phase 2: X calibration ─────────────────────────────────────

def phase2_x(bot, cam, z_tap):
    """Scan X offsets to hit the right circle (left:62%, 12 % of screen width)."""
    print("\n" + "=" * 50)
    print("PHASE 2: X calibration  (tap right ×3)")
    print("=" * 50)

    cam.wait_for_white()
    time.sleep(0.5)

    x_offset = None

    # Right circle center is 12 % of viewport width from center.
    # Phone width ≈ 65-75 mm → ~8-9 mm.  Scan 3–18 mm in 1.5 mm steps.
    for x_raw in np.arange(3.0, 18.0, 1.5):
        x = round(float(x_raw), 2)
        print(f"  Try X={x:.1f} mm …", end=" ", flush=True)
        move_xy(bot, x, 0)
        tap_once(bot, z_tap)
        time.sleep(0.3)

        if cam.wait_for_green(timeout=1.0):
            x_offset = x
            print("HIT!")
            break
        else:
            print("miss")

    if x_offset is None:
        print("\n  FAILED — could not hit right circle.")
        return None

    # Complete remaining 2 taps
    successes = 1
    attempts = 0
    while successes < 3 and attempts < 10:
        cam.wait_for_white()
        time.sleep(0.3)
        move_xy(bot, x_offset, 0)
        print(f"  Tap right {successes + 1}/3 …", end=" ", flush=True)
        tap_once(bot, z_tap)
        time.sleep(0.3)
        if cam.wait_for_green(timeout=1.0):
            successes += 1
            print("ok")
        else:
            print("miss")
        attempts += 1

    print(f"\n  Phase 2 done — x_offset = {x_offset} mm (= 12 % screen width)")
    return x_offset


# ─── Phase 3: Y calibration ─────────────────────────────────────

def phase3_y(bot, cam, z_tap):
    """Scan Y offsets to hit the down circle (top:62%, 12 % of screen height)."""
    print("\n" + "=" * 50)
    print("PHASE 3: Y calibration  (tap down ×3)")
    print("=" * 50)

    cam.wait_for_white()
    time.sleep(0.5)

    # Return to center column first
    move_xy(bot, 0, 0)

    y_offset = None

    # Down circle center is 12 % of viewport height from center.
    # Phone height ≈ 130-155 mm → ~16-19 mm.  Scan 6–30 mm in 2 mm steps.
    for y_raw in np.arange(6.0, 30.0, 2.0):
        y = round(float(y_raw), 2)
        print(f"  Try Y={y:.1f} mm …", end=" ", flush=True)
        move_xy(bot, 0, y)
        tap_once(bot, z_tap)
        time.sleep(0.3)

        if cam.wait_for_green(timeout=1.0):
            y_offset = y
            print("HIT!")
            break
        else:
            print("miss")

    if y_offset is None:
        print("\n  FAILED — could not hit down circle.")
        return None

    # Complete remaining 2 taps
    successes = 1
    attempts = 0
    while successes < 3 and attempts < 10:
        cam.wait_for_white()
        time.sleep(0.3)
        move_xy(bot, 0, y_offset)
        print(f"  Tap down {successes + 1}/3 …", end=" ", flush=True)
        tap_once(bot, z_tap)
        time.sleep(0.3)
        if cam.wait_for_green(timeout=1.0):
            successes += 1
            print("ok")
        else:
            print("miss")
        attempts += 1

    print(f"\n  Phase 3 done — y_offset = {y_offset} mm (= 12 % screen height)")
    return y_offset


# ─── Phase 4: Long press ────────────────────────────────────────

def phase4_long_press(bot, cam, z_tap):
    """Long press center ×3  (must hold > 800 ms)."""
    print("\n" + "=" * 50)
    print("PHASE 4: Long press center  (×3, 800 ms hold)")
    print("=" * 50)

    cam.wait_for_white()
    time.sleep(0.5)
    move_xy(bot, 0, 0)

    # Patch Z_DOWN so the existing _tap_with_vibration uses our calibrated depth
    bot.Z_DOWN = z_tap

    successes = 0
    attempts = 0
    while successes < 3 and attempts < 10:
        if successes > 0:
            cam.wait_for_white()
            time.sleep(0.3)
        print(f"  Long press {successes + 1}/3 …", end=" ", flush=True)
        bot._tap_with_vibration(duration=0.9)   # > 800 ms with margin
        time.sleep(0.3)
        if cam.wait_for_green(timeout=1.5):
            successes += 1
            print("ok")
        else:
            print("miss")
        attempts += 1

    print(f"\n  Phase 4 done  ({successes}/3)")


# ─── Phase 5: Swipe ─────────────────────────────────────────────

def phase5_swipe(bot, cam, z_tap, x_offset, y_offset):
    """Swipe in 4 directions from center (up / down / right / left)."""
    print("\n" + "=" * 50)
    print("PHASE 5: Swipe  (up → down → right → left)")
    print("=" * 50)

    cam.wait_for_white()
    time.sleep(0.5)

    # Swipe distances — generous margin over the 60 px threshold.
    # 60 px ≈ 15 % viewport width (X) or ≈ 7 % viewport height (Y).
    sx = round(x_offset * 2.0, 2)     # X swipe distance (mm)
    sy = round(y_offset * 1.2, 2)     # Y swipe distance (mm)
    swipe_speed = 4000                 # mm/min — moderate finger speed

    # (name, end_x, end_y)  — start is always (0, 0) = center
    directions = [
        ("UP",    0,    -sy),
        ("DOWN",  0,     sy),
        ("RIGHT", sx,    0),
        ("LEFT",  -sx,   0),
    ]

    for name, dx, dy in directions:
        cam.wait_for_white()
        time.sleep(0.3)
        move_xy(bot, 0, 0)             # start at center

        success = False
        for attempt in range(4):
            print(f"  Swipe {name} …", end=" ", flush=True)

            # Pen down at center (start of swipe)
            bot._send(f'G1G90 Z{z_tap:.2f} F5000')
            time.sleep(0.1)
            # Drag to destination — G1 at controlled speed
            bot._send(f'G1 X{dx:.2f} Y{dy:.2f} F{swipe_speed}')
            # Wait for motion to complete (distance / speed)
            dist = (dx ** 2 + dy ** 2) ** 0.5
            motion_time = (dist / swipe_speed) * 60   # seconds
            time.sleep(motion_time + 0.1)
            # Pen up
            bot._send(f'G1G90 Z0.00 F5000')
            time.sleep(0.3)

            if cam.wait_for_green(timeout=1.5):
                print("ok")
                success = True
                break
            else:
                print("miss")
                move_xy(bot, 0, 0)     # reset for retry
                time.sleep(0.5)

        if not success:
            print(f"  WARNING: Swipe {name} failed after retries")

        # Return to center for next direction
        bot._send(f'G1G90 Z0.00 F5000')
        time.sleep(0.1)
        move_xy(bot, 0, 0)

    print("\n  Phase 5 done")


# ─── Main ───────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="PhysiClaw stylus calibration")
    parser.add_argument("--camera", type=int, default=0,
                        help="Top-camera device index (default: 0)")
    parser.add_argument("--port", type=str, default=None,
                        help="GRBL serial port (auto-detect if omitted)")
    args = parser.parse_args()

    cam = TopCamera(args.camera)
    bot = GrblDevice(args.port)
    bot.init()

    try:
        # Phase 1 — Z depth
        z_tap = phase1_z(bot, cam)
        if z_tap is None:
            sys.exit(1)

        # Phase 2 — X mapping
        x_offset = phase2_x(bot, cam, z_tap)
        if x_offset is None:
            sys.exit(1)

        # Phase 3 — Y mapping
        y_offset = phase3_y(bot, cam, z_tap)
        if y_offset is None:
            sys.exit(1)

        # Phase 4 — long press verification
        phase4_long_press(bot, cam, z_tap)

        # Phase 5 — swipe verification
        phase5_swipe(bot, cam, z_tap, x_offset, y_offset)

        # ── Save results ─────────────────────────────────────
        result = {
            "z_tap_mm": z_tap,
            "x_12pct_mm": x_offset,
            "y_12pct_mm": y_offset,
            "screen_w_mm": round(x_offset / 0.12, 1),
            "screen_h_mm": round(y_offset / 0.12, 1),
        }
        with open("calibration.json", "w") as f:
            json.dump(result, f, indent=2)

        print("\n" + "=" * 50)
        print("CALIBRATION COMPLETE")
        print("=" * 50)
        print(f"  Z tap depth:    {z_tap} mm")
        print(f"  X per 12%:      {x_offset} mm")
        print(f"  Y per 12%:      {y_offset} mm")
        print(f"  Screen width:   ~{result['screen_w_mm']} mm")
        print(f"  Screen height:  ~{result['screen_h_mm']} mm")
        print(f"  Saved to calibration.json")

    finally:
        bot.pen_up()
        bot.home()
        bot.close()
        cam.close()


if __name__ == "__main__":
    main()
