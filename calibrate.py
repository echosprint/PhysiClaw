"""
PhysiClaw stylus calibration.

Open /pen-calib on the phone, position the stylus just above the center
orange circle, then run:

    uv run python calibrate.py [--camera INDEX]

Note: On macOS, OpenCV won't trigger the camera permission dialog.
If the camera returns blank frames, run `imagesnap` once first to
grant camera access to your terminal app, then re-run this script.

Phases (green flash = 1 success, each flash lasts 1 s):
  1. Probe Z depth for tap contact       (10 greens on center)
  2. Probe 4 directions for phone-right  (3 greens on right circle)
  3. Probe 2 directions for phone-down   (3 greens on down circle)
  4. Verify long press                   (3 greens, hold 800 ms)
  5. Verify swipe                        (4 greens: up / down / right / left)

Results are saved to calibration.json.
"""

import argparse
import json
import sys
import time

from camera import Camera
from stylus_arm import StylusArm


MAX_RETRIES = 20  # max attempts per phase before giving up

# Calibration uses slower Z speed (F5000, 83 mm/s) for safety during probing.
# Human finger tap is ~F6000 (100 mm/s). Capacitive screens register on
# contact, not speed, so slower is fine.
PROBE_Z_SPEED = 5000
SLOW_Z_SPEED = 3000   # extra slow for initial Z probing


def tap_once(arm: StylusArm, z_tap: float, z_speed: int = PROBE_Z_SPEED) -> None:
    """Pen down then up at current XY position."""
    arm._pen_down(z=z_tap, speed=z_speed)
    time.sleep(0.15)
    arm._pen_up()


def move_xy(arm: StylusArm, x: float, y: float) -> None:
    """Rapid move to absolute XY (pen must be up)."""
    arm._fast_move(x, y)
    time.sleep(0.3)


# ─── Phase 1: Z-axis probing ────────────────────────────────────

def phase1_z(stylus_arm: StylusArm, cam: Camera) -> float | None:
    """Probe Z depth for tap registration, then complete 10 center taps."""
    print("\n" + "=" * 50)
    print("PHASE 1: Z-axis calibration  (tap center x10)")
    print("=" * 50)

    z_contact = None

    # Probe from 0.5 mm downward in 0.3 mm steps, max 5 mm.
    # F3000 (50 mm/s) — extra slow during probing to protect the screen.
    for z_raw in [0.5 + i * 0.3 for i in range(16)]:
        z = round(float(z_raw), 2)
        print(f"  Probe Z={z:.2f} mm …", end=" ", flush=True)
        tap_once(stylus_arm, z, z_speed=SLOW_Z_SPEED)
        time.sleep(0.3)

        if cam.wait_for_green(timeout=1.0):
            z_contact = z
            print(f"CONTACT at Z={z_contact:.2f} mm")
            break
        else:
            print("no tap")

    if z_contact is None:
        print("\n  FAILED — no contact up to 5 mm. Check stylus alignment.")
        return None

    # Find the contact boundary by zigzagging:
    #   hit → retreat (smaller Z) until miss → advance (larger Z) until hit → ...
    # Each direction change narrows in on the true contact threshold.
    # Average all contact Z values for a robust result.
    RETREAT_STEP = -0.06  # pull back after hit (larger step, move away fast)
    ADVANCE_STEP = 0.3   # approach after miss (smaller step, careful near screen)

    z_hits = [z_contact]
    z_now = z_contact
    step = RETREAT_STEP   # start by retreating
    successes = 1
    attempts = 0

    while successes < 10 and attempts < 2 * MAX_RETRIES:
        cam.wait_for_white()
        time.sleep(0.3)

        z_try = round(z_now + step, 2)
        if z_try < 0.1:     # don't go above the screen
            step = ADVANCE_STEP
            z_try = round(z_now + step, 2)
        if z_try > 5.0:     # don't press past safe limit
            step = RETREAT_STEP
            z_try = round(z_now + step, 2)

        print(f"  Tap {successes + 1}/10 (Z={z_try:.2f}) …", end=" ", flush=True)
        tap_once(stylus_arm, z_try, z_speed=SLOW_Z_SPEED)
        time.sleep(0.3)

        if cam.wait_for_green(timeout=1.0):
            successes += 1
            z_hits.append(z_try)
            z_now = z_try
            step = RETREAT_STEP   # hit → retreat next
            print("ok")
        else:
            z_now = z_try
            step = ADVANCE_STEP  # miss → advance next
            print("miss")

        attempts += 1

    z_tap = round(sum(z_hits) / len(z_hits), 2)
    print(f"\n  Phase 1 done — z_tap = {z_tap} mm  "
          f"(avg of {len(z_hits)} hits, range {min(z_hits):.2f}-{max(z_hits):.2f})")
    return z_tap


# All four arm directions to probe
SCAN_DIRS = [
    ('X+', 1, 0),
    ('X-', -1, 0),
    ('Y+', 0, 1),
    ('Y-', 0, -1),
]

# Distance candidates (mm) — 12 % of typical phone screen
SCAN_DISTANCES = [3.0 + i * 1.5 for i in range(10)]


def _probe_direction(stylus_arm: StylusArm, cam: Camera, z_tap: float, directions: list[tuple[str, int, int]]) -> tuple[int, int, float] | None:
    """Try each direction x distance until a green hit.
    Return (axis_sign_x, axis_sign_y, distance_mm) or None.
    """
    for dir_name, ax, ay in directions:
        for dist in SCAN_DISTANCES:
            x = round(ax * dist, 2)
            y = round(ay * dist, 2)
            print(f"  {dir_name} {dist:.1f} mm …", end=" ", flush=True)
            move_xy(stylus_arm, x, y)
            tap_once(stylus_arm, z_tap)
            time.sleep(0.3)

            if cam.wait_for_green(timeout=1.0):
                print("HIT!")
                return (ax, ay, dist)
            else:
                print("miss")

        # Return to center before trying next direction
        move_xy(stylus_arm, 0, 0)
        time.sleep(0.2)

    return None


def _repeat_taps(stylus_arm: StylusArm, cam: Camera, z_tap: float, ax: int, ay: int, dist: float, count: int) -> int:
    """Tap the same target (count) more times to confirm."""
    successes = 0
    attempts = 0
    x = round(ax * dist, 2)
    y = round(ay * dist, 2)
    while successes < count and attempts < MAX_RETRIES:
        cam.wait_for_white()
        time.sleep(0.3)
        move_xy(stylus_arm, x, y)
        print(f"  Confirm {successes + 1}/{count} …", end=" ", flush=True)
        tap_once(stylus_arm, z_tap)
        time.sleep(0.3)
        if cam.wait_for_green(timeout=1.0):
            successes += 1
            print("ok")
        else:
            print("miss")
        attempts += 1
    return successes


# ─── Phase 2: find phone-right direction ─────────────────────────

def phase2_right(stylus_arm: StylusArm, cam: Camera, z_tap: float) -> tuple[int, int, float] | None:
    """Probe all 4 arm directions to find which one is phone-right."""
    print("\n" + "=" * 50)
    print("PHASE 2: Find phone-right direction  (tap right ×3)")
    print("=" * 50)

    cam.wait_for_white()
    time.sleep(0.5)

    result = _probe_direction(stylus_arm, cam, z_tap, SCAN_DIRS)
    if result is None:
        print("\n  FAILED — could not hit right circle in any direction.")
        return None

    ax, ay, dist = result
    print(f"  Phone-right = arm ({'X+' if ax > 0 else 'X-' if ax < 0 else 'Y+' if ay > 0 else 'Y-'})")

    # Confirm with 2 more taps
    _repeat_taps(stylus_arm, cam, z_tap, ax, ay, dist, 2)

    # Return to center
    move_xy(stylus_arm, 0, 0)

    print(f"\n  Phase 2 done — right = ({ax}, {ay}) × {dist} mm")
    return (ax, ay, dist)


# ─── Phase 3: find phone-down direction ──────────────────────────

def phase3_down(stylus_arm: StylusArm, cam: Camera, z_tap: float, right_vec: tuple[int, int]) -> tuple[int, int, float] | None:
    """Probe the 2 perpendicular directions to find phone-down."""
    print("\n" + "=" * 50)
    print("PHASE 3: Find phone-down direction  (tap down ×3)")
    print("=" * 50)

    cam.wait_for_white()
    time.sleep(0.5)
    move_xy(stylus_arm, 0, 0)

    # Perpendicular to right_vec
    rax, ray = right_vec
    if rax != 0:
        # Right is along X axis → down must be along Y
        perp_dirs = [('Y+', 0, 1), ('Y-', 0, -1)]
    else:
        # Right is along Y axis → down must be along X
        perp_dirs = [('X+', 1, 0), ('X-', -1, 0)]

    result = _probe_direction(stylus_arm, cam, z_tap, perp_dirs)
    if result is None:
        print("\n  FAILED — could not hit down circle.")
        return None

    ax, ay, dist = result
    print(f"  Phone-down = arm ({'X+' if ax > 0 else 'X-' if ax < 0 else 'Y+' if ay > 0 else 'Y-'})")

    # Confirm with 2 more taps
    _repeat_taps(stylus_arm, cam, z_tap, ax, ay, dist, 2)

    # Return to center
    move_xy(stylus_arm, 0, 0)

    print(f"\n  Phase 3 done — down = ({ax}, {ay}) × {dist} mm")
    return (ax, ay, dist)


# ─── Phase 4: Long press ────────────────────────────────────────

def phase4_long_press(stylus_arm: StylusArm, cam: Camera, z_tap: float) -> None:
    """Long press center ×3  (must hold > 800 ms)."""
    print("\n" + "=" * 50)
    print("PHASE 4: Long press center  (×3, 800 ms hold)")
    print("=" * 50)

    cam.wait_for_white()
    time.sleep(0.5)
    move_xy(stylus_arm, 0, 0)

    # Patch Z_DOWN so the existing _tap_with_vibration uses our calibrated depth
    stylus_arm.Z_DOWN = z_tap

    successes = 0
    attempts = 0
    while successes < 3 and attempts < MAX_RETRIES:
        if successes > 0:
            cam.wait_for_white()
            time.sleep(0.3)
        print(f"  Long press {successes + 1}/3 …", end=" ", flush=True)
        stylus_arm._tap_with_vibration(duration=0.9)   # > 800 ms with margin
        time.sleep(0.3)
        if cam.wait_for_green(timeout=1.5):
            successes += 1
            print("ok")
        else:
            print("miss")
        attempts += 1

    print(f"\n  Phase 4 done  ({successes}/3)")


# ─── Phase 5: Swipe ─────────────────────────────────────────────

def phase5_swipe(stylus_arm: StylusArm, cam: Camera, z_tap: float, right_result: tuple[int, int, float], down_result: tuple[int, int, float]) -> None:
    """Swipe in 4 directions from center (up / down / right / left)."""
    print("\n" + "=" * 50)
    print("PHASE 5: Swipe  (up → down → right → left)")
    print("=" * 50)

    cam.wait_for_white()
    time.sleep(0.5)

    rax, ray, r_dist = right_result
    dax, day, d_dist = down_result
    swipe_speed = 4000
    # Swipe distance: generous margin over 60 px threshold
    sr = round(r_dist * 2.0, 2)
    sd = round(d_dist * 1.2, 2)

    # Map phone directions to arm (dx, dy)
    directions = [
        ("UP",    -dax * sd, -day * sd),  # opposite of down
        ("DOWN",   dax * sd,  day * sd),
        ("RIGHT",  rax * sr,  ray * sr),
        ("LEFT",  -rax * sr, -ray * sr),  # opposite of right
    ]

    for name, dx, dy in directions:
        cam.wait_for_white()
        time.sleep(0.3)
        move_xy(stylus_arm, 0, 0)

        success = False
        for _ in range(MAX_RETRIES):
            print(f"  Swipe {name} …", end=" ", flush=True)

            stylus_arm._pen_down(z=z_tap)
            time.sleep(0.1)
            stylus_arm._linear_move(dx, dy, speed=swipe_speed)
            dist = (dx ** 2 + dy ** 2) ** 0.5
            motion_time = (dist / swipe_speed) * 60
            time.sleep(motion_time + 0.1)
            stylus_arm._pen_up()
            time.sleep(0.3)

            if cam.wait_for_green(timeout=1.5):
                print("ok")
                success = True
                break
            else:
                print("miss")
                move_xy(stylus_arm, 0, 0)
                time.sleep(0.5)

        if not success:
            print(f"  WARNING: Swipe {name} failed after retries")

        stylus_arm._pen_up()
        time.sleep(0.1)
        move_xy(stylus_arm, 0, 0)

    print("\n  Phase 5 done")


# ─── Main ───────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="PhysiClaw stylus calibration")
    parser.add_argument("--camera", type=int, default=0,
                        help="Top-camera device index (default: 0)")
    parser.add_argument("--port", type=str, default=None,
                        help="GRBL serial port (auto-detect if omitted)")
    args = parser.parse_args()

    cam = Camera(args.camera)
    stylus_arm = StylusArm(args.port)
    stylus_arm.setup()

    try:
        # Phase 1 — Z depth
        z_tap = phase1_z(stylus_arm, cam)
        if z_tap is None:
            sys.exit(1)

        # Phase 2 — find phone-right direction
        right_result = phase2_right(stylus_arm, cam, z_tap)
        if right_result is None:
            sys.exit(1)

        # Phase 3 — find phone-down direction
        right_vec = (right_result[0], right_result[1])
        down_result = phase3_down(stylus_arm, cam, z_tap, right_vec)
        if down_result is None:
            sys.exit(1)

        # Phase 4 — long press verification
        phase4_long_press(stylus_arm, cam, z_tap)

        # Phase 5 — swipe verification
        phase5_swipe(stylus_arm, cam, z_tap, right_result, down_result)

        # ── Save results ─────────────────────────────────────
        rax, ray, r_dist = right_result
        dax, day, d_dist = down_result
        result = {
            "z_tap_mm": z_tap,
            "right_vec": [rax, ray],
            "right_dist_mm": r_dist,
            "down_vec": [dax, day],
            "down_dist_mm": d_dist,
            "screen_w_mm": round(r_dist / 0.12, 1),
            "screen_h_mm": round(d_dist / 0.12, 1),
        }
        with open("calibration.json", "w") as f:
            json.dump(result, f, indent=2)

        print("\n" + "=" * 50)
        print("CALIBRATION COMPLETE")
        print("=" * 50)
        print(f"  Z tap depth:    {z_tap} mm")
        print(f"  Phone right:    arm ({rax}, {ray}) × {r_dist} mm")
        print(f"  Phone down:     arm ({dax}, {day}) × {d_dist} mm")
        print(f"  Screen width:   ~{result['screen_w_mm']} mm")
        print(f"  Screen height:  ~{result['screen_h_mm']} mm")
        print(f"  Saved to calibration.json")

    finally:
        stylus_arm._pen_up()
        stylus_arm._fast_move(0, 0)
        stylus_arm.close()
        cam.close()


if __name__ == "__main__":
    main()
