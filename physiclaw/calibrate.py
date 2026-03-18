"""
Calibration phase functions for PhysiClaw stylus arm.

These are called by PhysiClaw.calibrate() in core.py.
Each phase takes a StylusArm and Camera, runs probing/verification,
and returns results. No orchestration or file I/O here.

Phases (green flash = 1 success, each flash lasts 1 s):
  1. Probe Z depth for tap contact       (10 greens on center)
  2. Probe 4 directions for phone-right  (3 greens on right circle)
  3. Probe 2 directions for phone-down   (3 greens on down circle)
  4. Verify long press                   (3 greens, hold 800 ms)
  5. Verify swipe                        (4 greens: up / down / right / left)
"""

import logging
import time

from physiclaw.camera import Camera
from physiclaw.stylus_arm import StylusArm

log = logging.getLogger(__name__)

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
    arm.wait_idle()


# ─── Phase 1: Z-axis probing ────────────────────────────────────

def phase1_z(stylus_arm: StylusArm, cam: Camera) -> float | None:
    """Probe Z depth for tap registration, then complete 10 center taps."""
    log.info("Phase 1: Z-axis calibration  (tap center x10)")

    z_contact = None

    # Probe from 0.5 mm downward in 0.3 mm steps, max 5 mm.
    # F3000 (50 mm/s) — extra slow during probing to protect the screen.
    for z_raw in [0.5 + i * 0.3 for i in range(16)]:
        z = round(float(z_raw), 2)
        log.debug(f"  Probe Z={z:.2f} mm …")
        tap_once(stylus_arm, z, z_speed=SLOW_Z_SPEED)
        time.sleep(0.3)

        if cam.wait_for_green(timeout=1.0):
            z_contact = z
            log.debug(f"  CONTACT at Z={z_contact:.2f} mm")
            break
        else:
            log.debug(f"  no tap")

    if z_contact is None:
        log.warning("Phase 1 FAILED — no contact up to 5 mm. Check stylus alignment.")
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

        log.debug(f"  Tap {successes + 1}/10 (Z={z_try:.2f}) …")
        tap_once(stylus_arm, z_try, z_speed=SLOW_Z_SPEED)
        time.sleep(0.3)

        if cam.wait_for_green(timeout=1.0):
            successes += 1
            z_hits.append(z_try)
            z_now = z_try
            step = RETREAT_STEP   # hit → retreat next
            log.debug(f"  ok")
        else:
            z_now = z_try
            step = ADVANCE_STEP  # miss → advance next
            log.debug(f"  miss")

        attempts += 1

    # Use the deeper side of hits — average alone sits right at threshold
    # which causes unreliable contact. Bias toward max for sustained contact.
    z_avg = sum(z_hits) / len(z_hits)
    z_max = max(z_hits)
    z_tap = round((z_avg + z_max) / 2, 2)  # midpoint between avg and deepest hit
    log.info(f"Phase 1 done — z_tap = {z_tap} mm  "
             f"(hits: {len(z_hits)}, range {min(z_hits):.2f}-{z_max:.2f})")
    return z_tap


# All four arm directions to probe
SCAN_DIRS = [
    ('X+', 1, 0),
    ('X-', -1, 0),
    ('Y+', 0, 1),
    ('Y-', 0, -1),
]

# Distance candidates (mm) — 12 % of typical phone screen
# Range 3–25.5 mm covers phones from small (width ~37mm) to large (height ~210mm)
SCAN_DISTANCES = [3.0 + i * 1.5 for i in range(16)]


def _probe_direction(stylus_arm: StylusArm, cam: Camera, z_tap: float, directions: list[tuple[str, int, int]]) -> tuple[int, int, float] | None:
    """Try each direction x distance until a green hit.
    Return (axis_sign_x, axis_sign_y, distance_mm) or None.
    """
    for dir_name, ax, ay in directions:
        for dist in SCAN_DISTANCES:
            x = round(ax * dist, 2)
            y = round(ay * dist, 2)
            log.debug(f"  {dir_name} {dist:.1f} mm …")
            move_xy(stylus_arm, x, y)
            tap_once(stylus_arm, z_tap)
            time.sleep(0.3)

            if cam.wait_for_green(timeout=1.0):
                log.debug(f"  HIT!")
                return (ax, ay, dist)
            else:
                log.debug(f"  miss")

        # Return to center before trying next direction
        move_xy(stylus_arm, 0, 0)
        time.sleep(0.2)

    return None


def _repeat_taps(stylus_arm: StylusArm, cam: Camera, z_tap: float, ax: int, ay: int, dist: float, count: int) -> int:
    """Tap the same target (count) more times to confirm."""
    successes = 0
    attempts = 0
    while successes < count and attempts < MAX_RETRIES:
        cam.wait_for_white()
        time.sleep(0.3)
        # Add noise along the scan axis to explore distance variation
        noise = attempts % 5  # 0, 1, 2, 3, 4 mm
        x = round(ax * (dist + noise), 2)
        y = round(ay * (dist + noise), 2)
        move_xy(stylus_arm, x, y)
        log.debug(f"  Confirm {successes + 1}/{count} …")
        tap_once(stylus_arm, z_tap)
        time.sleep(0.3)
        if cam.wait_for_green(timeout=1.0):
            successes += 1
            log.debug(f"  ok")
        else:
            log.debug(f"  miss")
        attempts += 1
    return successes


# ─── Phase 2: find phone-right direction ─────────────────────────

def phase2_right(stylus_arm: StylusArm, cam: Camera, z_tap: float) -> tuple[int, int, float] | None:
    """Probe all 4 arm directions to find which one is phone-right."""
    log.info("Phase 2: Find phone-right direction  (tap right x3)")

    cam.wait_for_white()
    time.sleep(0.5)

    result = _probe_direction(stylus_arm, cam, z_tap, SCAN_DIRS)
    if result is None:
        log.warning("Phase 2 FAILED — could not hit right circle in any direction.")
        return None

    ax, ay, dist = result
    dir_name = 'X+' if ax > 0 else 'X-' if ax < 0 else 'Y+' if ay > 0 else 'Y-'
    log.debug(f"  Phone-right = arm {dir_name}")

    # Confirm with 2 more taps
    _repeat_taps(stylus_arm, cam, z_tap, ax, ay, dist, 2)

    # Return to center
    move_xy(stylus_arm, 0, 0)

    log.info(f"Phase 2 done — right = ({ax}, {ay}) × {dist} mm")
    return (ax, ay, dist)


# ─── Phase 3: find phone-down direction ──────────────────────────

def phase3_down(stylus_arm: StylusArm, cam: Camera, z_tap: float, right_vec: tuple[int, int]) -> tuple[int, int, float] | None:
    """Probe the 2 perpendicular directions to find phone-down."""
    log.info("Phase 3: Find phone-down direction  (tap down x3)")

    cam.wait_for_white()
    time.sleep(0.5)
    move_xy(stylus_arm, 0, 0)

    # Perpendicular to right_vec — only need to check which axis, not the sign
    rax = right_vec[0]
    if rax != 0:
        # Right is along X axis → down must be along Y
        perp_dirs = [('Y+', 0, 1), ('Y-', 0, -1)]
    else:
        # Right is along Y axis → down must be along X
        perp_dirs = [('X+', 1, 0), ('X-', -1, 0)]

    result = _probe_direction(stylus_arm, cam, z_tap, perp_dirs)
    if result is None:
        log.warning("Phase 3 FAILED — could not hit down circle.")
        return None

    ax, ay, dist = result
    dir_name = 'X+' if ax > 0 else 'X-' if ax < 0 else 'Y+' if ay > 0 else 'Y-'
    log.debug(f"  Phone-down = arm {dir_name}")

    # Confirm with 2 more taps
    _repeat_taps(stylus_arm, cam, z_tap, ax, ay, dist, 2)

    # Return to center
    move_xy(stylus_arm, 0, 0)

    log.info(f"Phase 3 done — down = ({ax}, {ay}) × {dist} mm")
    return (ax, ay, dist)


# ─── Phase 4: Long press ────────────────────────────────────────

def phase4_long_press(stylus_arm: StylusArm, cam: Camera) -> None:
    """Long press center x3  (must hold > 800 ms)."""
    log.info("Phase 4: Long press center  (x3, 800 ms hold)")

    cam.wait_for_white()
    time.sleep(0.5)
    move_xy(stylus_arm, 0, 0)

    successes = 0
    attempts = 0
    while successes < 3 and attempts < MAX_RETRIES:
        if successes > 0:
            cam.wait_for_white()
            time.sleep(0.3)
        log.debug(f"  Long press {successes + 1}/3 …")
        stylus_arm.long_press()
        time.sleep(0.3)
        if cam.wait_for_green(timeout=1.5):
            successes += 1
            log.debug(f"  ok")
        else:
            log.debug(f"  miss")
        attempts += 1

    log.info(f"Phase 4 done  ({successes}/3)")


# ─── Phase 5: Swipe ─────────────────────────────────────────────

def phase5_swipe(stylus_arm: StylusArm, cam: Camera) -> None:
    """Swipe in 4 directions from center using the public swipe() API."""
    log.info("Phase 5: Swipe  (up / down / right / left)")

    cam.wait_for_white()
    time.sleep(0.5)

    for direction in ['top', 'bottom', 'right', 'left']:
        cam.wait_for_white()
        time.sleep(0.3)
        move_xy(stylus_arm, 0, 0)

        success = False
        for _ in range(MAX_RETRIES):
            log.debug(f"  Swipe {direction} …")

            stylus_arm.swipe(direction)
            time.sleep(0.5)

            if cam.wait_for_green(timeout=1.5):
                log.debug(f"  ok")
                success = True
                break
            else:
                log.debug(f"  miss")
                move_xy(stylus_arm, 0, 0)
                time.sleep(0.5)

        if not success:
            log.warning(f"  Swipe {direction} failed after retries")

        stylus_arm._pen_up()
        time.sleep(0.1)
        move_xy(stylus_arm, 0, 0)

    log.info("Phase 5 done")
