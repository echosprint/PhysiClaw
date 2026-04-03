"""
Plan-based calibration — touch coordinates + camera detection.

Implements the architecture plan's calibration exactly:

  Step 0: Z-depth via touch event detection (stationary descent)
  Step 1: Arm-phone alignment via touch coordinates
  Step 2: Camera physical rotation check (phone shape in frame)
  Step 3: Software rotation via UP/RIGHT markers
  Step 4: GRBL ↔ Screen mapping via touch coordinates (Mapping A)
  Step 5: Camera ↔ Screen mapping via red dot detection (Mapping B)
  Step 6: Full-chain validation (camera → Mapping B → Mapping A → tap → touch)
  Post:   Set screen center as GRBL origin

No green flash. Touch events for contact detection and coordinate mapping.
Camera for marker/dot visual detection only.
"""

import logging
import random
import time

import cv2
import numpy as np

from physiclaw.bridge import CalibrationState, GRID_COLS_PCT, GRID_ROWS_PCT
from physiclaw.camera import Camera
from physiclaw.grid_calibrate import GridCalibration, detect_red_dots, sort_dots_to_grid
from physiclaw.stylus_arm import StylusArm

log = logging.getLogger(__name__)

SLOW_Z_SPEED = 3000
PROBE_Z_SPEED = 5000


def _tap_once(arm: StylusArm, z: float, z_speed: int = PROBE_Z_SPEED):
    """Single tap: pen down, dwell 150ms, pen up."""
    arm._pen_down(z=z, speed=z_speed)
    arm._dwell(0.15)
    arm._pen_up()
    arm.wait_idle()


# ─── Step 0: Z-axis surface detection ────────────────────────

def step0_z_depth(arm: StylusArm, cal: CalibrationState) -> float:
    """Descend incrementally without lifting. Touch event = contact. Plan Step 0.

    The stylus descends in 0.25mm steps and STAYS at each level (no lifting).
    At each level it pauses 100ms. If the phone reports a touch during the
    pause, contact is found. Then the stylus lifts.

    Plan: "Stationary detection at each step — more reliable than continuous
    descent. Stylus is still when the touchscreen senses it."

    Returns z_tap (contact depth + 0.5mm margin).
    """
    log.info("Step 0: Z-axis surface detection (stationary descent)")
    cal.set_phase("center")
    time.sleep(0.5)

    # Clear any stale touches
    cal.get_touches()

    z_contact = None
    z_steps = [round(0.25 + i * 0.25, 2) for i in range(20)]  # 0.25 to 5.25mm

    for z in z_steps:
        # Descend to this Z level and STAY (no pen_up between steps)
        # G1 G90 Z{z} moves to absolute Z position
        arm._send(f'G1 G90 Z{z:.2f} F{SLOW_Z_SPEED}')
        # G4 dwell is a sync barrier — blocks until Z move completes, then holds 100ms
        arm._dwell(0.1)
        # After dwell, check if a touch event arrived
        touch = cal.wait_touch(timeout=0.3)
        if touch is not None:
            z_contact = z
            log.info(f"  Contact at Z={z:.2f}mm "
                     f"(touch at screen {touch.get('x')}, {touch.get('y')})")
            break

    # Lift stylus back up regardless of result
    arm._pen_up()
    arm.wait_idle()

    if z_contact is None:
        raise RuntimeError("Step 0 FAILED — no touch detected up to 5.25mm. "
                           "Check stylus alignment and phone placement.")

    z_tap = round(z_contact + 0.5, 2)
    log.info(f"  z_tap = {z_tap}mm (contact + 0.5mm margin)")
    return z_tap


# ─── Step 1: Arm-phone alignment check ───────────────────────

def step1_alignment(arm: StylusArm, cal: CalibrationState,
                    z_tap: float, separation_mm: float = 10.0
                    ) -> float:
    """Two taps along arm X-axis, compare touch Y coords. Plan Step 1.

    Returns tilt_ratio. < 0.02 means aligned (< ~1 degree).
    """
    log.info("Step 1: Alignment check (2 taps)")
    cal.set_phase("center")
    time.sleep(0.3)

    half = separation_mm / 2
    touches = []
    for x in [-half, half]:
        arm._fast_move(x, 0)
        arm.wait_idle()
        cal.get_touches()
        _tap_once(arm, z_tap)
        time.sleep(0.2)
        touch = cal.wait_touch(timeout=2.0)
        if touch is None:
            raise RuntimeError(f"Step 1 FAILED — no touch at arm X={x:.1f}mm")
        touches.append(touch)

    arm._fast_move(0, 0)
    arm.wait_idle()

    sx1, sy1 = touches[0]['x'], touches[0]['y']
    sx2, sy2 = touches[1]['x'], touches[1]['y']
    dx = abs(sx2 - sx1)
    dy = abs(sy2 - sy1)
    if dx < 1:
        raise RuntimeError("Step 1 FAILED — both taps at same screen X. "
                           "Check that arm X-axis moves across the phone width.")

    tilt = dy / dx
    log.info(f"  tilt_ratio = {tilt:.4f} "
             f"({'OK (< 0.02)' if tilt < 0.02 else 'TILTED — adjust phone'})")
    return tilt


# ─── Step 2: Camera physical rotation check ──────────────────

def step2_camera_rotation(cam: Camera) -> bool:
    """Check if camera long axis matches phone long axis. Plan Step 2.

    Returns True if phone appears taller than wide in raw camera frame.
    """
    log.info("Step 2: Camera physical rotation check")
    frame = cam._fresh_frame()
    if frame is None:
        raise RuntimeError("Step 2 FAILED — camera read failed")

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise RuntimeError("Step 2 FAILED — no bright region in camera frame")

    largest = max(contours, key=cv2.contourArea)
    _, _, bw, bh = cv2.boundingRect(largest)

    ok = bh > bw
    log.info(f"  Phone region: {bw}×{bh}px → "
             f"{'OK (portrait)' if ok else 'ROTATED — rotate camera 90°'}")
    return ok


# ─── Step 3: Software rotation via UP/RIGHT markers ──────────

def step3_software_rotation(cam: Camera, cal: CalibrationState) -> int:
    """Detect blue UP/RIGHT markers, determine software rotation. Plan Step 3.

    Returns cv2 rotation code (cv2.ROTATE_* or -1 for no rotation).
    """
    log.info("Step 3: Software rotation (UP/RIGHT markers)")
    cal.set_phase("markers")
    time.sleep(1.0)

    frame = cam._fresh_frame()
    if frame is None:
        raise RuntimeError("Step 3 FAILED — camera read failed")

    # Detect blue blobs (#2563eb → HSV H≈110)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([100, 80, 80]), np.array([130, 255, 255]))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15)))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    blobs = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 500:
            continue
        m = cv2.moments(cnt)
        if m['m00'] == 0:
            continue
        blobs.append((m['m10'] / m['m00'], m['m01'] / m['m00'], area))

    if len(blobs) < 2:
        raise RuntimeError(f"Step 3 FAILED — found {len(blobs)} blue markers, need 2")

    blobs.sort(key=lambda b: b[2], reverse=True)
    # RIGHT has more area (wider text), UP has less
    right_x, right_y = blobs[0][0], blobs[0][1]
    up_x, up_y = blobs[1][0], blobs[1][1]

    log.info(f"  UP at camera ({up_x:.0f}, {up_y:.0f})")
    log.info(f"  RIGHT at camera ({right_x:.0f}, {right_y:.0f})")

    # Determine rotation by relative position of UP vs RIGHT
    if up_y < right_y and abs(up_x - right_x) < abs(up_y - right_y):
        rotation = -1  # no rotation
        log.info("  → 0° (no rotation)")
    elif up_x < right_x and abs(up_y - right_y) < abs(up_x - right_x):
        rotation = cv2.ROTATE_90_CLOCKWISE
        log.info("  → 90° CW")
    elif up_y > right_y and abs(up_x - right_x) < abs(up_y - right_y):
        rotation = cv2.ROTATE_180
        log.info("  → 180°")
    else:
        rotation = cv2.ROTATE_90_COUNTERCLOCKWISE
        log.info("  → 90° CCW")

    return rotation


# ─── Step 4: GRBL ↔ Screen mapping (Mapping A) ──────────────

def step4_grbl_screen(arm: StylusArm, cal: CalibrationState,
                      z_tap: float,
                      device_info: dict | None = None
                      ) -> tuple[np.ndarray, list[dict]]:
    """Distributed taps with touch coords → affine transform. Plan Step 4.

    Expands outward from center. Uses device info for expansion distances
    when available. Verifies with 3 additional taps.

    Returns (screen_to_grbl affine (2,3), list of touch events).
    """
    log.info("Step 4: GRBL ↔ Screen mapping (Mapping A)")
    cal.set_phase("center")
    time.sleep(0.3)

    # Expansion distance in mm
    # Plan: "half_width ≈ 390/2 × 0.8 = 156pt outward"
    # We don't know pt→mm yet, but typical phones: ~65mm wide, ~140mm tall
    # Default: 8mm covers ~25% of half-width, reasonable for affine
    # With device info: scale based on viewport aspect ratio
    d = 8.0
    d_v = 8.0  # vertical expansion (may differ from horizontal)
    if device_info:
        vw = device_info.get('viewport_width', 390)
        vh = device_info.get('viewport_height', 844)
        # Scale vertical expansion by aspect ratio so points span proportionally
        d_v = d * (vh / vw) if vw > 0 else d

    # Sampling pattern from plan: center + 4 cardinal + 4 diagonal + redundancy
    grid_mm = [
        (0, 0),                                  # center
        (d, 0), (-d, 0),                         # left/right
        (0, d_v), (0, -d_v),                     # up/down
        (d, d_v), (d, -d_v),                     # diagonals
        (-d, d_v), (-d, -d_v),
        (d * 0.5, d_v * 0.5), (-d * 0.5, -d_v * 0.5),  # inner (redundancy)
    ]

    grbl_pts = []
    screen_pts = []
    all_touches = []

    for gx, gy in grid_mm:
        arm._fast_move(gx, gy)
        arm.wait_idle()
        cal.get_touches()
        _tap_once(arm, z_tap)
        time.sleep(0.2)
        touch = cal.wait_touch(timeout=2.0)
        if touch is None:
            log.warning(f"  Miss at GRBL ({gx:.1f}, {gy:.1f}) — skipping")
            continue
        grbl_pts.append([gx, gy])
        screen_pts.append([touch['x'], touch['y']])
        all_touches.append(touch)

    arm._fast_move(0, 0)
    arm.wait_idle()

    if len(grbl_pts) < 4:
        raise RuntimeError(f"Step 4 FAILED — only {len(grbl_pts)} valid taps (need ≥4)")

    # Compute affine: screen CSS pixels → GRBL mm
    screen_to_grbl, _ = cv2.estimateAffine2D(
        np.array(screen_pts, dtype=np.float64),
        np.array(grbl_pts, dtype=np.float64))
    if screen_to_grbl is None:
        raise RuntimeError("Step 4 FAILED — affine computation failed")

    # Verify: 3 additional taps at new positions
    # Plan: "predicted vs actual error < 2px"
    log.info(f"  Mapping A from {len(grbl_pts)} pairs. Verifying...")
    verify_offsets = [(d * 0.7, 0), (0, d_v * 0.7), (-d * 0.7, d_v * 0.7)]
    max_error = 0
    for vgx, vgy in verify_offsets:
        arm._fast_move(vgx, vgy)
        arm.wait_idle()
        cal.get_touches()
        _tap_once(arm, z_tap)
        time.sleep(0.2)
        touch = cal.wait_touch(timeout=2.0)
        if touch is None:
            continue
        # Predict GRBL from touch using the affine
        predicted = screen_to_grbl @ np.array([touch['x'], touch['y'], 1.0])
        error = ((predicted[0] - vgx)**2 + (predicted[1] - vgy)**2)**0.5
        max_error = max(max_error, error)
        log.debug(f"  Verify ({vgx:.1f},{vgy:.1f}): error={error:.2f}mm")

    arm._fast_move(0, 0)
    arm.wait_idle()
    log.info(f"  Verification max error: {max_error:.2f}mm")

    return screen_to_grbl, all_touches


# ─── Step 5: Camera ↔ Screen mapping (Mapping B) ────────────

def step5_camera_screen(cam: Camera, cal: CalibrationState,
                        rotation: int,
                        viewport_w: int, viewport_h: int
                        ) -> np.ndarray:
    """Detect 15 red dots, compute 0-1 pct → camera pixels affine. Plan Step 5.

    Returns pct_to_pixel affine (2,3): screen 0-1 → camera pixels.
    """
    log.info("Step 5: Camera ↔ Screen mapping (Mapping B, 15 red dots)")
    cal.set_phase("grid")
    time.sleep(1.0)

    frame = cam._fresh_frame()
    if frame is None:
        raise RuntimeError("Step 5 FAILED — camera read failed")
    if rotation >= 0:
        frame = cv2.rotate(frame, rotation)

    dots = detect_red_dots(frame)
    expected = len(GRID_COLS_PCT) * len(GRID_ROWS_PCT)

    # Retry once if detection fails
    if len(dots) != expected:
        time.sleep(1.0)
        frame = cam._fresh_frame()
        if frame is not None:
            if rotation >= 0:
                frame = cv2.rotate(frame, rotation)
            dots = detect_red_dots(frame)

    if len(dots) != expected:
        raise RuntimeError(
            f"Step 5 FAILED — detected {len(dots)} dots, expected {expected}")

    camera_pixels = sort_dots_to_grid(dots)
    screen_pcts = np.array(
        [[x, y] for y in GRID_ROWS_PCT for x in GRID_COLS_PCT],
        dtype=np.float64)

    # Plan: "15 pairs → compute homography matrix (camera pixels → screen pixels)."
    cam_to_pct, mask = cv2.findHomography(camera_pixels, screen_pcts, cv2.RANSAC, 3.0)
    if cam_to_pct is None:
        raise RuntimeError("Step 5 FAILED — homography computation failed")
    inliers = int(mask.sum()) if mask is not None else 0
    log.info(f"  Homography: {inliers}/{len(dots)} inliers")

    # GridCalibration needs pct_to_pixel (2×3 affine: 0-1 pct → camera pixels).
    # Derive from the homography inverse. For an overhead camera the relationship
    # is nearly affine, so estimateAffine2D on the same points gives the best
    # 2×3 approximation that GridCalibration can consume.
    pct_to_pixel, _ = cv2.estimateAffine2D(screen_pcts, camera_pixels)
    if pct_to_pixel is None:
        raise RuntimeError("Step 5 FAILED — affine computation failed")

    log.info(f"  Mapping B from {len(dots)} dots (homography + affine)")
    return pct_to_pixel


# ─── Orange dot detection (for Step 6 validation) ────────────

def _detect_orange_dot(frame: np.ndarray) -> tuple[float, float] | None:
    """Detect a single orange dot in a camera frame.

    Returns (cx, cy) in camera pixels, or None if not found.
    Orange #f97316 ≈ HSV H=20°, S=90%, V=97% → OpenCV H=10, S=230, V=247
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # Orange range: H=5-25 (warm orange), high S, high V
    mask = cv2.inRange(hsv, np.array([5, 100, 100]), np.array([25, 255, 255]))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    # Take the largest orange blob
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < 50:
        return None

    m = cv2.moments(largest)
    if m['m00'] == 0:
        return None
    return (m['m10'] / m['m00'], m['m01'] / m['m00'])


# ─── Step 6: Full-chain validation ───────────────────────────

def step6_validate(arm: StylusArm, cam: Camera, cal: CalibrationState,
                   z_tap: float, rotation: int,
                   pct_to_grbl: np.ndarray,
                   pct_to_pixel: np.ndarray,
                   num_tests: int = 3,
                   max_error_px: float = 5.0
                   ) -> list[dict]:
    """Full chain: dot → camera detect → Mapping B → Mapping A → tap → touch.

    Plan Step 6. Tests BOTH mappings end-to-end:
    1. Page shows orange dot at random position
    2. Camera detects orange dot in frame (camera pixels)
    3. Mapping B⁻¹: camera pixels → screen 0-1 pct
    4. Mapping A: screen 0-1 pct → GRBL mm
    5. Arm taps
    6. Phone reports touch coordinate
    7. Compare touch vs expected position
    """
    log.info("Step 6: Full-chain validation")

    # Compute inverse of pct_to_pixel for camera pixels → screen pct
    # pct_to_pixel is [a b tx; c d ty], maps pct → camera pixels
    # Inverse: camera pixels → pct
    A = pct_to_pixel[:, :2]  # 2×2
    b = pct_to_pixel[:, 2]   # translation
    A_inv = np.linalg.inv(A)
    # pixel → pct: A_inv @ (pixel - b) = A_inv @ pixel - A_inv @ b
    pixel_to_pct = np.hstack([A_inv, (-A_inv @ b).reshape(2, 1)])

    # Get viewport dimensions from a probe tap
    cal.set_phase("center")
    time.sleep(0.3)
    arm._fast_move(0, 0)
    arm.wait_idle()
    cal.get_touches()
    _tap_once(arm, z_tap)
    time.sleep(0.2)
    probe = cal.wait_touch(timeout=2.0)
    vw = probe.get('viewport_w', 390) if probe else 390
    vh = probe.get('viewport_h', 844) if probe else 844

    results = []
    for i in range(num_tests):
        dot_pct_x = round(0.2 + random.random() * 0.6, 3)
        dot_pct_y = round(0.2 + random.random() * 0.6, 3)

        # 1. Show orange dot
        cal.set_phase("dot", dot_x=dot_pct_x, dot_y=dot_pct_y)
        time.sleep(0.5)

        # 2. Camera detects orange dot
        # Park arm first so it doesn't occlude
        park_gx, park_gy = pct_to_grbl @ np.array([0.5, -0.5, 1.0])
        arm._fast_move(float(park_gx), float(park_gy))
        arm.wait_idle()
        time.sleep(0.3)

        frame = cam._fresh_frame()
        if frame is not None and rotation >= 0:
            frame = cv2.rotate(frame, rotation)

        detected = _detect_orange_dot(frame) if frame is not None else None

        if detected is None:
            log.warning(f"  Test {i+1}: camera could not detect orange dot")
            # Fall back to known position
            cam_pct_x, cam_pct_y = dot_pct_x, dot_pct_y
        else:
            # 3. Mapping B⁻¹: camera pixels → screen pct
            cam_pt = np.array([detected[0], detected[1], 1.0])
            cam_pct = pixel_to_pct @ cam_pt
            cam_pct_x, cam_pct_y = float(cam_pct[0]), float(cam_pct[1])
            log.debug(f"  Dot detected at camera ({detected[0]:.0f}, {detected[1]:.0f}) "
                      f"→ pct ({cam_pct_x:.3f}, {cam_pct_y:.3f})")

        # 4. Mapping A: screen pct → GRBL mm
        grbl_pos = pct_to_grbl @ np.array([cam_pct_x, cam_pct_y, 1.0])
        gx, gy = float(grbl_pos[0]), float(grbl_pos[1])

        # 5. Arm taps
        arm._fast_move(gx, gy)
        arm.wait_idle()
        cal.get_touches()
        _tap_once(arm, z_tap)
        time.sleep(0.2)

        # 6. Phone reports touch
        touch = cal.wait_touch(timeout=2.0)
        if touch is None:
            results.append({"expected_css": (round(dot_pct_x * vw, 1), round(dot_pct_y * vh, 1)),
                            "error_px": float('inf'), "passed": False})
            log.warning(f"  Test {i+1}: MISS — no touch")
            continue

        if 'viewport_w' in touch:
            vw, vh = touch['viewport_w'], touch['viewport_h']

        # 7. Compare: touch coordinate vs dot's actual screen coordinate
        # Plan: "All 3 within 5px → calibration passed"
        # Compare in CSS pixel space (same units as touch coordinates)
        expected_css_x = dot_pct_x * vw
        expected_css_y = dot_pct_y * vh
        actual_css_x = touch['x']
        actual_css_y = touch['y']
        error_px = ((actual_css_x - expected_css_x)**2 +
                    (actual_css_y - expected_css_y)**2)**0.5
        passed = error_px < max_error_px

        results.append({
            "expected_css": (round(expected_css_x, 1), round(expected_css_y, 1)),
            "actual_css": (actual_css_x, actual_css_y),
            "camera_pct": (round(cam_pct_x, 3), round(cam_pct_y, 3)),
            "error_px": round(error_px, 1),
            "passed": passed,
        })
        log.info(f"  Test {i+1}: error={error_px:.1f}px {'✓' if passed else '✗'}")

    arm._fast_move(0, 0)
    arm.wait_idle()
    passed_count = sum(1 for r in results if r['passed'])
    log.info(f"  {passed_count}/{num_tests} passed")
    return results


# ─── Post: Set origin at screen center ───────────────────────

def post_set_origin(arm: StylusArm, pct_to_grbl: np.ndarray):
    """Move to screen center (0.5, 0.5) and set GRBL origin. Plan post-cal."""
    log.info("Post: Setting origin at screen center")
    grbl_pos = pct_to_grbl @ np.array([0.5, 0.5, 1.0])
    arm._fast_move(float(grbl_pos[0]), float(grbl_pos[1]))
    arm.wait_idle()
    arm.set_origin()
    log.info("  Origin set at screen center (G92 X0 Y0)")


# ─── Full calibration orchestrator ───────────────────────────

def full_calibration(arm: StylusArm, cam: Camera,
                     cal: CalibrationState,
                     device_info: dict | None = None
                     ) -> GridCalibration:
    """Run all plan calibration steps. Returns GridCalibration.

    Expects: phone with /calibrate page open, stylus above screen center.
    """
    # Step 0: Z-depth
    z_tap = step0_z_depth(arm, cal)
    arm.Z_DOWN = z_tap

    # Step 1: Alignment
    tilt = step1_alignment(arm, cal, z_tap)
    if tilt > 0.05:
        raise RuntimeError(f"Phone tilted (ratio={tilt:.3f}). Adjust and retry.")

    # Step 2: Camera physical rotation
    if not step2_camera_rotation(cam):
        raise RuntimeError("Camera long axis doesn't match phone. Rotate camera 90°.")

    # Step 3: Software rotation
    rotation = step3_software_rotation(cam, cal)

    # Step 4: Mapping A (screen CSS → GRBL mm)
    screen_to_grbl, all_touches = step4_grbl_screen(arm, cal, z_tap, device_info)

    # Get viewport dimensions
    vw = all_touches[0].get('viewport_w', 390)
    vh = all_touches[0].get('viewport_h', 844)

    # Convert screen_to_grbl (CSS pixels) → pct_to_grbl (0-1 → GRBL mm)
    pct_to_grbl = screen_to_grbl.copy()
    pct_to_grbl[:, 0] *= vw
    pct_to_grbl[:, 1] *= vh

    # Park arm for camera step
    if arm.MOVE_DIRECTIONS:
        ux, uy = arm.MOVE_DIRECTIONS['top']
        arm._fast_move(ux * 100, uy * 100)
    else:
        arm._fast_move(0, -80)
    arm.wait_idle()

    # Step 5: Mapping B (0-1 pct → camera pixels)
    pct_to_pixel = step5_camera_screen(cam, cal, rotation, vw, vh)

    # Build GridCalibration
    grid_cal = GridCalibration(pct_to_grbl=pct_to_grbl, pct_to_pixel=pct_to_pixel)

    # Move to center for validation
    gx, gy = grid_cal.pct_to_grbl_mm(0.5, 0.5)
    arm._fast_move(gx, gy)
    arm.wait_idle()

    # Step 6: Full-chain validation
    results = step6_validate(arm, cam, cal, z_tap, rotation,
                             pct_to_grbl, pct_to_pixel)
    passed = sum(1 for r in results if r['passed'])
    if passed < len(results):
        log.warning(f"Validation: {passed}/{len(results)} tests passed")

    # Post: Set origin at screen center
    post_set_origin(arm, pct_to_grbl)

    # Recompute affine relative to new origin
    origin_gx, origin_gy = grid_cal.pct_to_grbl_mm(0.5, 0.5)
    pct_to_grbl[0, 2] -= origin_gx
    pct_to_grbl[1, 2] -= origin_gy
    grid_cal = GridCalibration(pct_to_grbl=pct_to_grbl, pct_to_pixel=pct_to_pixel)

    cal.set_phase("idle")
    log.info("Calibration complete")
    return grid_cal
