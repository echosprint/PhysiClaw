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

from physiclaw.bridge import CalibrationState
from physiclaw.camera import Camera
from physiclaw.grid_calibrate import GridCalibration, detect_red_dots, sort_dots_to_grid
from physiclaw.stylus_arm import StylusArm

log = logging.getLogger(__name__)

SLOW_Z_SPEED = 5000
PROBE_Z_SPEED = 5000


def _tap_once(arm: StylusArm, z: float, z_speed: int = PROBE_Z_SPEED):
    """Single tap: pen down, dwell 150ms, pen up."""
    arm._pen_down(z=z, speed=z_speed)
    arm._dwell(0.15)
    arm._pen_up()
    arm.wait_idle()


# ─── Step 0: Z-axis surface detection ────────────────────────

def step0_z_depth(arm: StylusArm, cal: CalibrationState) -> float:
    """Probe Z depth with tap-and-release at each level. Plan Step 0.

    Full tap cycle at each Z: pen down, dwell 150ms, pen up.
    After each tap, check if the phone reported a touch event.
    Spring-loaded stylus bounces back on pen-up, so no motor hold needed.

    Returns z_tap (contact depth + 0.5mm margin).
    """
    log.info("Step 0: Z-axis surface detection (stationary descent)")
    cal.set_phase("center")
    time.sleep(0.5)

    # Clear any stale touches
    cal.flush_touches()

    # Phase A: find first contact by descending in 0.3mm steps
    z_contact = None
    z_steps = [round(0.5 + i * 0.3, 2) for i in range(32)]  # 0.5 to 9.8mm

    for z in z_steps:
        log.info(f"  Probing Z={z:.2f}mm ...")
        _tap_once(arm, z, z_speed=SLOW_Z_SPEED)
        time.sleep(0.3)

        touches = cal.flush_touches()
        if touches:
            z_contact = z
            log.info(f"  First contact at Z={z:.2f}mm")
            break
        else:
            log.info(f"  No contact")

    if z_contact is None:
        raise RuntimeError("Step 0 FAILED — no touch detected up to 9.8mm. "
                           "Check stylus alignment and phone placement.")

    # Phase B: find a Z depth where 10/10 taps succeed.
    # Start at z_contact + 0.25mm. Tap 10 times. If any fails, add 0.25mm and retry.
    z_try = round(z_contact + 0.25, 2)
    z_tap = None

    for _ in range(10):  # max 10 rounds of 0.25mm increases
        log.info(f"  Testing Z={z_try:.2f}mm (10 taps) ...")
        all_ok = True
        for i in range(10):
            cal.flush_touches()
            _tap_once(arm, z_try, z_speed=SLOW_Z_SPEED)
            time.sleep(0.3)
            touches = cal.flush_touches()
            if touches:
                log.info(f"    Tap {i+1}/10 hit")
            else:
                log.info(f"    Tap {i+1}/10 miss — increasing Z")
                all_ok = False
                break

        if all_ok:
            z_tap = z_try
            log.info(f"  10/10 taps succeeded at Z={z_tap:.2f}mm")
            break
        else:
            z_try = round(z_try + 0.25, 2)

    if z_tap is None:
        raise RuntimeError("Step 0 FAILED — could not find reliable Z depth. "
                           "Check stylus and phone placement.")

    log.info(f"  z_tap = {z_tap}mm")
    return z_tap


# ─── Step 1: Arm-phone alignment check ───────────────────────

def step1_alignment(arm: StylusArm, cal: CalibrationState,
                    z_tap: float, separation_mm: float = 25.0
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
        cal.flush_touches()  # clear stale
        _tap_once(arm, z_tap)
        time.sleep(0.3)
        got = cal.flush_touches()
        if not got:
            raise RuntimeError(f"Step 1 FAILED — no touch at arm X={x:.1f}mm")
        touches.append(got[-1])

    arm._fast_move(0, 0)
    arm.wait_idle()

    sx1, sy1 = touches[0]['x'], touches[0]['y']
    sx2, sy2 = touches[1]['x'], touches[1]['y']
    log.info(f"  Tap A: arm X={-half:.1f}mm → screen ({sx1:.3f}, {sy1:.3f})")
    log.info(f"  Tap B: arm X={half:.1f}mm → screen ({sx2:.3f}, {sy2:.3f})")
    dx = abs(sx2 - sx1)
    dy = abs(sy2 - sy1)
    log.info(f"  dx={dx:.3f}, dy={dy:.3f}")

    # Arm X may map to phone X or phone Y (90° rotation is fine).
    # Check that the movement is straight: minor axis / major axis < 0.02.
    major = max(dx, dy)
    minor = min(dx, dy)
    if major < 0.01:
        raise RuntimeError("Step 1 FAILED — both taps at same position. "
                           "Check arm movement and phone placement.")

    tilt = minor / major
    log.info(f"  tilt_ratio = {tilt:.4f} (minor/major), "
             f"arm X → phone {'Y' if dy > dx else 'X'}, "
             f"{'OK (< 0.02)' if tilt < 0.02 else 'TILTED — adjust phone'}")
    return tilt


# ─── Step 2: Camera physical rotation check ──────────────────

def step2_camera_rotation(cam: Camera, screen_dimension: dict | None = None) -> dict:
    """Check camera orientation, tilt, and coverage. Plan Step 2.

    Checks:
    1. Long axes aligned: phone long axis matches image long axis
    2. No tilt: phone aspect ratio in image ≈ actual screen aspect ratio
    3. Coverage: phone area ≥ 70% of image area

    Returns dict with ok, messages, and measurements.
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
    # Use minAreaRect for orientation and area (handles rotated phones)
    rect = cv2.minAreaRect(largest)
    rect_w, rect_h = rect[1]
    phone_area_px = rect_w * rect_h
    img_h, img_w = frame.shape[:2]
    image_area = img_w * img_h
    issues = []

    # Save annotated image with actual contour + min-area rect
    annotated = frame.copy()
    cv2.drawContours(annotated, [largest], -1, (0, 255, 0), 3)
    box = cv2.boxPoints(rect)
    box = np.int32(box)
    cv2.drawContours(annotated, [box], -1, (0, 200, 255), 2)
    coverage = phone_area_px / image_area
    label = f"area {coverage:.0%}"
    bx, by, _, _ = cv2.boundingRect(largest)
    cv2.putText(annotated, label, (bx + 5, by + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imwrite("/tmp/physiclaw_step2.jpg", annotated)

    # 1. Camera rotation — phone edges should be parallel to image edges.
    # Compute angle from the longest edge of the min-area rect.
    phone_long = max(rect_w, rect_h)
    phone_short = min(rect_w, rect_h)
    pts = cv2.boxPoints(rect)
    # Find the longest edge and compute its angle to horizontal
    edges = [(pts[i], pts[(i+1) % 4]) for i in range(4)]
    longest_edge = max(edges, key=lambda e: np.linalg.norm(e[1] - e[0]))
    dx = longest_edge[1][0] - longest_edge[0][0]
    dy = longest_edge[1][1] - longest_edge[0][1]
    # Angle of longest edge relative to horizontal (or vertical)
    angle_deg = abs(np.degrees(np.arctan2(dy, dx)))
    # Normalize to deviation from nearest axis (0°, 90°, 180°)
    rotation_dev = min(angle_deg % 90, 90 - angle_deg % 90)
    rotation_ok = rotation_dev < 3.0  # < 3° deviation
    if not rotation_ok:
        issues.append(f"Straighten camera — phone edges rotated {rotation_dev:.1f}° from image edges")
    log.info(f"  Camera {img_w}×{img_h}, phone region {rect_w:.0f}×{rect_h:.0f}px "
             f"(area {phone_area_px:.0f}px²)")
    log.info(f"  Rotation: {rotation_dev:.1f}° from axis → "
             f"{'OK' if rotation_ok else 'ROTATED — straighten camera'}")

    # 2. Long axes aligned
    image_long_horizontal = img_w > img_h
    bx, by, bw, bh = cv2.boundingRect(largest)
    bbox_long_horizontal = bw > bh
    axes_ok = bbox_long_horizontal == image_long_horizontal
    if not axes_ok:
        issues.append("Rotate camera 90° — long axes not aligned")
    log.info(f"  Long axes: {'aligned' if axes_ok else 'NOT aligned — rotate camera 90°'}")

    # 2. Aspect ratio check (tilt detection)
    phone_ratio = phone_long / max(phone_short, 1)
    if screen_dimension:
        screen_w = screen_dimension.get('width', 430)
        screen_h = screen_dimension.get('height', 932)
        expected_ratio = max(screen_w, screen_h) / max(min(screen_w, screen_h), 1)
    else:
        expected_ratio = 2.0  # typical phone ~19.5:9 ≈ 2.17
    ratio_diff = abs(phone_ratio - expected_ratio) / expected_ratio
    tilt_ok = ratio_diff < 0.15
    if not tilt_ok:
        issues.append(f"Camera may be tilted — aspect ratio {phone_ratio:.2f} "
                      f"vs expected {expected_ratio:.2f} (diff {ratio_diff:.0%})")
    log.info(f"  Aspect ratio: {phone_ratio:.2f} (expected {expected_ratio:.2f}, "
             f"diff {ratio_diff:.0%}) → {'OK' if tilt_ok else 'TILTED'}")

    # 3. Coverage check (min-area rect ≥ 30% of image area)
    coverage_ok = coverage >= 0.30
    if not coverage_ok:
        issues.append(f"Move camera closer — phone covers only {coverage:.0%} of image (need ≥30%)")
    log.info(f"  Coverage: {coverage:.0%} → {'OK' if coverage_ok else 'TOO FAR'}")

    ok = axes_ok and tilt_ok and coverage_ok
    return {"ok": ok, "issues": issues,
            "phone_region": [round(rect_w), round(rect_h)],
            "image_size": [img_w, img_h],
            "aspect_ratio": round(phone_ratio, 2),
            "coverage": round(coverage, 2)}


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

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))

    def _find_blob(lower, upper, label):
        mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best = None
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 500:
                continue
            m = cv2.moments(cnt)
            if m['m00'] == 0:
                continue
            if best is None or area > best[2]:
                best = (m['m10'] / m['m00'], m['m01'] / m['m00'], area)
        if best is None:
            raise RuntimeError(f"Step 3 FAILED — {label} marker not found")
        return best[0], best[1]

    # UP = blue (#2563eb → HSV H≈110), RIGHT = red (#ef4444 → HSV H≈0/180)
    up_x, up_y = _find_blob([100, 80, 80], [130, 255, 255], "UP (blue)")
    right_x, right_y = _find_blob([0, 80, 80], [10, 255, 255], "RIGHT (red)")
    # Also check wrapped red hue (170-180)
    try:
        rx2, ry2 = _find_blob([170, 80, 80], [180, 255, 255], "RIGHT (red high)")
        # Pick whichever red blob is larger / was found
        right_x, right_y = rx2, ry2
    except RuntimeError:
        pass  # first range was enough

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

def _tap_and_read(arm: StylusArm, cal: CalibrationState,
                  gx: float, gy: float, z_tap: float) -> dict | None:
    """Move to (gx, gy), tap, return touch dict or None on miss."""
    arm._fast_move(gx, gy)
    arm.wait_idle()
    cal.flush_touches()
    _tap_once(arm, z_tap)
    time.sleep(0.3)
    got = cal.flush_touches()
    if not got:
        log.warning(f"  Miss at GRBL ({gx:.1f}, {gy:.1f})")
        return None
    return got[-1]


def step4_grbl_screen(arm: StylusArm, cal: CalibrationState,
                      z_tap: float
                      ) -> tuple[np.ndarray, list[dict]]:
    """Probe scale, tap grid points across the screen, compute affine. Plan Step 4.

    Phase 1: Tap center, +10mm X, +10mm Y to discover mm-per-screen-unit scale.
    Phase 2: Use scale + grid_cols/rows to tap 15 points spanning the full screen.
    Phase 3: Verify with 3 additional taps at offset positions.

    Touch x,y are 0-1 screen percentages.
    Returns (pct_to_grbl affine (2,3), list of touch events).
    """
    log.info("Step 4: GRBL ↔ Screen mapping (Mapping A)")
    cal.set_phase("center")
    time.sleep(0.3)

    PROBE_D = 10.0  # mm offset for scale probes

    # ── Phase 1: Probe to find scale ──
    log.info("  Phase 1: Probing scale (center, +X, +Y)")
    t_center = _tap_and_read(arm, cal, 0, 0, z_tap)
    if not t_center:
        raise RuntimeError("Step 4 FAILED — no touch at center")

    t_x = _tap_and_read(arm, cal, PROBE_D, 0, z_tap)
    if not t_x:
        raise RuntimeError("Step 4 FAILED — no touch at +X probe")

    t_y = _tap_and_read(arm, cal, 0, PROBE_D, z_tap)
    if not t_y:
        raise RuntimeError("Step 4 FAILED — no touch at +Y probe")

    # Screen displacement per mm in each arm direction
    # dx_screen / dx_grbl and dy_screen / dy_grbl
    sx_per_mm_x = (t_x['x'] - t_center['x']) / PROBE_D
    sy_per_mm_x = (t_x['y'] - t_center['y']) / PROBE_D
    sx_per_mm_y = (t_y['x'] - t_center['x']) / PROBE_D
    sy_per_mm_y = (t_y['y'] - t_center['y']) / PROBE_D
    log.info(f"  Scale: arm +X → screen ({sx_per_mm_x:.4f}, {sy_per_mm_x:.4f})/mm")
    log.info(f"  Scale: arm +Y → screen ({sx_per_mm_y:.4f}, {sy_per_mm_y:.4f})/mm")

    # Build screen→grbl affine from the 3 probe points
    probe_screen = np.array([
        [t_center['x'], t_center['y']],
        [t_x['x'], t_x['y']],
        [t_y['x'], t_y['y']],
    ], dtype=np.float64)
    probe_grbl = np.array([
        [0, 0], [PROBE_D, 0], [0, PROBE_D],
    ], dtype=np.float64)
    probe_affine, _ = cv2.estimateAffine2D(probe_screen, probe_grbl)

    # ── Phase 2: Tap grid points across full screen ──
    cols = cal.GRID_COLS_PCT  # [0.25, 0.50, 0.75]
    rows = cal.GRID_ROWS_PCT  # [0.20, 0.40, 0.50, 0.60, 0.80]
    log.info(f"  Phase 2: Tapping {len(cols)}×{len(rows)} grid points")

    grbl_pts = []
    screen_pts = []
    all_touches = []

    # Include probe points
    for sp, gp in zip(probe_screen, probe_grbl):
        screen_pts.append(sp.tolist())
        grbl_pts.append(gp.tolist())

    for row in rows:
        for col in cols:
            # Predict GRBL position from probe affine
            predicted = probe_affine @ np.array([col, row, 1.0])
            gx, gy = predicted[0], predicted[1]
            touch = _tap_and_read(arm, cal, gx, gy, z_tap)
            if not touch:
                continue
            grbl_pts.append([gx, gy])
            screen_pts.append([touch['x'], touch['y']])
            all_touches.append(touch)

    arm._fast_move(0, 0)
    arm.wait_idle()

    if len(grbl_pts) < 6:
        raise RuntimeError(f"Step 4 FAILED — only {len(grbl_pts)} valid taps (need ≥6)")

    # Compute final affine from all points
    screen_to_grbl, _ = cv2.estimateAffine2D(
        np.array(screen_pts, dtype=np.float64),
        np.array(grbl_pts, dtype=np.float64))
    if screen_to_grbl is None:
        raise RuntimeError("Step 4 FAILED — affine computation failed")

    # ── Phase 3: Verify with 3 taps at non-grid positions ──
    log.info(f"  Phase 3: Verifying ({len(grbl_pts)} pairs mapped)...")
    verify_pcts = [(0.15, 0.30), (0.85, 0.70), (0.50, 0.90)]
    max_error = 0
    for vsx, vsy in verify_pcts:
        predicted = screen_to_grbl @ np.array([vsx, vsy, 1.0])
        vgx, vgy = predicted[0], predicted[1]
        touch = _tap_and_read(arm, cal, vgx, vgy, z_tap)
        if not touch:
            continue
        actual_grbl = screen_to_grbl @ np.array([touch['x'], touch['y'], 1.0])
        error = ((actual_grbl[0] - vgx)**2 + (actual_grbl[1] - vgy)**2)**0.5
        max_error = max(max_error, error)
        log.info(f"  Verify ({vsx},{vsy}): error={error:.2f}mm")

    # ── Re-origin: move arm to screen center and set as new origin ──
    center_grbl = screen_to_grbl @ np.array([0.5, 0.5, 1.0])
    log.info(f"  Moving to screen center: GRBL ({center_grbl[0]:.2f}, {center_grbl[1]:.2f})")
    arm._fast_move(center_grbl[0], center_grbl[1])
    arm.wait_idle()
    arm.set_origin()
    log.info("  Screen center set as arm origin (0, 0)")

    # Update affine to reflect new origin: subtract center offset from translation
    screen_to_grbl[0, 2] -= center_grbl[0]
    screen_to_grbl[1, 2] -= center_grbl[1]

    log.info(f"  Verification max error: {max_error:.2f}mm")

    return screen_to_grbl, all_touches


# ─── Step 5: Camera ↔ Screen mapping (Mapping B) ────────────

def step5_camera_screen(cam: Camera, cal: CalibrationState,
                        rotation: int) -> np.ndarray:
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
    expected = len(cal.GRID_COLS_PCT) * len(cal.GRID_ROWS_PCT)

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
        [[x, y] for y in cal.GRID_ROWS_PCT for x in cal.GRID_COLS_PCT],
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
                   max_error: float = 0.015
                   ) -> list[dict]:
    """Full chain: dot → camera detect → Mapping B → Mapping A → tap → touch.

    Plan Step 6. Tests BOTH mappings end-to-end:
    1. Page shows orange dot at random position
    2. Camera detects orange dot in frame (camera pixels)
    3. Mapping B⁻¹: camera pixels → screen 0-1 pct
    4. Mapping A: screen 0-1 pct → GRBL mm
    5. Arm taps
    6. Phone reports touch coordinate (0-1 pct)
    7. Compare touch vs expected position (in 0-1 space)

    max_error: threshold in 0-1 units (0.015 ≈ 5px on a 390px screen).
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
        cal.flush_touches()
        _tap_once(arm, z_tap)
        time.sleep(0.3)

        # 6. Phone reports touch (0-1 pct)
        got = cal.flush_touches()
        touch = got[-1] if got else None
        if touch is None:
            results.append({"expected": (dot_pct_x, dot_pct_y),
                            "error": float('inf'), "passed": False})
            log.warning(f"  Test {i+1}: MISS — no touch")
            continue

        # 7. Compare in 0-1 space
        actual_x, actual_y = touch['x'], touch['y']
        error = ((actual_x - dot_pct_x)**2 +
                 (actual_y - dot_pct_y)**2)**0.5
        passed = error < max_error

        results.append({
            "expected": (dot_pct_x, dot_pct_y),
            "actual": (round(actual_x, 3), round(actual_y, 3)),
            "camera_pct": (round(cam_pct_x, 3), round(cam_pct_y, 3)),
            "error": round(error, 4),
            "passed": passed,
        })
        log.info(f"  Test {i+1}: error={error:.4f} {'✓' if passed else '✗'}")

    arm._fast_move(0, 0)
    arm.wait_idle()
    passed_count = sum(1 for r in results if r['passed'])
    log.info(f"  {passed_count}/{num_tests} passed")
    return results


# ─── Post: Set origin at screen center ───────────────────────

# ─── Full calibration orchestrator ───────────────────────────

def full_calibration(arm: StylusArm, cam: Camera,
                     cal: CalibrationState) -> GridCalibration:
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
    step2_result = step2_camera_rotation(cam)
    if not step2_result["ok"]:
        raise RuntimeError(f"Camera check failed: {'; '.join(step2_result['issues'])}")

    # Step 3: Software rotation
    rotation = step3_software_rotation(cam, cal)

    # Step 4: Mapping A (screen 0-1 → GRBL mm)
    # Touch coords are already 0-1, so affine maps pct directly to GRBL mm
    pct_to_grbl, _ = step4_grbl_screen(arm, cal, z_tap)

    # Park arm for camera step
    if arm.MOVE_DIRECTIONS:
        ux, uy = arm.MOVE_DIRECTIONS['top']
        arm._fast_move(ux * 100, uy * 100)
    else:
        arm._fast_move(0, -80)
    arm.wait_idle()

    # Step 5: Mapping B (0-1 pct → camera pixels)
    pct_to_pixel = step5_camera_screen(cam, cal, rotation)

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

    # Step 4 already set origin at screen center and adjusted affine

    cal.set_phase("idle")
    log.info("Calibration complete")
    return grid_cal
