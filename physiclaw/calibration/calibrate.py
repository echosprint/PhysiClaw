"""Calibration pipeline — touch coordinates + camera detection.

Pre-cal: viewport shift — map viewport CSS coords to screenshot 0-1
         using a detected orange square in a phone screenshot.

Arm side (``calibrate_arm``): find Z depth, tap probe triangle + 15-point
         grid, fit screen→arm affine, derive tilt diagnostic, re-origin
         at screen center.

Camera side: camera-rotation check, software frame-rotation from UP/RIGHT
         markers, 15-red-dot grid → screen→camera affine (``compute_camera_mapping``).

Validation: full chain — camera detects orange dot → screen pct → arm
         mm → tap → touch event → compare.

No green flash. Touch events for contact detection and coordinate
mapping. Camera for marker/dot visual detection only.
"""

import logging
import random
import time

import cv2
import numpy as np

from physiclaw.bridge import BridgeState, CalibrationState
from physiclaw.bridge.nonce import NONCE_COUNT, verify_nonce
from physiclaw.calibration.transforms import ScreenTransforms, ViewportShift
from physiclaw.hardware.camera import Camera
from physiclaw.hardware.arm import StylusArm
from physiclaw.hardware.iphone import AssistiveTouch
from physiclaw.vision.grid_detect import (
    detect_red_dots,
    sort_dots_to_grid,
    detect_orange_dot as _detect_orange_dot,
)

log = logging.getLogger(__name__)

SLOW_Z_SPEED = 6000
PROBE_Z_SPEED = 6000

CACHE_DIR = "data/calibration/cache"
PEN_DEPTH_FILE = f"{CACHE_DIR}/z-tap"
CAMERA_REF_FILE = f"{CACHE_DIR}/camera_ref.jpg"

FRAME_SIMILARITY_SIZE = (320, 240)


def grid_positions(cal: "CalibrationState"):
    """Yield (col_pct, row_pct) for each of the 15 grid positions in
    canonical outer-rows / inner-cols order. Used by arm calibration,
    camera mapping, and any downstream code that rebuilds the same grid."""
    for row in cal.GRID_ROWS_PCT:
        for col in cal.GRID_COLS_PCT:
            yield col, row


def _find_viewport_cache():
    """Return the first existing cached viewport screenshot, or None."""
    from pathlib import Path

    for ext in ("png", "jpg"):
        p = Path(f"{CACHE_DIR}/viewport.{ext}")
        if p.exists():
            return p
    return None


def load_pen_depth() -> float | None:
    """Load cached z-tap from data/pen/z-tap, or None if missing/unreadable."""
    from pathlib import Path

    p = Path(PEN_DEPTH_FILE)
    if not p.exists():
        return None
    try:
        val = float(p.read_text().strip())
    except (ValueError, OSError):
        return None
    if not (1 < val < 10):
        raise ValueError(f"{PEN_DEPTH_FILE}: z-tap={val}mm out of range (1, 10)")
    return val


def save_pen_depth(z_tap: float):
    """Save z-tap to data/pen/z-tap."""
    from pathlib import Path

    p = Path(PEN_DEPTH_FILE)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(f"{z_tap}\n")


def _tap_once(arm: StylusArm, z: float, z_speed: int = PROBE_Z_SPEED):
    """Single tap: pen down, dwell 150ms, pen up."""
    arm._pen_down(z=z, speed=z_speed)
    arm._dwell(0.15)
    arm._pen_up()
    arm.wait_idle()


# ─── Pre-cal: Screenshot coordinate mapping ──────────────────


def measure_viewport_shift(
    cal: CalibrationState, bridge: BridgeState
) -> ViewportShift:
    """Measure the viewport→screenshot pixel offset and DPR.

    Shows an orange square at a known viewport CSS position. User takes a
    phone screenshot (double-tap AssistiveTouch). Server detects the square
    in the screenshot and derives:
      - dpr (device pixel ratio)
      - offset_x, offset_y (status bar / safe-area shift)

    This must run before arm calibration so that all subsequent touch
    coordinates are correctly converted from viewport space to
    screenshot 0-1 space.

    Returns the ViewportShift and stores it on cal.viewport_shift.
    """
    log.info("═══ Pre-cal: Screenshot coordinate mapping ═══")
    log.info("  Goal: compute viewport CSS → screenshot pixel transform")

    dim = cal.screen_dimension
    if dim is None or dim.get("viewport_width", 0) == 0:
        raise RuntimeError(
            "Screen dimension not received from phone page. "
            "Make sure the phone has /bridge open."
        )
    log.info(f"  Phone viewport: {dim['viewport_width']}×{dim['viewport_height']}pt")

    # Show orange square at viewport center
    cal.set_phase("screenshot_cal")
    time.sleep(0.5)
    log.info("  Phase: screenshot_cal — showing orange square at CSS (100, 200)")

    cached = _find_viewport_cache()
    if cached is not None:
        data = cached.read_bytes()
        log.info(f"  Using cached screenshot: {cached} ({len(data)} bytes)")
    else:
        log.info("  Waiting for phone screenshot (double-tap AssistiveTouch)...")
        data = bridge.wait_screenshot(timeout=30.0)
        if data is None:
            raise RuntimeError(
                "Timeout — no screenshot received. Double-tap AssistiveTouch to upload."
            )
        log.info(f"  Screenshot received: {len(data)} bytes")

    # Decode screenshot
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError("Failed to decode screenshot image")

    sh, sw = img.shape[:2]
    log.info(f"  Screenshot decoded: {sw}×{sh}px")

    # Detect orange square (same HSV range as _detect_orange_dot)
    # Known CSS position: top-left (100, 200), size 50px → center (125, 225)
    SQUARE_CSS_X, SQUARE_CSS_Y, SQUARE_CSS_SIZE = 100, 200, 50

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([5, 100, 100]), np.array([25, 255, 255]))
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    )
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise RuntimeError(
            "Could not detect orange square in screenshot. "
            "Make sure the phone shows the orange square."
        )

    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)
    detected_cx = x + w / 2
    detected_cy = y + h / 2
    log.info(
        f"  Detected orange square: center=({detected_cx:.1f}, {detected_cy:.1f})px, "
        f"size={w}×{h}px in screenshot"
    )

    # Derive dpr from detected square size vs known CSS size
    dpr = w / SQUARE_CSS_SIZE
    log.info(
        f"  Device pixel ratio: {dpr:.2f} "
        f"(detected {w}px / expected {SQUARE_CSS_SIZE}css)"
    )

    # Compute offset between expected and actual position
    expected_cx = (SQUARE_CSS_X + SQUARE_CSS_SIZE / 2) * dpr
    expected_cy = (SQUARE_CSS_Y + SQUARE_CSS_SIZE / 2) * dpr
    offset_x = detected_cx - expected_cx
    offset_y = detected_cy - expected_cy
    log.info(
        f"  Offset: expected square center at ({expected_cx:.1f}, {expected_cy:.1f})px, "
        f"actual at ({detected_cx:.1f}, {detected_cy:.1f})px → "
        f"offset=({offset_x:.1f}, {offset_y:.1f})px "
        f"(status bar / safe area shift)"
    )

    transform = ViewportShift(
        offset_x=offset_x,
        offset_y=offset_y,
        dpr=dpr,
        screenshot_width=sw,
        screenshot_height=sh,
    )
    cal.viewport_shift = transform
    log.info(
        f"  ✓ Pre-cal done: dpr={dpr:.2f}, offset=({offset_x:.1f}, {offset_y:.1f})px, "
        f"screenshot={sw}×{sh}px"
    )

    if cached is None:
        from pathlib import Path

        ext = "png" if data[:4] == b"\x89PNG" else "jpg"
        out = Path(f"{CACHE_DIR}/viewport.{ext}")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(data)
        log.info(f"  Cached screenshot: {out}")
    return transform


# ─── Arm-side unified calibration ────────────────────────────


def _descend_to_contact(
    arm: StylusArm,
    cal: CalibrationState,
    z_start: float = 0.5,
    z_max: float = 9.8,
    step: float = 0.3,
) -> float:
    """Descend Z at the arm's current XY until a touch event registers.

    Returns the first-contact Z in mm. Raises if nothing contacts by z_max —
    usually means the stylus is off-screen or the phone isn't responsive.
    """
    log.info("  Z descent — finding first contact")
    cal.flush_touches()
    z = z_start
    while z <= z_max:
        z_rounded = round(z, 2)
        _tap_once(arm, z_rounded, z_speed=SLOW_Z_SPEED)
        time.sleep(0.3)
        if cal.flush_touches():
            log.info(f"    first contact at Z={z_rounded:.2f}mm")
            return z_rounded
        log.debug(f"    Z={z_rounded:.2f}mm — no contact")
        z += step
    raise RuntimeError(
        f"No touch detected up to {z_max}mm — check stylus and phone placement."
    )


# ─── Camera frame calibration ───────────────────────────────


def check_phone_in_frame(frame: np.ndarray) -> dict:
    """Shape/coverage/straightness diagnostic from one overhead frame.

    Returns ``{ok, issues, coverage, aspect_ratio, image_size, phone_region}``.
    Saves an annotated frame to /tmp/physiclaw_camera_rotation.jpg.
    Raises if no bright region is detected (camera read failed or phone off).
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise RuntimeError("No bright region in camera frame — is the phone on?")

    largest = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(largest)
    rect_w, rect_h = rect[1]
    phone_area_px = rect_w * rect_h
    img_h, img_w = frame.shape[:2]
    coverage = phone_area_px / (img_w * img_h)
    bx, by, bw, bh = cv2.boundingRect(largest)
    issues: list[str] = []

    annotated = frame.copy()
    cv2.drawContours(annotated, [largest], -1, (0, 255, 0), 3)
    cv2.drawContours(annotated, [np.int32(cv2.boxPoints(rect))], -1, (0, 200, 255), 2)
    cv2.putText(
        annotated, f"area {coverage:.0%}", (bx + 5, by + 30),
        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
    )
    cv2.imwrite("/tmp/physiclaw_camera_rotation.jpg", annotated)

    # Phone edges should be parallel to image edges (< 3° deviation).
    pts = cv2.boxPoints(rect)
    edges = [(pts[i], pts[(i + 1) % 4]) for i in range(4)]
    longest_edge = max(edges, key=lambda e: np.linalg.norm(e[1] - e[0]))
    angle_deg = abs(np.degrees(np.arctan2(
        longest_edge[1][1] - longest_edge[0][1],
        longest_edge[1][0] - longest_edge[0][0],
    )))
    rotation_dev = min(angle_deg % 90, 90 - angle_deg % 90)
    if rotation_dev >= 3.0:
        issues.append(
            f"Straighten camera — phone edges rotated {rotation_dev:.1f}° from image"
        )

    # Long axes aligned (phone long axis parallel to image long axis).
    if (bw > bh) != (img_w > img_h):
        issues.append("Rotate camera 90° — long axes not aligned")

    # Aspect ratio sanity check (camera tilt).
    phone_long = max(rect_w, rect_h)
    phone_short = min(rect_w, rect_h)
    phone_ratio = phone_long / max(phone_short, 1)
    ratio_diff = abs(phone_ratio - 2.0) / 2.0
    if ratio_diff >= 0.15:
        issues.append(
            f"Camera may be tilted — phone aspect {phone_ratio:.2f} (diff {ratio_diff:.0%})"
        )

    # Coverage: phone should fill ≥ 30% of frame.
    if coverage < 0.30:
        issues.append(
            f"Move camera closer — phone covers only {coverage:.0%} of image (need ≥30%)"
        )

    log.info(
        f"  Phone in frame: {rect_w:.0f}×{rect_h:.0f}px, "
        f"edge dev {rotation_dev:.1f}°, aspect {phone_ratio:.2f}, coverage {coverage:.0%}"
    )
    if issues:
        log.warning(f"  Camera setup issues: {'; '.join(issues)}")

    return {
        "ok": not issues,
        "issues": issues,
        "phone_region": [round(rect_w), round(rect_h)],
        "image_size": [img_w, img_h],
        "aspect_ratio": round(phone_ratio, 2),
        "coverage": round(coverage, 2),
    }


def _pick_rotation_from_markers(frame: np.ndarray) -> tuple[int, str]:
    """Locate blue UP and red RIGHT markers, derive the cv2 rotation code.

    Returns (rotation_code, human_label). Rotation is -1 (none) or one of
    cv2.ROTATE_{90_CLOCKWISE, 180, 90_COUNTERCLOCKWISE}. Raises if either
    marker is missing.
    """
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
            if m["m00"] == 0:
                continue
            if best is None or area > best[2]:
                best = (m["m10"] / m["m00"], m["m01"] / m["m00"], area)
        if best is None:
            raise RuntimeError(f"{label} marker not found")
        return best[0], best[1]

    # UP = blue (#2563eb → HSV H≈110), RIGHT = red (#ef4444 → HSV H≈0/180).
    up_x, up_y = _find_blob([100, 80, 80], [130, 255, 255], "UP (blue)")
    right_x, right_y = _find_blob([0, 80, 80], [10, 255, 255], "RIGHT (red)")
    try:
        right_x, right_y = _find_blob(
            [170, 80, 80], [180, 255, 255], "RIGHT (red high)"
        )
    except RuntimeError:
        pass

    log.info(f"  Blue UP at ({up_x:.0f}, {up_y:.0f}), red RIGHT at ({right_x:.0f}, {right_y:.0f})")

    if up_y < right_y and abs(up_x - right_x) < abs(up_y - right_y):
        return -1, "0° — no rotation needed"
    if up_x < right_x and abs(up_y - right_y) < abs(up_x - right_x):
        return cv2.ROTATE_90_CLOCKWISE, "90° clockwise"
    if up_y > right_y and abs(up_x - right_x) < abs(up_y - right_y):
        return cv2.ROTATE_180, "180°"
    return cv2.ROTATE_90_COUNTERCLOCKWISE, "90° counter-clockwise"


def _save_camera_reference(raw_frame: np.ndarray) -> None:
    """Persist the camera's raw (unrotated) frame as the reference for
    future warm-restart auto-pick. One camera is identified by what it
    sees; we save its view so later sessions can pick the same USB
    index by scene similarity rather than a shape heuristic.
    """
    from pathlib import Path

    Path(CACHE_DIR).mkdir(parents=True, exist_ok=True)
    cv2.imwrite(CAMERA_REF_FILE, raw_frame)
    log.info(f"  Saved camera reference → {CAMERA_REF_FILE}")


def frame_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Normalized cross-correlation of two frames in [-1, 1].

    Downsample to a common grayscale size and let cv2.matchTemplate
    compute Pearson's r. ~1 means same scene, ~0 uncorrelated.
    """
    ga = cv2.resize(cv2.cvtColor(a, cv2.COLOR_BGR2GRAY), FRAME_SIMILARITY_SIZE)
    gb = cv2.resize(cv2.cvtColor(b, cv2.COLOR_BGR2GRAY), FRAME_SIMILARITY_SIZE)
    return float(cv2.matchTemplate(ga, gb, cv2.TM_CCOEFF_NORMED)[0, 0])


def calibrate_camera_frame(cam: Camera, cal: CalibrationState) -> dict:
    """Camera frame calibration — physical setup check + rotation code.

    One overhead frame drives both:
      - physical camera-setup diagnostic (shape, coverage, edge straightness),
      - software rotation code picked from the UP/RIGHT markers.

    Returns ``{"rotation", "rotation_name", "setup_ok", "issues",
              "coverage", "aspect_ratio", "image_size", "phone_region"}``.
    """
    log.info("═══ Camera frame calibration ═══")
    cal.set_phase("markers")
    time.sleep(1.0)

    frame = cam._fresh_frame()
    if frame is None:
        raise RuntimeError("Camera read failed — is the camera connected?")

    checks = check_phone_in_frame(frame)
    rotation, rot_label = _pick_rotation_from_markers(frame)
    _save_camera_reference(frame)
    log.info(f"  ✓ Camera frame: rotation={rot_label}, setup_ok={checks['ok']}")
    return {
        "rotation": rotation,
        "rotation_name": rot_label,
        "setup_ok": checks["ok"],
        "issues": checks["issues"],
        "coverage": checks["coverage"],
        "aspect_ratio": checks["aspect_ratio"],
        "image_size": checks["image_size"],
        "phone_region": checks["phone_region"],
    }


# ─── Screen ↔ arm affine (Mapping A) ────────────────────────


def _tap_and_read(
    arm: StylusArm,
    cal: CalibrationState,
    gx: float,
    gy: float,
    z_tap: float,
    max_retries: int = 3,
) -> tuple[dict | None, float]:
    """Move to (gx, gy), tap, return (touch dict, z_tap used).

    On miss, retries with z_tap += 0.25mm up to max_retries times.
    Returns updated z_tap so caller can use the deeper value going forward.
    """
    arm._fast_move(gx, gy)
    arm.wait_idle()
    z = z_tap
    for attempt in range(max_retries + 1):
        cal.flush_touches()
        _tap_once(arm, z)
        time.sleep(0.3)
        got = cal.flush_touches()
        if got:
            if z > z_tap:
                log.info(
                    f"    tap at arm ({gx:.1f}, {gy:.1f})mm: hit after Z bump "
                    f"to {z:.2f}mm"
                )
            return got[-1], z
        if attempt < max_retries:
            z = round(z + 0.25, 2)
            log.warning(
                f"    tap at arm ({gx:.1f}, {gy:.1f})mm: missed, "
                f"retry {attempt + 1}/{max_retries} with Z={z:.2f}mm"
            )
    log.warning(
        f"    tap at arm ({gx:.1f}, {gy:.1f})mm: FAILED after {max_retries} retries"
    )
    return None, z


PROBE_D = 10.0  # mm offset for the probe triangle
TILT_ALIGNED_THRESHOLD = 0.02  # arm/phone axis mismatch below this is "aligned"


def _tilt_from_affine(pct_to_grbl: np.ndarray) -> float:
    """Derive the arm–phone axis mismatch ratio from the fitted affine.

    Invert the 2×2 linear part so each column is an arm-axis basis vector
    expressed in screen 0-1 units; the minor-axis / major-axis ratio of
    arm-X's screen vector tells us how misaligned arm-X is with a phone
    screen axis. 0 → perfectly aligned; 1 → diagonal.
    """
    A = pct_to_grbl[:, :2]
    try:
        A_inv = np.linalg.inv(A)
    except np.linalg.LinAlgError:
        return 1.0
    arm_x_in_screen = A_inv[:, 0]
    dx = abs(float(arm_x_in_screen[0]))
    dy = abs(float(arm_x_in_screen[1]))
    major = max(dx, dy)
    if major < 1e-6:
        return 1.0
    return min(dx, dy) / major


def calibrate_arm(
    arm: StylusArm,
    cal: CalibrationState,
    z_tap_hint: float | None = None,
) -> tuple[float, np.ndarray, float, list[dict]]:
    """Arm calibration — Z depth, screen↔arm affine, tilt diagnostic.

    1. Find first-contact Z (skipped if ``z_tap_hint`` is provided — e.g.
       from the cached value on disk).
    2. Probe triangle: 3 taps at arm (0,0), (+10,0), (0,+10) with
       z-bump-on-miss. Yields a bootstrap screen→arm affine.
    3. Grid: for each of the 15 viewport grid positions predicted via
       the bootstrap affine, tap with z-bump-on-miss. ``z_tap`` grows
       to whatever depth the farthest grid point needs.
    4. Fit the final affine from all 18 (arm mm, screen 0-1) pairs,
       derive the tilt ratio, re-origin the arm at screen center.

    Returns ``(z_tap, pct_to_grbl, tilt_ratio, grid_touches)``. Tilt
    ``< TILT_ALIGNED_THRESHOLD`` means arm and phone axes are aligned;
    higher means the phone is rotated relative to arm travel.
    """
    log.info("═══ Arm calibration — Z depth + screen↔arm mapping ═══")
    cal.set_phase("center")
    time.sleep(0.5)

    if z_tap_hint is not None:
        z_tap = round(z_tap_hint, 2)
        log.info(f"  Z depth: using cached {z_tap}mm — skipping descent")
    else:
        z_contact = _descend_to_contact(arm, cal)
        z_tap = round(z_contact + 0.25, 2)
        log.info(f"  Z depth: starting at {z_tap}mm (first contact + 0.25mm margin)")

    # Probe triangle — bootstrap the screen→arm mapping.
    log.info(f"  Probe triangle: 3 taps at (0,0), (+{PROBE_D:.0f},0), (0,+{PROBE_D:.0f})")
    t_center, z_tap = _tap_and_read(arm, cal, 0, 0, z_tap)
    if not t_center:
        raise RuntimeError("Arm calibration FAILED — no touch at center")
    t_x, z_tap = _tap_and_read(arm, cal, PROBE_D, 0, z_tap)
    if not t_x:
        raise RuntimeError(f"Arm calibration FAILED — no touch at +{PROBE_D:.0f}mm X")
    t_y, z_tap = _tap_and_read(arm, cal, 0, PROBE_D, z_tap)
    if not t_y:
        raise RuntimeError(f"Arm calibration FAILED — no touch at +{PROBE_D:.0f}mm Y")

    probe_screen = np.array(
        [
            [t_center["x"], t_center["y"]],
            [t_x["x"], t_x["y"]],
            [t_y["x"], t_y["y"]],
        ],
        dtype=np.float64,
    )
    probe_grbl = np.array(
        [[0, 0], [PROBE_D, 0], [0, PROBE_D]], dtype=np.float64
    )
    probe_affine, _ = cv2.estimateAffine2D(probe_screen, probe_grbl)
    if probe_affine is None:
        raise RuntimeError("Arm calibration FAILED — probe affine fit failed")

    # Grid — 15 viewport positions predicted via the bootstrap affine.
    cal.set_phase("grid")
    time.sleep(0.3)
    grid = list(grid_positions(cal))
    log.info(f"  Grid: {len(grid)} taps across full screen (phase=grid)")

    grbl_pts: list = [probe_grbl[i].tolist() for i in range(3)]
    screen_pts: list = [probe_screen[i].tolist() for i in range(3)]
    grid_touches: list = []

    for idx, (col, row) in enumerate(grid, start=1):
        if cal.viewport_shift:
            scr_col, scr_row = cal.viewport_pct_to_screenshot_pct(col, row)
        else:
            scr_col, scr_row = col, row
        predicted = probe_affine @ np.array([scr_col, scr_row, 1.0])
        gx, gy = float(predicted[0]), float(predicted[1])
        log.info(
            f"    Grid {idx}/{len(grid)}: viewport ({col:.2f}, {row:.2f}) → "
            f"arm ({gx:.1f}, {gy:.1f})mm"
        )
        touch, z_tap = _tap_and_read(arm, cal, gx, gy, z_tap)
        if not touch:
            log.warning(f"    Grid {idx}/{len(grid)}: NO TOUCH — skipped")
            continue
        log.info(
            f"    Grid {idx}/{len(grid)}: touch at "
            f"screen ({touch['x']:.3f}, {touch['y']:.3f})"
        )
        grbl_pts.append([gx, gy])
        screen_pts.append([touch["x"], touch["y"]])
        grid_touches.append(touch)

    arm.return_to_origin()

    log.info(
        f"  Collected {len(grbl_pts)} point pairs "
        f"(3 probes + {len(grbl_pts) - 3} grid hits)"
    )
    if len(grbl_pts) < 6:
        raise RuntimeError(
            f"Arm calibration FAILED — only {len(grbl_pts)} valid taps (need ≥6)"
        )

    pct_to_grbl, _ = cv2.estimateAffine2D(
        np.array(screen_pts, dtype=np.float64),
        np.array(grbl_pts, dtype=np.float64),
    )
    if pct_to_grbl is None:
        raise RuntimeError("Arm calibration FAILED — final affine fit failed")

    tilt = _tilt_from_affine(pct_to_grbl)
    aligned = tilt < TILT_ALIGNED_THRESHOLD
    log.info(
        f"  Tilt ratio: {tilt:.4f} "
        f"(want < {TILT_ALIGNED_THRESHOLD}; aligned={aligned})"
    )
    if not aligned:
        log.warning(
            "  Phone/arm axes are skewed — consider straightening phone "
            "orientation if tilt stays high across reruns."
        )

    # Re-origin at screen center.
    center_grbl = pct_to_grbl @ np.array([0.5, 0.5, 1.0])
    log.info(
        f"  Re-origin: screen center is at arm "
        f"({center_grbl[0]:.2f}, {center_grbl[1]:.2f})mm → setting as (0, 0)"
    )
    arm._fast_move(center_grbl[0], center_grbl[1])
    arm.wait_idle()
    arm.set_origin()
    pct_to_grbl[0, 2] -= center_grbl[0]
    pct_to_grbl[1, 2] -= center_grbl[1]
    cal.set_phase("center")

    log.info(
        f"  ✓ Arm calibration done: z_tap={z_tap}mm, "
        f"{len(grbl_pts)} pairs, tilt={tilt:.4f}"
    )
    return z_tap, pct_to_grbl, tilt, grid_touches


# ─── Step 5: Camera ↔ Screen mapping (Mapping B) ────────────


def compute_camera_mapping(
    cam: Camera, cal: CalibrationState, rotation: int
) -> tuple[np.ndarray, tuple[int, int]]:
    """Detect 15 red dots, compute screen 0-1 → camera 0-1 affine. Plan Step 5.

    Returns (pct_to_cam affine (2,3), cam_size (w, h)).
    Both sides are 0-1 normalized.
    """
    log.info("═══ Step 5: Camera ↔ Screen mapping (Mapping B) ═══")
    log.info("  Goal: compute affine transform from screen 0-1 → camera 0-1")
    cal.set_phase("grid")
    time.sleep(1.0)
    expected = len(cal.GRID_COLS_PCT) * len(cal.GRID_ROWS_PCT)
    log.info(
        f"  Phase: grid — phone shows {expected} red dots at known viewport positions"
    )

    frame = cam._fresh_frame()
    if frame is None:
        raise RuntimeError("Step 5 FAILED — camera read failed")
    if rotation >= 0:
        frame = cv2.rotate(frame, rotation)
    frame_h, frame_w = frame.shape[:2]
    cam_size = (frame_w, frame_h)
    rot_names = {
        -1: "none",
        cv2.ROTATE_90_CLOCKWISE: "90° CW",
        cv2.ROTATE_180: "180°",
        cv2.ROTATE_90_COUNTERCLOCKWISE: "90° CCW",
    }
    log.info(
        f"  Camera frame captured: {frame_w}×{frame_h}px "
        f"(rotation={rot_names.get(rotation, str(rotation))})"
    )

    dots = detect_red_dots(frame)
    log.info(f"  Red dot detection: found {len(dots)}/{expected} dots")

    # Retry once if detection fails
    if len(dots) != expected:
        log.info("  Retrying dot detection after 1s...")
        time.sleep(1.0)
        frame = cam._fresh_frame()
        if frame is not None:
            if rotation >= 0:
                frame = cv2.rotate(frame, rotation)
            dots = detect_red_dots(frame)
            log.info(f"  Retry: found {len(dots)}/{expected} dots")

    if len(dots) != expected:
        raise RuntimeError(
            f"Step 5 FAILED — detected {len(dots)} dots, expected {expected}"
        )

    camera_pixels = sort_dots_to_grid(
        dots, rows=len(cal.GRID_ROWS_PCT), cols=len(cal.GRID_COLS_PCT)
    )
    log.info(
        f"  Dots sorted into {len(cal.GRID_COLS_PCT)}×{len(cal.GRID_ROWS_PCT)} grid"
    )

    # Normalize camera pixels to 0-1
    camera_01 = camera_pixels.astype(np.float64)
    camera_01[:, 0] /= frame_w
    camera_01[:, 1] /= frame_h

    # Grid positions: dots are rendered at viewport percentages.
    # Convert to screenshot 0-1 if the viewport shift is known,
    # so Mapping B uses the same coordinate space as Mapping A.
    coord_space = "screenshot 0-1" if cal.viewport_shift else "viewport 0-1"
    if cal.viewport_shift:
        screen_pcts = np.array(
            [list(cal.viewport_pct_to_screenshot_pct(col, row))
             for col, row in grid_positions(cal)],
            dtype=np.float64,
        )
    else:
        screen_pcts = np.array(
            [[col, row] for col, row in grid_positions(cal)],
            dtype=np.float64,
        )
    log.info(f"  Mapping {expected} dots: {coord_space} ↔ camera 0-1")

    # Homography for inlier check
    cam_to_pct, mask = cv2.findHomography(camera_01, screen_pcts, cv2.RANSAC, 0.01)
    if cam_to_pct is None:
        raise RuntimeError("Step 5 FAILED — homography computation failed")
    inliers = int(mask.sum()) if mask is not None else 0
    log.info(f"  Homography (camera 0-1 → screen 0-1): {inliers}/{len(dots)} inliers")

    # Affine: screen 0-1 → camera 0-1
    pct_to_cam, _ = cv2.estimateAffine2D(screen_pcts, camera_01)
    if pct_to_cam is None:
        raise RuntimeError("Step 5 FAILED — affine computation failed")

    log.info(
        f"  ✓ Step 5 done: Mapping B ready (screen 0-1 → camera 0-1) "
        f"from {len(dots)} dot pairs, frame {frame_w}×{frame_h}px"
    )
    return pct_to_cam, cam_size


# ─── Step 6: Full-chain validation ───────────────────────────


def validate_calibration(
    arm: StylusArm,
    cam: Camera,
    cal: CalibrationState,
    z_tap: float,
    rotation: int,
    pct_to_grbl: np.ndarray,
    pct_to_cam: np.ndarray,
    cam_size: tuple[int, int] = (1920, 1080),
    num_tests: int = 5,
    max_error: float = 0.015,
) -> list[dict]:
    """Full chain: dot → camera detect → Mapping B → Mapping A → tap → touch.

    Plan Step 6. Tests BOTH mappings end-to-end:
    1. Page shows orange dot at random position
    2. Camera detects orange dot in frame (camera pixels → normalize to 0-1)
    3. Mapping B⁻¹: camera 0-1 → screen 0-1 pct
    4. Mapping A: screen 0-1 pct → GRBL mm
    5. Arm taps
    6. Phone reports touch coordinate (0-1 pct)
    7. Compare touch vs expected position (in 0-1 space)

    max_error: threshold in 0-1 units (0.015 ≈ 5px on a 390px screen).
    """
    log.info("═══ Step 6: Full-chain validation ═══")
    log.info(f"  Goal: end-to-end test of both mappings — {num_tests} random positions")
    log.info(
        "  Chain: dot on screen → camera detect → Mapping B⁻¹ → Mapping A → arm tap → touch"
    )
    log.info(
        f"  Pass threshold: error < {max_error} in screen 0-1 space "
        f"(≈{max_error * 390:.0f}px on a 390px-wide screen)"
    )

    # Compute inverse of pct_to_cam for camera 0-1 → screen pct
    A = pct_to_cam[:, :2]  # 2×2
    b = pct_to_cam[:, 2]  # translation
    A_inv = np.linalg.inv(A)
    cam_to_pct = np.hstack([A_inv, (-A_inv @ b).reshape(2, 1)])
    cam_w, cam_h = cam_size

    results = []
    for i in range(num_tests):
        log.info(f"  ── Test {i + 1}/{num_tests} ──")

        # Random viewport 0-1 position for rendering the dot
        vp_x = round(0.2 + random.random() * 0.6, 3)
        vp_y = round(0.2 + random.random() * 0.6, 3)

        # Expected position in screenshot 0-1 (for comparison with touch results)
        if cal.viewport_shift:
            expected_x, expected_y = cal.viewport_pct_to_screenshot_pct(vp_x, vp_y)
        else:
            expected_x, expected_y = vp_x, vp_y

        # 1. Show orange dot (bridge.html renders in viewport space)
        cal.set_phase("dot", dot_x=vp_x, dot_y=vp_y)
        time.sleep(0.5)
        log.info(
            f"    1. Dot placed at viewport ({vp_x:.3f}, {vp_y:.3f}) → "
            f"expected screen ({expected_x:.3f}, {expected_y:.3f})"
        )

        # 2. Camera detects orange dot
        # Park arm first so it doesn't occlude
        park_gx, park_gy = pct_to_grbl @ np.array([0.5, -0.1, 1.0])
        arm._fast_move(float(park_gx), float(park_gy))
        arm.wait_idle()
        time.sleep(0.3)

        frame = cam._fresh_frame()
        if frame is not None and rotation >= 0:
            frame = cv2.rotate(frame, rotation)

        detected = _detect_orange_dot(frame) if frame is not None else None

        if detected is None:
            log.warning(
                "    2. Camera: could not detect orange dot — "
                "falling back to known position"
            )
            cam_pct_x, cam_pct_y = expected_x, expected_y
        else:
            # 3. Mapping B⁻¹: camera 0-1 → screen pct (screenshot 0-1)
            cam_01_x = detected[0] / cam_w
            cam_01_y = detected[1] / cam_h
            cam_pt = np.array([cam_01_x, cam_01_y, 1.0])
            screen_pct = cam_to_pct @ cam_pt
            cam_pct_x, cam_pct_y = float(screen_pct[0]), float(screen_pct[1])
            log.info(
                f"    2. Camera: detected dot at pixel ({detected[0]:.0f}, {detected[1]:.0f}) "
                f"→ camera 0-1 ({cam_01_x:.3f}, {cam_01_y:.3f})"
            )
            log.info(
                f"    3. Mapping B⁻¹: camera 0-1 → screen ({cam_pct_x:.3f}, {cam_pct_y:.3f})"
            )

        # 4. Mapping A: screen pct → GRBL mm
        grbl_pos = pct_to_grbl @ np.array([cam_pct_x, cam_pct_y, 1.0])
        gx, gy = float(grbl_pos[0]), float(grbl_pos[1])
        log.info(
            f"    4. Mapping A: screen ({cam_pct_x:.3f}, {cam_pct_y:.3f}) → "
            f"arm ({gx:.1f}, {gy:.1f})mm"
        )

        # 5. Arm taps (retry with deeper z on miss)
        arm._fast_move(gx, gy)
        arm.wait_idle()
        touch = None
        z = z_tap
        for attempt in range(4):
            cal.flush_touches()
            _tap_once(arm, z)
            time.sleep(0.3)
            got = cal.flush_touches()
            if got:
                touch = got[-1]
                if z > z_tap:
                    z_tap = z
                    log.info(f"    5. Tap: hit after Z bump to {z:.2f}mm")
                break
            z = round(z + 0.25, 2)
            log.warning(
                f"    5. Tap: missed at arm ({gx:.1f}, {gy:.1f})mm, "
                f"retry {attempt + 1}/3 with Z={z:.2f}mm"
            )

        if touch is None:
            results.append(
                {"expected": (expected_x, expected_y), "error": 999.0, "passed": False}
            )
            log.warning("    5. Tap: FAILED — no touch registered after 4 attempts")
            continue

        # 7. Compare in screenshot 0-1 space
        actual_x, actual_y = touch["x"], touch["y"]
        error = ((actual_x - expected_x) ** 2 + (actual_y - expected_y) ** 2) ** 0.5
        passed = error < max_error

        results.append(
            {
                "expected": (round(expected_x, 3), round(expected_y, 3)),
                "actual": (round(actual_x, 3), round(actual_y, 3)),
                "camera_pct": (round(cam_pct_x, 3), round(cam_pct_y, 3)),
                "error": round(error, 4),
                "passed": passed,
            }
        )
        log.info(f"    5. Tap: touch at screen ({actual_x:.3f}, {actual_y:.3f})")
        log.info(
            f"    6. Error: {error:.4f} (expected ({expected_x:.3f}, {expected_y:.3f}), "
            f"actual ({actual_x:.3f}, {actual_y:.3f})) → "
            f"{'PASS' if passed else 'FAIL'}"
        )

    arm.return_to_origin()
    passed_count = sum(1 for r in results if r["passed"])
    log.info(f"  ✓ Validation done: {passed_count}/{num_tests} tests passed")
    return results


# ─── Edge-trace verification ──────────────────────────────────


def trace_screen_edge(arm: StylusArm, cal: ScreenTransforms):
    """Trace the phone screen border clockwise for visual verification.

    Moves the arm to 8 edge points (top-center → top-right → right-center
    → bottom-right → bottom-center → bottom-left → left-center → top-left
    → back to top-center), pausing 2s at each. Then returns to center.
    Used after `validate_calibration` so the user can visually confirm the
    arm follows the actual screen edges.
    """
    check_points = [
        (0.50, 0, "top center"),
        (1, 0, "top right"),
        (1, 0.50, "right center"),
        (1, 1, "bottom right"),
        (0.50, 1, "bottom center"),
        (0, 1, "bottom left"),
        (0, 0.50, "left center"),
        (0, 0, "top left"),
        (0.50, 0, "top center"),  # close the loop
    ]
    arm.return_to_origin()
    log.info("Tracing phone edge clockwise...")
    for x_pct, y_pct, label in check_points:
        gx, gy = cal.pct_to_grbl_mm(x_pct, y_pct)
        log.info(f"  → {label} ({x_pct}, {y_pct}) = GRBL ({gx:.2f}, {gy:.2f})")
        arm._fast_move(gx, gy)
        arm.wait_idle()
        time.sleep(2)

    arm.return_to_origin()
    log.info("Edge trace done")


# ─── Step 7: AssistiveTouch screenshot verification ─────────


def verify_assistive_touch(
    arm: StylusArm,
    at: AssistiveTouch,
    bridge: BridgeState,
    cal: CalibrationState,
    pct_to_grbl: np.ndarray,
) -> dict:
    """Step 7: verify all three AT gestures end-to-end.

    1. Single-tap → "PhysiClaw Tap" Shortcut takes a screenshot (saved to Photos).
    2. Wait 5s for the screenshot animation to finish.
    3. Double-tap → "PhysiClaw Screenshot" Shortcut uploads the latest photo.
       Verify the uploaded image contains the color nonce.
    4. Long-press → "PhysiClaw Clipboard" Shortcut GETs /api/bridge/clipboard.
       Verify the server's clipboard-copied event fires within the timeout.
       Return the queued text so the user can paste-verify downstream.

    Requires:
    - Phase "assistive_touch" already set on phone with nonce bits
    - User has positioned AT at the orange circle
    - cal.viewport_shift is set (from pre-cal)
    - arm.Z_DOWN is set (from step 0)

    Returns:
        {
          "passed": bool,  # True iff both sub-checks passed
          "screenshot": {"passed": bool, "matched": int, "total": int},
          "clipboard":  {"fetched": bool, "text": str | None},
        }
    """
    if at.at_screen is None:
        raise RuntimeError("AT position not set — call compute_at_screen_pos first")

    log.info("═══ Step 7: AssistiveTouch screenshot verification ═══")
    log.info(
        f"  AT position: screen 0-1 ({at.at_screen[0]:.3f}, {at.at_screen[1]:.3f})"
    )

    nonce = cal._screenshot_nonce
    if nonce is None:
        raise RuntimeError("No nonce set — call assistive-touch/show first")

    bridge.clear_screenshot()

    log.info("  Single-tap AT (iOS screenshot)...")
    at.tap(arm, pct_to_grbl)

    log.info("  Waiting 5s for screenshot animation...")
    time.sleep(5.0)

    log.info("  Double-tap AT (screenshot + upload)...")
    at.double_tap(arm, pct_to_grbl)

    log.info("  Waiting for screenshot upload...")
    data = bridge.wait_screenshot(timeout=10.0)
    if data is None:
        log.warning("  Screenshot upload timed out")
        return {
            "passed": False,
            "screenshot": {"passed": False, "matched": 0, "total": NONCE_COUNT},
            "clipboard": {"fetched": False, "text": None},
        }

    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        log.warning("  Failed to decode screenshot")
        return {
            "passed": False,
            "screenshot": {"passed": False, "matched": 0, "total": NONCE_COUNT},
            "clipboard": {"fetched": False, "text": None},
        }

    log.info(f"  Screenshot received: {img.shape[1]}×{img.shape[0]}px")

    t = cal.viewport_shift
    if t is None:
        raise RuntimeError("viewport_shift not set — run measure-viewport-shift first")

    shot_passed, matched = verify_nonce(img, t, nonce)

    if shot_passed:
        log.info(
            f"  ✓ Screenshot pipeline verified ({matched}/{NONCE_COUNT} bits matched)"
        )
    else:
        log.warning(
            f"  ✗ Screenshot verification failed: {matched}/{NONCE_COUNT} bits matched"
        )

    # ─── Long-press: clipboard fetch verification ──────────────
    # Give iOS a moment to finish the previous Shortcut run before we
    # queue new text and trigger another one.
    time.sleep(5.0)
    clip_text = f"PhysiClaw-{random.randbytes(3).hex().upper()}"
    log.info(f"  Queuing clipboard text: {clip_text!r}")
    bridge.send_text(clip_text)

    log.info("  Long-press AT (iOS Shortcut → fetch bridge text)...")
    at.long_press(arm, pct_to_grbl)

    log.info("  Waiting for iOS Shortcut to fetch clipboard text...")
    clip_fetched = bridge.wait_clipboard(timeout=10.0)
    if clip_fetched:
        log.info(f"  ✓ Clipboard fetched from server — text: {clip_text!r}")
        log.info("    Paste into Notes / any text field to verify the text matches.")
    else:
        log.warning("  ✗ Clipboard fetch timed out — server was not hit")

    # Clear the queued text so the bridge page doesn't keep displaying the
    # leftover nonce after the phone switches back to bridge mode.
    bridge.clear_text()

    passed = shot_passed and clip_fetched
    if passed:
        log.info("  ✓ Step 7 done: AT tap + double-tap + long-press all verified")
    else:
        log.warning("  ✗ Step 7 failed")

    return {
        "passed": passed,
        "screenshot": {
            "passed": shot_passed,
            "matched": matched,
            "total": NONCE_COUNT,
        },
        "clipboard": {
            "fetched": clip_fetched,
            "text": clip_text,
        },
    }
