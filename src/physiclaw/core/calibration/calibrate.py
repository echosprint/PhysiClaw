"""Calibration pipeline — touch coordinates + camera detection.

Pre-cal: viewport shift — map viewport CSS coords to screenshot 0-1
         using a detected orange square in a phone screenshot.

Arm side (``calibrate_arm``): tap probe triangle + 15-point grid (each
         tap fires the solenoid — no Z depth to find), fit screen→arm
         affine, derive tilt diagnostic, re-origin at screen center.

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
from pathlib import Path

import cv2
import numpy as np

from physiclaw import paths
from physiclaw.core.bridge import BridgeState, CalibrationState, PageState
from physiclaw.core.bridge.nonce import NONCE_COUNT, verify_nonce
from physiclaw.core.calibration.transforms import (
    PARK_PCT,
    ScreenTransforms,
    ViewportShift,
)
from physiclaw.core.hardware.camera import Camera
from physiclaw.core.hardware.arm import StylusArm
from physiclaw.core.hardware.iphone import AssistiveTouch
from physiclaw.core.vision.grid_detect import (
    detect_red_dots,
    detect_screen_corners,
    screen_polygon,
    sort_dots_to_grid,
    detect_orange_dot as _detect_orange_dot,
)
from physiclaw.core.vision.blobs import find_largest_hsv_blob
from physiclaw.core.vision.colors import ORANGE_HSV_RANGE, hsv_mask, red_ranges
from physiclaw.core.vision.preprocess import to_hsv
from physiclaw.core.vision.util import check_phone_in_frame

log = logging.getLogger(__name__)

# Slightly longer than a normal tap so flaky first contacts still register
# during calibration probing.
CAL_STRIKE_DURATION = 0.15

VIEWPORT_CACHE_STEM = paths.calibration_cache_dir() / "viewport"


def grid_positions(cal: "CalibrationState"):
    """Yield (col_pct, row_pct) for each of the 15 grid positions in
    canonical outer-rows / inner-cols order. Used by arm calibration,
    camera mapping, and any downstream code that rebuilds the same grid."""
    for row in cal.GRID_ROWS_PCT:
        for col in cal.GRID_COLS_PCT:
            yield col, row


def _find_viewport_cache() -> Path | None:
    """Return the first existing cached viewport screenshot, or None."""
    for ext in ("png", "jpg"):
        p = VIEWPORT_CACHE_STEM.with_suffix(f".{ext}")
        if p.exists():
            return p
    return None


def _tap_once(arm: StylusArm):
    """Single calibration tap: fire the solenoid for CAL_STRIKE_DURATION."""
    arm.solenoid.tap(CAL_STRIKE_DURATION)
    arm.wait_idle()


# ─── Pre-cal: Screenshot coordinate mapping ──────────────────


def measure_viewport_shift(
    cal: CalibrationState, bridge: BridgeState, *, fresh: bool = False,
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

    `fresh=True` bypasses the disk cache at `VIEWPORT_CACHE_STEM` and
    always waits for a fresh screenshot — interactive setup defaults to
    this so the operator gets a real measurement, not a cached one
    from a possibly-stale rig position.

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

    cached = None if fresh else _find_viewport_cache()
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

    # Detect orange square — ORANGE_HSV_RANGE is the same table
    # _detect_orange_dot matches against, so the two can't drift.
    # Known CSS position: top-left (100, 200), size 50px → center (125, 225)
    SQUARE_CSS_X, SQUARE_CSS_Y, SQUARE_CSS_SIZE = 100, 200, 50

    mask = hsv_mask(to_hsv(img), [ORANGE_HSV_RANGE])
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
        ext = "png" if data[:4] == b"\x89PNG" else "jpg"
        out = VIEWPORT_CACHE_STEM.with_suffix(f".{ext}")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(data)
        log.info(f"  Cached screenshot: {out}")
    return transform


# ─── Camera frame calibration ───────────────────────────────


def _pick_rotation_from_markers(frame: np.ndarray) -> tuple[int, str]:
    """Locate blue UP and red RIGHT markers, derive the cv2 rotation code.

    Returns (rotation_code, human_label). Rotation is -1 (none) or one of
    cv2.ROTATE_{90_CLOCKWISE, 180, 90_COUNTERCLOCKWISE}. Raises if either
    marker is missing.
    """
    def _blob(lower, upper=None):
        return find_largest_hsv_blob(
            frame, lower, upper, min_area=500,
            morph_op=cv2.MORPH_CLOSE, morph_kernel=(15, 15),
        )

    # UP = blue (#2563eb → H≈110). RIGHT = red (#ef4444), which straddles the
    # 0/180 hue seam — red_ranges covers both ends, since the camera commonly
    # renders the on-screen red near the high end (H≈175), not near 0.
    up = _blob([100, 80, 80], [130, 255, 255])
    if up is None:
        raise RuntimeError("UP (blue) marker not found")
    right = _blob(red_ranges(80, 80))
    if right is None:
        raise RuntimeError("RIGHT (red) marker not found")
    up_x, up_y = up
    right_x, right_y = right

    log.info(f"  Blue UP at ({up_x:.0f}, {up_y:.0f}), red RIGHT at ({right_x:.0f}, {right_y:.0f})")

    if up_y < right_y and abs(up_x - right_x) < abs(up_y - right_y):
        return -1, "0° — no rotation needed"
    if up_x < right_x and abs(up_y - right_y) < abs(up_x - right_x):
        return cv2.ROTATE_90_CLOCKWISE, "90° clockwise"
    if up_y > right_y and abs(up_x - right_x) < abs(up_y - right_y):
        return cv2.ROTATE_180, "180°"
    return cv2.ROTATE_90_COUNTERCLOCKWISE, "90° counter-clockwise"


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
    max_retries: int = 3,
) -> dict | None:
    """Move to (gx, gy), tap, return the touch dict (or None on failure).

    On a miss (no touch registered) the solenoid is simply re-fired — the
    stroke is fixed, so there's no depth to deepen; misses are just flaky
    contact or a brief unresponsive screen.
    """
    arm._fast_move(gx, gy)
    arm.wait_idle()
    for attempt in range(max_retries + 1):
        cal.flush_touches()
        _tap_once(arm)
        time.sleep(0.3)
        got = cal.flush_touches()
        if got:
            return got[-1]
        if attempt < max_retries:
            log.warning(
                f"    tap at arm ({gx:.1f}, {gy:.1f})mm: missed, "
                f"retry {attempt + 1}/{max_retries}"
            )
    log.warning(
        f"    tap at arm ({gx:.1f}, {gy:.1f})mm: FAILED after {max_retries} retries"
    )
    return None


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
) -> tuple[np.ndarray, float, list[dict]]:
    """Arm calibration — screen↔arm affine + tilt diagnostic.

    1. Probe triangle: 3 taps at arm (0,0), (+10,0), (0,+10), re-firing on
       a miss. Yields a bootstrap screen→arm affine.
    2. Grid: for each of the 15 viewport grid positions predicted via the
       bootstrap affine, tap (re-fire on miss).
    3. Fit the final affine from all 18 (arm mm, screen 0-1) pairs, derive
       the tilt ratio, re-origin the arm at screen center.

    Each tap fires the solenoid — there is no Z depth to find or bump.

    Returns ``(pct_to_grbl, tilt_ratio, grid_touches)``. Tilt
    ``< TILT_ALIGNED_THRESHOLD`` means arm and phone axes are aligned;
    higher means the phone is rotated relative to arm travel.
    """
    log.info("═══ Arm calibration — screen↔arm mapping ═══")
    cal.set_phase("center")
    time.sleep(0.5)

    # Probe triangle — bootstrap the screen→arm mapping.
    log.info(f"  Probe triangle: 3 taps at (0,0), (+{PROBE_D:.0f},0), (0,+{PROBE_D:.0f})")
    t_center = _tap_and_read(arm, cal, 0, 0)
    if not t_center:
        raise RuntimeError("Arm calibration FAILED — no touch at center")
    t_x = _tap_and_read(arm, cal, PROBE_D, 0)
    if not t_x:
        raise RuntimeError(f"Arm calibration FAILED — no touch at +{PROBE_D:.0f}mm X")
    t_y = _tap_and_read(arm, cal, 0, PROBE_D)
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
        touch = _tap_and_read(arm, cal, gx, gy)
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
        f"  ✓ Arm calibration done: {len(grbl_pts)} pairs, tilt={tilt:.4f}"
    )
    return pct_to_grbl, tilt, grid_touches


# ─── Camera ↔ Screen mapping (Mapping B) ────────────────────


# Mapping B acceptance gates — turn a silently-wrong fit into a loud retry.
# A correct grid reprojects every dot to within a few px; a mis-corresponded
# one lands a full row/column away, so these cleanly separate good from bad
# while tolerating lens distortion and the mild perspective an affine can't
# model.
GRID_FIT_MIN_INLIERS = 14          # of 15; RANSAC homography @ 0.01 screen-0-1
GRID_FIT_MAX_RESIDUAL = 0.04       # max affine reprojection error, camera 0-1
# Outward margin on the corner-bounded screen polygon, as a fraction of the
# shorter frame side — generous enough never to clip a real grid dot.
SCREEN_POLY_MARGIN_FRAC = 0.05


def _detect_screen_region(cam: Camera, rotation: int) -> np.ndarray | None:
    """Grab a frame in the ``corners`` phase and return the screen-bounding
    polygon (camera px, rotated frame) from the RGBM corner blocks — or None if
    fewer than two corners are found (caller then proceeds ungated)."""
    f = cam._fresh_frame()
    if f is None:
        return None
    frame = cv2.rotate(f, rotation) if rotation >= 0 else f
    corners = detect_screen_corners(frame)
    margin = min(frame.shape[:2]) * SCREEN_POLY_MARGIN_FRAC
    poly = screen_polygon(corners, margin=margin)
    log.info(
        f"  Screen corners: detected {len(corners)} "
        f"→ {'polygon gate active' if poly is not None else 'no gate (need ≥2)'}"
    )
    return poly


def _mask_outside(frame: np.ndarray, poly: np.ndarray | None) -> np.ndarray:
    """Black out everything outside ``poly`` so off-screen reflections can't be
    detected as dots. No-op when ``poly`` is None."""
    if poly is None:
        return frame
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [poly.astype(np.int32)], 255)
    return cv2.bitwise_and(frame, frame, mask=mask)


def _fit_grid_mapping(
    screen_pcts: np.ndarray, camera_01: np.ndarray
) -> tuple[np.ndarray | None, int, float]:
    """Fit screen 0-1 → camera 0-1 affine and score the correspondence.

    Returns (pct_to_cam, inliers, max_residual). ``pct_to_cam`` is None when the
    fit fails outright. ``inliers`` counts RANSAC-homography agreements (a
    wrong correspondence falls out as an outlier); ``max_residual`` is the
    largest affine reprojection error in camera 0-1 — it catches a fit too
    skewed for an affine, or a swapped dot the homography still tolerated.
    """
    _, mask = cv2.findHomography(camera_01, screen_pcts, cv2.RANSAC, 0.01)
    inliers = int(mask.sum()) if mask is not None else 0
    pct_to_cam, _ = cv2.estimateAffine2D(screen_pcts, camera_01)
    if pct_to_cam is None:
        return None, inliers, float("inf")
    proj = (pct_to_cam @ np.c_[screen_pcts, np.ones(len(screen_pcts))].T).T
    max_res = float(np.max(np.linalg.norm(proj - camera_01, axis=1)))
    return pct_to_cam, inliers, max_res


def compute_camera_mapping(
    cam: Camera, cal: CalibrationState, rotation: int
) -> tuple[np.ndarray, tuple[int, int]]:
    """Detect 15 red dots, compute screen 0-1 → camera 0-1 affine.

    Returns (pct_to_cam affine (2,3), cam_size (w, h)).
    Both sides are 0-1 normalized.

    Robustness: the screen is first fenced off via the RGBM corner blocks so an
    off-screen reflection can't masquerade as a dot, and every candidate fit is
    gated on homography inliers + reprojection residual — a mis-corresponded
    grid is retried, then fails loudly, never returned as a silently-wrong map.
    """
    log.info("═══ Camera ↔ Screen mapping (Mapping B) ═══")
    log.info("  Goal: compute affine transform from screen 0-1 → camera 0-1")
    expected = len(cal.GRID_COLS_PCT) * len(cal.GRID_ROWS_PCT)

    rot_names = {
        -1: "none",
        cv2.ROTATE_90_CLOCKWISE: "90° CW",
        cv2.ROTATE_180: "180°",
        cv2.ROTATE_90_COUNTERCLOCKWISE: "90° CCW",
    }

    # Known grid target positions (frame-independent), in the same row-major
    # order sort_dots_to_grid yields. Convert to screenshot 0-1 when the
    # viewport shift is known so Mapping B shares Mapping A's coordinate space.
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

    # 1. Fence the screen via the corner blocks (camera is static, so one
    #    capture serves the whole grid pass). A separate frame from the grid:
    #    a corner cluster's red block would otherwise read as a 16th dot.
    cal.set_phase("corners")
    time.sleep(1.0)
    screen_poly = _detect_screen_region(cam, rotation)

    # 2. Grid phase: grab a few fresh frames; accept the first that yields the
    #    full grid inside the screen AND passes the fit-quality gate. Off-screen
    #    reflections are masked out before detection.
    cal.set_phase("grid")
    time.sleep(1.0)
    log.info(
        f"  Phase: grid — phone shows {expected} red dots at known viewport positions"
    )

    best: tuple[np.ndarray, tuple[int, int]] | None = None
    frame = None
    last_dots = 0
    for attempt in range(4):
        f = cam._fresh_frame()
        if f is None:
            time.sleep(0.7)
            continue
        frame = cv2.rotate(f, rotation) if rotation >= 0 else f
        dots = detect_red_dots(_mask_outside(frame, screen_poly), expected=expected)
        last_dots = len(dots)
        log.info(
            f"  Red dot detection (try {attempt + 1}): "
            f"found {len(dots)}/{expected} dots inside the screen"
        )
        if len(dots) != expected:
            time.sleep(0.7)
            continue

        frame_h, frame_w = frame.shape[:2]
        camera_pixels = sort_dots_to_grid(
            dots, rows=len(cal.GRID_ROWS_PCT), cols=len(cal.GRID_COLS_PCT)
        )
        camera_01 = camera_pixels.astype(np.float64)
        camera_01[:, 0] /= frame_w
        camera_01[:, 1] /= frame_h

        pct_to_cam, inliers, max_res = _fit_grid_mapping(screen_pcts, camera_01)
        log.info(
            f"    fit: {inliers}/{expected} homography inliers, "
            f"max reprojection {max_res:.4f} (camera 0-1)"
        )
        if (
            pct_to_cam is not None
            and inliers >= GRID_FIT_MIN_INLIERS
            and max_res <= GRID_FIT_MAX_RESIDUAL
        ):
            log.info(
                f"  Camera frame: {frame_w}×{frame_h}px "
                f"(rotation={rot_names.get(rotation, str(rotation))}); "
                f"mapping {coord_space} ↔ camera 0-1"
            )
            best = (pct_to_cam, (frame_w, frame_h))
            break
        log.warning(
            f"    fit rejected (need ≥{GRID_FIT_MIN_INLIERS} inliers and "
            f"residual ≤{GRID_FIT_MAX_RESIDUAL}) — retrying"
        )
        time.sleep(0.7)

    if frame is None:
        raise RuntimeError("Camera mapping FAILED — camera read failed")
    if best is None:
        raise RuntimeError(
            f"Camera mapping FAILED — no frame produced a clean {expected}-dot "
            f"grid passing the fit gate (last saw {last_dots} dots inside the "
            "screen). Check lighting, focus, the grid page, and that the camera "
            "sees the whole screen."
        )

    pct_to_cam, cam_size = best
    log.info(
        f"  ✓ Camera mapping done: Mapping B ready (screen 0-1 → camera 0-1) "
        f"from {expected} dot pairs, frame {cam_size[0]}×{cam_size[1]}px"
    )
    return pct_to_cam, cam_size


# ─── Full-chain validation ───────────────────────────────────

# Match the orange validation dot to the blob nearest its Mapping-B-predicted
# camera position, within this fraction of the shorter frame side — a stray
# reflection elsewhere can't win, and a match farther than this is rejected so
# the known-position fallback kicks in.
DOT_MATCH_MAX_DIST_FRAC = 0.15
# A back-projected screen pct outside this margin is treated as off-screen and
# rejected, so the arm is never sent past the panel edge.
DOT_OFFSCREEN_LO = -0.05
DOT_OFFSCREEN_HI = 1.05


def validate_calibration(
    arm: StylusArm,
    cam: Camera,
    cal: CalibrationState,
    rotation: int,
    pct_to_grbl: np.ndarray,
    pct_to_cam: np.ndarray,
    cam_size: tuple[int, int] = (1920, 1080),
    num_tests: int = 5,
    max_error: float = 0.015,
) -> list[dict]:
    """Full chain: dot → camera detect → Mapping B → Mapping A → tap → touch.

    Tests BOTH mappings end-to-end:
    1. Page shows orange dot at random position
    2. Camera detects orange dot in frame (camera pixels → normalize to 0-1)
    3. Mapping B⁻¹: camera 0-1 → screen 0-1 pct
    4. Mapping A: screen 0-1 pct → GRBL mm
    5. Arm taps
    6. Phone reports touch coordinate (0-1 pct)
    7. Compare touch vs expected position (in 0-1 space)

    max_error: threshold in 0-1 units (0.015 ≈ 5px on a 390px screen).
    """
    log.info("═══ Full-chain validation ═══")
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
        # Park arm first so it doesn't occlude — the canonical off-phone spot.
        park_gx, park_gy = pct_to_grbl @ np.array([*PARK_PCT, 1.0])
        arm._fast_move(float(park_gx), float(park_gy))
        arm.wait_idle()
        time.sleep(0.3)

        frame = cam._fresh_frame()
        if frame is not None and rotation >= 0:
            frame = cv2.rotate(frame, rotation)

        # We know where the dot should land in the camera (Mapping B forward),
        # so match the orange blob nearest that prediction, not the largest: a
        # reflection or glare elsewhere can dwarf the small on-screen dot, and a
        # largest-blob search would lock onto it and steer the arm off the
        # panel. The distance cap rejects a match too far from the prediction,
        # falling back to the known position instead.
        exp_cam = pct_to_cam @ np.array([expected_x, expected_y, 1.0])
        expected_px = (float(exp_cam[0]) * cam_w, float(exp_cam[1]) * cam_h)
        max_dist = DOT_MATCH_MAX_DIST_FRAC * min(cam_w, cam_h)
        detected = (
            _detect_orange_dot(frame, near=expected_px, max_dist=max_dist)
            if frame is not None
            else None
        )

        if detected is None:
            log.warning(
                "    2. Camera: no orange dot near the predicted spot — "
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
            # Final guard: never drive the arm to a target off the screen. Even
            # with the nearest-blob match, a bad read could back-project past
            # the panel; fall back to the known position instead.
            if not (
                DOT_OFFSCREEN_LO <= cam_pct_x <= DOT_OFFSCREEN_HI
                and DOT_OFFSCREEN_LO <= cam_pct_y <= DOT_OFFSCREEN_HI
            ):
                log.warning(
                    f"    3. Back-projection ({cam_pct_x:.3f}, {cam_pct_y:.3f}) "
                    "is off-screen — rejecting, using known position"
                )
                cam_pct_x, cam_pct_y = expected_x, expected_y

        # 4. Mapping A: screen pct → GRBL mm
        grbl_pos = pct_to_grbl @ np.array([cam_pct_x, cam_pct_y, 1.0])
        gx, gy = float(grbl_pos[0]), float(grbl_pos[1])
        log.info(
            f"    4. Mapping A: screen ({cam_pct_x:.3f}, {cam_pct_y:.3f}) → "
            f"arm ({gx:.1f}, {gy:.1f})mm"
        )

        # 5. Arm taps (re-fire the solenoid on miss — no depth to bump)
        arm._fast_move(gx, gy)
        arm.wait_idle()
        touch = None
        for attempt in range(4):
            cal.flush_touches()
            _tap_once(arm)
            time.sleep(0.3)
            got = cal.flush_touches()
            if got:
                touch = got[-1]
                break
            log.warning(
                f"    5. Tap: missed at arm ({gx:.1f}, {gy:.1f})mm, "
                f"retry {attempt + 1}/3"
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


# ─── AssistiveTouch screenshot verification ─────────────────


def verify_assistive_touch(
    arm: StylusArm,
    at: AssistiveTouch,
    bridge: BridgeState,
    cal: CalibrationState,
    pct_to_grbl: np.ndarray,
    phone: PageState,
) -> dict:
    """Verify all three AT gestures end-to-end.

    1. Single-tap → "PhysiClaw Tap" Shortcut takes a screenshot (saved to Photos).
    2. Wait 5s for the screenshot animation to finish.
    3. Double-tap → "PhysiClaw Screenshot" Shortcut uploads the latest photo.
       Verify the uploaded image contains the grey nonce grid.
    4. Long-press → "PhysiClaw Clipboard" Shortcut GETs /api/bridge/clipboard.
       Verify the server's clipboard-copied event fires within the timeout.
       Return the queued text so the user can paste-verify downstream.

    Requires:
    - assistive-touch/show has run (generated the nonce bits) — this call
      re-displays the grid itself, so the phone may be in any mode on entry
    - User has positioned AT at the orange circle
    - cal.viewport_shift is set (from pre-cal)

    Leaves the phone on the plain bridge page after the clipboard confirmation.

    Returns:
        {
          "passed": bool,  # True iff both sub-checks passed
          "screenshot": {"passed": bool, "matched": int, "total": int},
          "clipboard":  {"fetched": bool, "text": str | None},
        }
    """
    if at.at_screen is None:
        raise RuntimeError("AT position not set — call compute_at_screen_pos first")

    log.info("═══ AssistiveTouch screenshot verification ═══")
    log.info(
        f"  AT position: screen 0-1 ({at.at_screen[0]:.3f}, {at.at_screen[1]:.3f})"
    )

    nonce = cal._screenshot_nonce
    if nonce is None:
        raise RuntimeError("No nonce set — call assistive-touch/show first")

    bridge.clear_screenshot()

    # Show the nonce grid for the screenshot sub-step, whatever mode the
    # page was left in — a prior verify ends in bridge mode after its
    # clipboard confirmation. Re-establishing it here makes this call
    # self-sufficient instead of relying on assistive-touch/show still
    # being on-screen.
    phone.set_mode("calibrate", "assistive_touch", nonce_bits=nonce)
    time.sleep(0.5)  # let the phone poll and render the grid before the shot

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

    shot_passed, matched = verify_nonce(img, t, nonce, css_x=cal.nonce_x())

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
    # Flip the phone to bridge mode so it shows the text, then the
    # "✓ written to clipboard — paste to verify" confirmation once the
    # Shortcut fetches it. The AT circle / nonce grid were only needed
    # for the screenshot sub-steps; the long-press lands on the iOS AT
    # button regardless of what the web page is showing underneath.
    phone.set_mode("bridge")
    bridge.send_text(clip_text)

    log.info("  Long-press AT (iOS Shortcut → fetch bridge text)...")
    at.long_press(arm, pct_to_grbl)

    log.info("  Waiting for iOS Shortcut to fetch clipboard text...")
    clip_fetched = bridge.wait_clipboard(timeout=10.0)
    if clip_fetched:
        log.info(f"  ✓ Clipboard fetched from server — text: {clip_text!r}")
        log.info("    Phone now shows the copied text — paste anywhere to verify.")
        # Hold the confirmation on-screen so the operator can read it
        # before we clear the text out.
        time.sleep(3.0)
    else:
        log.warning("  ✗ Clipboard fetch timed out — server was not hit")

    # Clear the queued text — the phone drops back to the plain bridge page
    # ("PhysiClaw"). We deliberately stay in bridge mode rather than
    # restoring the calibration grid: a re-run re-establishes its own grid at
    # the screenshot sub-step above, so ending here keeps the success screen
    # clean instead of flashing the grid box back up.
    bridge.clear_text()

    passed = shot_passed and clip_fetched
    if passed:
        log.info("  ✓ AssistiveTouch done: tap + double-tap + long-press all verified")
    else:
        log.warning("  ✗ AssistiveTouch verification failed")

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
