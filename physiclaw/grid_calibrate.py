"""
Grid calibration — red dot detection, affine transforms, coordinate mapping.

Phase 6 of calibration: after phases 1-5 complete, the pen-calib page shows
a 3×5 grid of red dots at known screen percentages. This module detects
those dots, visits each with the arm, and computes affine transforms that
map screen percentages to GRBL mm coordinates and camera pixel coordinates.

The transforms enable coordinate-based tapping: the AI agent specifies a
target as a bounding box in screen percentages, and the arm moves directly
to the center without iterative move+check.
"""

import dataclasses
import logging
import time

import cv2
import numpy as np

from physiclaw.camera import Camera
from physiclaw.stylus_arm import StylusArm

log = logging.getLogger(__name__)

# ─── Grid layout (must match pen-calib page) ──────────────────

# 3 columns × 5 rows = 15 dots
# Each entry is (x_percent, y_percent) of the phone screen
GRID_COLS_PCT = [25.0, 50.0, 75.0]
GRID_ROWS_PCT = [20.0, 40.0, 50.0, 60.0, 80.0]
GRID_ROWS = len(GRID_ROWS_PCT)
GRID_COLS = len(GRID_COLS_PCT)

GRID_SCREEN_PCT = np.array(
    [[x, y] for y in GRID_ROWS_PCT for x in GRID_COLS_PCT],
    dtype=np.float64,
)  # shape (15, 2) — row-major order

CENTER_INDEX = 7  # (50%, 50%)

# Visit order: center ring first, then bottom row, then top row.
# Index map (row-major):
#   Row 20%: 0=(25,20)  1=(50,20)  2=(75,20)
#   Row 40%: 3=(25,40)  4=(50,40)  5=(75,40)
#   Row 50%: 6=(25,50)  7=(50,50)  8=(75,50)
#   Row 60%: 9=(25,60) 10=(50,60) 11=(75,60)
#   Row 80%:12=(25,80) 13=(50,80) 14=(75,80)
#
# Phase 6 handles center(7), right(8), down(10) separately via probing.
# Then visits the rest in this order:
VISIT_RING = [6, 4, 5, 3, 9, 11]     # remaining 6 dots around center
VISIT_BOTTOM = [13, 12, 14]           # 80% row (center first)
VISIT_TOP = [1, 0, 2]                 # 20% row (center first)
VISIT_REMAINING = VISIT_RING + VISIT_BOTTOM + VISIT_TOP

# ─── Red dot detection ────────────────────────────────────────


def detect_red_dots(frame: np.ndarray) -> list[tuple[float, float]]:
    """Detect red dots in a camera frame.

    Returns list of (cx, cy) pixel coordinates of detected dot centers.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Red wraps around in HSV — need two ranges
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = mask1 | mask2

    # Clean up noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    dots = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 50 or area > 10000:
            continue
        # Circularity check
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        if circularity < 0.5:
            continue
        m = cv2.moments(cnt)
        if m['m00'] == 0:
            continue
        cx = m['m10'] / m['m00']
        cy = m['m01'] / m['m00']
        dots.append((cx, cy))

    log.debug(f"Detected {len(dots)} red dots")
    return dots


def sort_dots_to_grid(dots: list[tuple[float, float]],
                      rows: int = GRID_ROWS,
                      cols: int = GRID_COLS) -> np.ndarray:
    """Sort detected dot centroids into row-major grid order.

    Returns shape (rows*cols, 2) array.
    Raises RuntimeError if dot count doesn't match expected grid.
    """
    expected = rows * cols
    if len(dots) != expected:
        raise RuntimeError(
            f"Expected {expected} red dots but detected {len(dots)}. "
            f"Check lighting, camera focus, and that the grid page is displayed.")

    # Sort by Y to group into rows
    dots_sorted = sorted(dots, key=lambda d: d[1])

    grid = []
    for r in range(rows):
        row_dots = dots_sorted[r * cols:(r + 1) * cols]
        # Sort by X within each row
        row_dots.sort(key=lambda d: d[0])
        grid.extend(row_dots)

    return np.array(grid, dtype=np.float64)  # (15, 2)


# ─── Affine transform computation ─────────────────────────────


def compute_affine_transforms(
    screen_pcts: np.ndarray,
    grbl_positions: np.ndarray,
    camera_pixels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute affine transforms using RANSAC for robustness.

    Args:
        screen_pcts: (N, 2) screen percentages (x%, y%)
        grbl_positions: (N, 2) GRBL mm positions
        camera_pixels: (N, 2) camera pixel positions

    Returns:
        (pct_to_grbl, pct_to_pixel) — each is a (2, 3) affine matrix.
        Apply via: [x_out, y_out] = M @ [x_in, y_in, 1]
    """
    pct_to_grbl, _ = cv2.estimateAffine2D(screen_pcts, grbl_positions)
    pct_to_pixel, _ = cv2.estimateAffine2D(screen_pcts, camera_pixels)

    if pct_to_grbl is None or pct_to_pixel is None:
        raise RuntimeError("Failed to compute affine transforms — "
                           "not enough valid calibration points")

    return pct_to_grbl, pct_to_pixel


# ─── GridCalibration dataclass ─────────────────────────────────


@dataclasses.dataclass
class GridCalibration:
    """Stores and applies grid calibration affine transforms."""

    pct_to_grbl: np.ndarray    # (2, 3) screen% → GRBL mm
    pct_to_pixel: np.ndarray   # (2, 3) screen% → camera pixels

    def bbox_center_pct(self, left: float, right: float,
                        top: float, bottom: float) -> tuple[float, float]:
        """Compute center of a bounding box in screen percentages."""
        return ((left + right) / 2, (top + bottom) / 2)

    def pct_to_grbl_mm(self, x_pct: float, y_pct: float) -> tuple[float, float]:
        """Convert screen percentage to GRBL mm coordinates."""
        pt = np.array([x_pct, y_pct, 1.0])
        result = self.pct_to_grbl @ pt
        return (float(result[0]), float(result[1]))

    def pct_to_cam_pixel(self, x_pct: float, y_pct: float) -> tuple[int, int]:
        """Convert screen percentage to camera pixel coordinates."""
        pt = np.array([x_pct, y_pct, 1.0])
        result = self.pct_to_pixel @ pt
        return (int(result[0]), int(result[1]))

    def bbox_to_pixel_rect(self, left: float, right: float,
                           top: float, bottom: float) -> tuple[tuple[int, int], tuple[int, int]]:
        """Convert bbox screen% to camera pixel rectangle (top-left, bottom-right)."""
        tl = self.pct_to_cam_pixel(left, top)
        br = self.pct_to_cam_pixel(right, bottom)
        return (tl, br)


# ─── Phase 6: grid calibration ────────────────────────────────


PROBE_DISTANCES_MM = [3.0 + i * 1.5 for i in range(16)]  # same as phase 2-3
EXPLORE_OFFSETS_MM = [1.0, -1.0, 2.0, -2.0, 3.0, -3.0]  # spiral out ±3mm (skip 0 — already tried)


def _tap_at(arm: StylusArm, cam: Camera, x: float, y: float,
            z_tap: float) -> bool:
    """Move to absolute position, tap, return True if green flash."""
    cam.wait_for_white()
    time.sleep(0.2)
    arm._fast_move(x, y)
    arm.wait_idle()
    arm._pen_down(z=z_tap)
    time.sleep(0.15)
    arm._pen_up()
    time.sleep(0.3)
    return cam.wait_for_green(timeout=1.0)


def _explore_dot(arm: StylusArm, cam: Camera,
                 cx: float, cy: float, z_tap: float,
                 right_vec: tuple[int, int],
                 down_vec: tuple[int, int]) -> tuple[float, float] | None:
    """Explore around (cx, cy) to find a dot. Dots are ≥5mm apart,
    so any green within ±3mm must be the target dot.

    Returns the GRBL (x, y) that triggered the green, or None.
    """
    rx, ry = right_vec
    dx, dy = down_vec
    for dr in EXPLORE_OFFSETS_MM:
        for dd in EXPLORE_OFFSETS_MM:
            x = cx + dr * rx + dd * dx
            y = cy + dr * ry + dd * dy
            if _tap_at(arm, cam, x, y, z_tap):
                return arm.position()
    return None


def _probe_dot(arm: StylusArm, cam: Camera, direction: tuple[int, int],
               z_tap: float) -> float | None:
    """Probe outward from origin in a direction until green flash.

    Returns the mm distance that triggered the green flash, or None.
    """
    ax, ay = direction
    for dist in PROBE_DISTANCES_MM:
        x = round(ax * dist, 2)
        y = round(ay * dist, 2)
        if _tap_at(arm, cam, x, y, z_tap):
            log.debug(f"  Probe hit at {dist:.1f}mm")
            return dist
    return None


def _go_center(arm: StylusArm):
    arm._fast_move(0, 0)
    arm.wait_idle()


def phase6_grid(arm: StylusArm, cam: Camera,
                right_vec: tuple[int, int],
                down_vec: tuple[int, int]) -> GridCalibration:
    """Phase 6: grid calibration using red dot page.

    Strategy — probe outward from center, no ambiguity at each step:
    1. Park, detect 15 red dots in camera (pixel positions)
    2. Verify center dot (50%,50%) — explore ±3mm if miss, reset origin
    3. Probe right with small steps → must hit (75%,50%), the only dot
       to the right in center row → mm_per_pct_h = dist / 25
    4. Probe down with small steps → must hit (50%,60%), the nearest dot
       below center in center column → mm_per_pct_v = dist / 10
    5. Visit remaining 12 dots: center ring, then bottom row, then top row
    6. Compute affine transforms from all 15 verified positions
    """
    log.info("Phase 6: Grid calibration  (15 red dots)")

    z_tap = arm.Z_DOWN
    rx, ry = right_vec
    dx, dy = down_vec

    grbl_positions = np.zeros((len(GRID_SCREEN_PCT), 2), dtype=np.float64)
    verified = 0

    # 1. Park arm, wait for grid page, detect dots
    park_vec = arm.MOVE_DIRECTIONS['top']
    arm._fast_move(park_vec[0] * 100, park_vec[1] * 100)
    arm.wait_idle()

    grid_px = None
    deadline = time.time() + 3.0
    while time.time() < deadline:
        frame = cam.snapshot()
        if frame is None:
            raise RuntimeError("Phase 6 FAILED — camera capture failed")
        dots = detect_red_dots(frame)
        if len(dots) == GRID_ROWS * GRID_COLS:
            grid_px = sort_dots_to_grid(dots)
            break
        time.sleep(0.3)

    if grid_px is None:
        raise RuntimeError(
            f"Phase 6 FAILED — expected {GRID_ROWS * GRID_COLS} red dots "
            f"but detected {len(dots)}. Is the grid page displayed?")
    log.info(f"  Detected {len(dots)} dots, sorted into {GRID_ROWS}×{GRID_COLS} grid")

    # 2. Verify center dot (index 7: 50%, 50%)
    _go_center(arm)
    log.info("  Verifying center dot...")
    if _tap_at(arm, cam, 0, 0, z_tap):
        log.info("  Center confirmed at origin")
    else:
        log.info("  Center missed — exploring ±3mm...")
        center_pos = _explore_dot(arm, cam, 0, 0, z_tap, right_vec, down_vec)
        if center_pos is None:
            raise RuntimeError("Phase 6 FAILED — could not find center dot")
        arm._fast_move(center_pos[0], center_pos[1])
        arm.wait_idle()
        arm.set_origin()
        log.info(f"  Origin reset — was off by "
                 f"({center_pos[0]:.2f}, {center_pos[1]:.2f})mm")
    grbl_positions[CENTER_INDEX] = [0.0, 0.0]
    verified += 1

    # 3. Probe right → hit (75%, 50%) — only dot right of center in row 50%
    _go_center(arm)
    log.info("  Probing right → (75%, 50%)...")
    right_dist = _probe_dot(arm, cam, right_vec, z_tap)
    if right_dist is None:
        raise RuntimeError("Phase 6 FAILED — could not hit (75%, 50%)")
    mm_per_pct_h = right_dist / 25.0
    grbl_positions[8] = list(arm.position())  # index 8 = (75%, 50%)
    verified += 1
    log.info(f"  Horizontal: {right_dist:.1f}mm / 25% = {mm_per_pct_h:.3f} mm/pct")

    # 4. Probe down → hit (50%, 60%) — nearest dot below center in column 50%
    _go_center(arm)
    log.info("  Probing down → (50%, 60%)...")
    down_dist = _probe_dot(arm, cam, down_vec, z_tap)
    if down_dist is None:
        raise RuntimeError("Phase 6 FAILED — could not hit (50%, 60%)")
    mm_per_pct_v = down_dist / 10.0
    grbl_positions[10] = list(arm.position())  # index 10 = (50%, 60%)
    verified += 1
    log.info(f"  Vertical: {down_dist:.1f}mm / 10% = {mm_per_pct_v:.3f} mm/pct")

    _go_center(arm)

    # 5. Visit remaining 12 dots: center ring → bottom → top
    for i in VISIT_REMAINING:
        x_pct, y_pct = GRID_SCREEN_PCT[i]
        h_offset = (x_pct - 50) * mm_per_pct_h
        v_offset = (y_pct - 50) * mm_per_pct_v
        target_x = h_offset * rx + v_offset * dx
        target_y = h_offset * ry + v_offset * dy

        if _tap_at(arm, cam, target_x, target_y, z_tap):
            # Direct hit
            grbl_positions[i] = list(arm.position())
            verified += 1
            log.debug(f"  Dot {i} ({x_pct}%, {y_pct}%) ✓ "
                      f"GRBL ({grbl_positions[i][0]:.2f}, {grbl_positions[i][1]:.2f})")
        else:
            # Miss — explore ±3mm around target
            pos = _explore_dot(arm, cam, target_x, target_y, z_tap,
                               right_vec, down_vec)
            if pos is not None:
                grbl_positions[i] = [pos[0], pos[1]]
                verified += 1
                log.debug(f"  Dot {i} ({x_pct}%, {y_pct}%) ✓ (explored) "
                          f"GRBL ({pos[0]:.2f}, {pos[1]:.2f})")
            else:
                grbl_positions[i] = [target_x, target_y]
                log.debug(f"  Dot {i} ({x_pct}%, {y_pct}%) ✗ "
                          f"estimate ({target_x:.2f}, {target_y:.2f})")

    log.info(f"  Verified {verified}/{len(GRID_SCREEN_PCT)} dots")
    if verified < 3:
        raise RuntimeError(
            f"Phase 6 FAILED — only {verified}/15 dots verified, need at least 3")

    _go_center(arm)

    # 6. Compute affine transforms with RANSAC
    pct_to_grbl, pct_to_pixel = compute_affine_transforms(
        GRID_SCREEN_PCT, grbl_positions, grid_px)

    cal = GridCalibration(pct_to_grbl=pct_to_grbl, pct_to_pixel=pct_to_pixel)

    # 7. Save debug image: draw phone screen boundary and grid on camera frame
    debug = frame.copy()
    # Green rectangle: phone screen edges
    corners_pct = [(0, 0), (100, 0), (100, 100), (0, 100)]
    corners_px = [cal.pct_to_cam_pixel(x, y) for x, y in corners_pct]
    for j in range(4):
        cv2.line(debug, corners_px[j], corners_px[(j + 1) % 4], (0, 255, 0), 2)
    # Green edge lines along the clockwise check path
    edge_pcts = [(50, 0), (100, 0), (100, 50), (100, 100),
                 (50, 100), (0, 100), (0, 50), (0, 0), (50, 0)]
    edge_px = [cal.pct_to_cam_pixel(x, y) for x, y in edge_pcts]
    for j in range(len(edge_px) - 1):
        cv2.line(debug, edge_px[j], edge_px[j + 1], (0, 255, 255), 1)
    # Blue circles: 15 calibrated dot positions
    for i in range(len(GRID_SCREEN_PCT)):
        x_pct, y_pct = GRID_SCREEN_PCT[i]
        px = cal.pct_to_cam_pixel(x_pct, y_pct)
        cv2.circle(debug, px, 5, (255, 0, 0), -1)
    from physiclaw.camera import SNAPSHOT_DIR
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    debug_path = SNAPSHOT_DIR / 'grid_calibration.jpg'
    cv2.imwrite(str(debug_path), debug)
    log.info(f"Phase 6 done — debug image saved to {debug_path}")

    # 8. Visual verification: move arm to 4 screen edges for dev to check
    check_points = [
        (50, 0, "top center"),
        (100, 0, "top right"),
        (100, 50, "right center"),
        (100, 100, "bottom right"),
        (50, 100, "bottom center"),
        (0, 100, "bottom left"),
        (0, 50, "left center"),
        (0, 0, "top left"),
        (50, 0, "top center"),  # close the loop
    ]
    _go_center(arm)
    log.info("  Visual check — tracing phone edge clockwise...")
    for x_pct, y_pct, label in check_points:
        gx, gy = cal.pct_to_grbl_mm(x_pct, y_pct)
        log.info(f"    → {label} ({x_pct}%, {y_pct}%) = GRBL ({gx:.2f}, {gy:.2f})")
        arm._fast_move(gx, gy)
        arm.wait_idle()
        time.sleep(2)

    _go_center(arm)
    log.info("  Visual check done")

    return cal
