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


MAX_RETRIES = 10
PROBE_DISTANCES_MM = [3.0 + i * 1.5 for i in range(16)]  # same as phase 2-3


def _probe_dot(arm: StylusArm, cam: Camera, direction: tuple[int, int],
               z_tap: float) -> float | None:
    """Probe outward from origin in a direction until green flash.

    Returns the mm distance that triggered the green flash, or None.
    Used to find the scale factor for the first adjacent dot.
    """
    ax, ay = direction
    cam.wait_for_white()
    for dist in PROBE_DISTANCES_MM:
        x = round(ax * dist, 2)
        y = round(ay * dist, 2)
        arm._fast_move(x, y)
        arm.wait_idle()
        arm._pen_down(z=z_tap)
        time.sleep(0.15)
        arm._pen_up()
        time.sleep(0.3)
        if cam.wait_for_green(timeout=1.0):
            log.debug(f"  Probe hit at {dist:.1f}mm")
            cam.wait_for_white()
            return dist
    return None


def _tap_and_verify(arm: StylusArm, cam: Camera, x: float, y: float,
                    z_tap: float) -> bool:
    """Move to position, tap, return True if green flash detected."""
    arm._fast_move(x, y)
    arm.wait_idle()
    arm._pen_down(z=z_tap)
    time.sleep(0.15)
    arm._pen_up()
    time.sleep(0.3)
    hit = cam.wait_for_green(timeout=1.0)
    if hit:
        cam.wait_for_white()
    return hit


def phase6_grid(arm: StylusArm, cam: Camera,
                right_vec: tuple[int, int],
                down_vec: tuple[int, int]) -> GridCalibration:
    """Phase 6: grid calibration using red dot page.

    Each red dot shows a green flash when tapped, allowing verified
    measurement of GRBL positions. The algorithm:
    1. Detect 15 dots in camera image (pixel positions)
    2. Probe rightward from center to find mm-per-25%-horizontal scale
    3. Probe downward from center to find mm-per-10%-vertical scale
    4. Visit all 15 dots, tap each, verify green flash, record GRBL position
    5. Compute affine transforms from verified (screen%, GRBL mm) pairs

    Prerequisites:
        - Phases 1-5 complete (Z-depth and direction mapping set)
        - Pen-calib page showing the red dot grid on the phone
        - Arm at GRBL (0, 0) = center of the grid
    """
    log.info("Phase 6: Grid calibration  (15 red dots)")

    z_tap = arm.Z_DOWN

    # 1. Park arm out of the way, take photo and detect dots
    park_vec = arm.MOVE_DIRECTIONS['top']
    arm._fast_move(park_vec[0] * 100, park_vec[1] * 100)
    arm.wait_idle()

    frame = cam.snapshot()
    if frame is None:
        raise RuntimeError("Phase 6 FAILED — camera capture failed")

    dots = detect_red_dots(frame)
    grid_px = sort_dots_to_grid(dots)
    log.info(f"  Detected {len(dots)} dots, sorted into {GRID_ROWS}×{GRID_COLS} grid")

    # Return to center
    arm._fast_move(0, 0)
    arm.wait_idle()

    # 2. Probe right to find mm distance for 25% horizontal offset
    #    Center (50%) → right neighbor (75%) = 25% apart
    log.info("  Probing right for scale...")
    right_dist = _probe_dot(arm, cam, right_vec, z_tap)
    if right_dist is None:
        raise RuntimeError("Phase 6 FAILED — could not hit right dot from center")
    mm_per_pct_h = right_dist / 25.0
    log.info(f"  Horizontal scale: {mm_per_pct_h:.3f} mm/pct ({right_dist:.1f}mm for 25%)")

    # Return to center
    arm._fast_move(0, 0)
    arm.wait_idle()
    time.sleep(0.3)

    # 3. Probe down to find mm distance for 10% vertical offset
    #    Center (50%) → down neighbor (60%) = 10% apart
    log.info("  Probing down for scale...")
    down_dist = _probe_dot(arm, cam, down_vec, z_tap)
    if down_dist is None:
        raise RuntimeError("Phase 6 FAILED — could not hit down dot from center")
    mm_per_pct_v = down_dist / 10.0
    log.info(f"  Vertical scale: {mm_per_pct_v:.3f} mm/pct ({down_dist:.1f}mm for 10%)")

    # Return to center
    arm._fast_move(0, 0)
    arm.wait_idle()
    time.sleep(0.3)

    # 4. Visit all 15 dots, tap each, verify green flash
    rx, ry = right_vec
    dx, dy = down_vec
    grbl_positions = np.zeros((len(GRID_SCREEN_PCT), 2), dtype=np.float64)
    verified = 0

    for i, (x_pct, y_pct) in enumerate(GRID_SCREEN_PCT):
        cam.wait_for_white()
        time.sleep(0.3)

        h_offset = (x_pct - 50) * mm_per_pct_h
        v_offset = (y_pct - 50) * mm_per_pct_v
        grbl_x = h_offset * rx + v_offset * dx
        grbl_y = h_offset * ry + v_offset * dy

        hit = _tap_and_verify(arm, cam, grbl_x, grbl_y, z_tap)
        if hit:
            actual_x, actual_y = arm.position()
            grbl_positions[i] = [actual_x, actual_y]
            verified += 1
            log.debug(f"  Dot {i} ({x_pct}%, {y_pct}%) ✓ GRBL ({actual_x:.2f}, {actual_y:.2f})")
        else:
            # Use computed position as fallback
            grbl_positions[i] = [grbl_x, grbl_y]
            log.debug(f"  Dot {i} ({x_pct}%, {y_pct}%) ✗ using estimate ({grbl_x:.2f}, {grbl_y:.2f})")

    log.info(f"  Verified {verified}/{len(GRID_SCREEN_PCT)} dots")

    if verified < 3:
        raise RuntimeError(f"Phase 6 FAILED — only {verified}/15 dots verified, need at least 3")

    # Return to center
    arm._fast_move(0, 0)
    arm.wait_idle()

    # 5. Compute affine transforms with RANSAC
    pct_to_grbl, pct_to_pixel = compute_affine_transforms(
        GRID_SCREEN_PCT, grbl_positions, grid_px)

    log.info("Phase 6 done — affine transforms computed")
    return GridCalibration(pct_to_grbl=pct_to_grbl, pct_to_pixel=pct_to_pixel)
