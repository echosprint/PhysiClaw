"""
Grid calibration — red dot detection, affine transforms, coordinate mapping.

Phase 6 of calibration: after phases 1-5 complete, the pen-calib page shows
a 3×5 grid of red dots at known screen positions. This module detects
those dots, visits each with the arm, and computes affine transforms that
map screen coordinates (0-1 decimals) to GRBL mm and camera pixels.

The transforms enable coordinate-based tapping: the AI agent specifies a
target as a bounding box in 0-1 decimals, and the arm moves directly
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
# Each entry is (x, y) as 0-1 decimals of the phone screen
GRID_COLS_PCT = [0.25, 0.50, 0.75]
GRID_ROWS_PCT = [0.20, 0.40, 0.50, 0.60, 0.80]
GRID_ROWS = len(GRID_ROWS_PCT)
GRID_COLS = len(GRID_COLS_PCT)

GRID_SCREEN_PCT = np.array(
    [[x, y] for y in GRID_ROWS_PCT for x in GRID_COLS_PCT],
    dtype=np.float64,
)  # shape (15, 2) — row-major order

CENTER_INDEX = 7  # (0.50, 0.50)

# Visit order: center ring first, then bottom row, then top row.
# Index map (row-major):
#   Row 0.20: 0=(0.25,0.20)  1=(0.50,0.20)  2=(0.75,0.20)
#   Row 0.40: 3=(0.25,0.40)  4=(0.50,0.40)  5=(0.75,0.40)
#   Row 0.50: 6=(0.25,0.50)  7=(0.50,0.50)  8=(0.75,0.50)
#   Row 0.60: 9=(0.25,0.60) 10=(0.50,0.60) 11=(0.75,0.60)
#   Row 0.80:12=(0.25,0.80) 13=(0.50,0.80) 14=(0.75,0.80)
#
# Phase 6 handles center(7), right(8), down(10) separately via probing.
# Then visits the rest in this order:
VISIT_RING = [6, 4, 5, 3, 9, 11]     # remaining 6 dots around center
VISIT_BOTTOM = [13, 12, 14]           # 0.80 row (center first)
VISIT_TOP = [1, 0, 2]                 # 0.20 row (center first)
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
        screen_pcts: (N, 2) screen coordinates as 0-1 decimals (x, y)
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
    """Stores and applies grid calibration affine transforms.

    All coordinates use 0-1 decimals (0=left/top, 1=right/bottom).
    Both mappings work in normalized space:
      - pct_to_grbl:  screen 0-1 → GRBL mm
      - pct_to_cam:   screen 0-1 → camera 0-1
    Camera pixel conversion happens at the boundary via cam_size.
    """

    pct_to_grbl: np.ndarray    # (2, 3) screen 0-1 → GRBL mm
    pct_to_cam: np.ndarray     # (2, 3) screen 0-1 → camera 0-1
    cam_size: tuple[int, int]  # (width, height) of camera frame in pixels

    # Backward compat: accept old kwarg name
    def __init__(self, pct_to_grbl: np.ndarray,
                 pct_to_cam: np.ndarray | None = None,
                 cam_size: tuple[int, int] = (1920, 1080),
                 pct_to_pixel: np.ndarray | None = None):
        self.pct_to_grbl = pct_to_grbl
        self.cam_size = cam_size
        if pct_to_cam is not None:
            self.pct_to_cam = pct_to_cam
        elif pct_to_pixel is not None:
            # Legacy: convert pixel affine to 0-1 affine
            w, h = cam_size
            scale = np.array([[1/w, 0, 0], [0, 1/h, 0]], dtype=np.float64)
            self.pct_to_cam = scale @ np.vstack([pct_to_pixel, [0, 0, 1]])
        else:
            raise ValueError("pct_to_cam or pct_to_pixel required")

    def bbox_center_pct(self, bbox: list[float]) -> tuple[float, float]:
        """Compute center of a bounding box in screen coordinates (0-1).

        Args:
            bbox: [left, top, right, bottom] as 0-1 decimals
        """
        return ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)

    def pct_to_grbl_mm(self, x: float, y: float) -> tuple[float, float]:
        """Convert screen coordinate (0-1) to GRBL mm."""
        pt = np.array([x, y, 1.0])
        result = self.pct_to_grbl @ pt
        return (float(result[0]), float(result[1]))

    def pct_to_cam_pixel(self, x: float, y: float) -> tuple[int, int]:
        """Convert screen coordinate (0-1) to camera pixel."""
        pt = np.array([x, y, 1.0])
        cam_01 = self.pct_to_cam @ pt
        w, h = self.cam_size
        return (int(cam_01[0] * w), int(cam_01[1] * h))

    def pixel_to_pct(self, px_x: int, px_y: int) -> tuple[float, float]:
        """Convert camera pixel to screen coordinate (0-1)."""
        w, h = self.cam_size
        cam_01 = np.array([px_x / w, px_y / h])
        A = self.pct_to_cam[:, :2]  # 2x2
        b = self.pct_to_cam[:, 2]   # translation
        pct = np.linalg.solve(A, cam_01 - b)
        return (float(pct[0]), float(pct[1]))

    def bbox_to_pixel_rect(self, bbox: list[float]) -> tuple[tuple[int, int], tuple[int, int]]:
        """Convert bbox [left, top, right, bottom] (0-1) to camera pixel rectangle."""
        tl = self.pct_to_cam_pixel(bbox[0], bbox[1])
        br = self.pct_to_cam_pixel(bbox[2], bbox[3])
        return (tl, br)


# ─── Phase 6: grid calibration ────────────────────────────────


PROBE_DISTANCES_MM = [3.0 + i * 1.5 for i in range(16)]  # same as phase 2-3

# Square-in-square search: concentric squares expanding from center.
# Each square has 0.5mm step along edges. Max half-edge = 2mm.
EXPLORE_MAX_R = 2.0    # mm — max half-edge of outermost square
EXPLORE_STEP = 0.5     # mm — spacing along each edge and between squares



def _tap_at(arm: StylusArm, cam: Camera, x: float, y: float,
            z_tap: float) -> bool:
    """Move to absolute position, tap, return True if green flash."""
    cam.wait_for_white()
    time.sleep(0.2)
    arm._fast_move(x, y)
    arm.wait_idle()
    arm._pen_down(z=z_tap)
    arm._dwell(0.15)
    arm._pen_up()
    arm.wait_idle()
    return cam.wait_for_green(timeout=1.0)


def _square_points(right_vec, down_vec):
    """Generate concentric square offsets in GRBL mm.

    Expands from inner square to outer: ±0.5mm, ±1.0mm, ±1.5mm, ±2.0mm.
    Each square walks its perimeter in 0.5mm steps.
    Tests nearby positions first, dense coverage, no gaps.
    """
    rx, ry = right_vec
    dx, dy = down_vec
    s = EXPLORE_STEP
    r = s
    while r <= EXPLORE_MAX_R + 0.01:
        # Walk perimeter of square with half-edge = r
        # Top edge: left to right
        col = -r
        while col <= r:
            yield (col * rx + (-r) * dx, col * ry + (-r) * dy)
            col += s
        # Right edge: top to bottom (skip top-right corner)
        row = -r + s
        while row <= r:
            yield (r * rx + row * dx, r * ry + row * dy)
            row += s
        # Bottom edge: right to left (skip bottom-right corner)
        col = r - s
        while col >= -r:
            yield (col * rx + r * dx, col * ry + r * dy)
            col -= s
        # Left edge: bottom to top (skip both corners)
        row = r - s
        while row > -r:
            yield ((-r) * rx + row * dx, (-r) * ry + row * dy)
            row -= s
        r += s


def _explore_dot(arm: StylusArm, cam: Camera,
                 cx: float, cy: float, z_tap: float,
                 right_vec: tuple[int, int],
                 down_vec: tuple[int, int]) -> tuple[float, float] | None:
    """Explore around (cx, cy) using concentric squares.

    Squares expand outward from ±0.5mm to ±2mm in 0.5mm steps.
    Each square's perimeter is walked in 0.5mm increments.
    Dots are ≥5mm apart, so any green within range must be the target.

    Returns the GRBL (x, y) that triggered the green, or None.
    """
    for ox, oy in _square_points(right_vec, down_vec):
        if _tap_at(arm, cam, cx + ox, cy + oy, z_tap):
            return arm.position()
    return None


def _probe_dot(arm: StylusArm, cam: Camera, direction: tuple[int, int],
               z_tap: float, start_dist: float = 0.0,
               max_dist: float = 25.5) -> float | None:
    """Probe outward from origin in a direction until green flash.

    Args:
        start_dist: skip distances before this value (jump ahead).
        max_dist: stop probing beyond this distance.

    Returns the mm distance that triggered the green flash, or None.
    """
    ax, ay = direction
    for dist in PROBE_DISTANCES_MM:
        if dist < start_dist:
            continue
        if dist > max_dist:
            break
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
                down_vec: tuple[int, int],
                right_dist: float,
                down_dist: float) -> GridCalibration:
    """Phase 6: grid calibration using red dot page.

    Uses phase 2-3 distances as starting points — the right/down calibration
    circles are near the grid dots, so we jump there and explore ±3mm.

    1. Park, detect 15 red dots in camera (pixel positions)
    2. Verify center dot (0.50, 0.50) — explore ±3mm if miss, reset origin
    3. Move right by phase-2 distance, explore → find (0.75, 0.50) dot
    4. Move down by phase-3 distance, explore → find (0.50, 0.60) dot
    5. Visit remaining 12 dots: center ring, then bottom row, then top row
    6. Compute affine transforms from all 15 verified positions

    Args:
        right_dist: mm distance from phase 2 (center to right circle)
        down_dist: mm distance from phase 3 (center to down circle)
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

    # 2. Verify center dot (index 7: 0.50, 0.50)
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

    # 3. Find (0.75, 0.50) — probe right, jumping to phase-2 distance
    _go_center(arm)
    log.info(f"  Probing right → (0.75, 0.50), starting at {right_dist:.1f}mm...")
    grid_right_dist = _probe_dot(arm, cam, right_vec, z_tap,
                                 start_dist=right_dist, max_dist=right_dist + 10)
    if grid_right_dist is None:
        raise RuntimeError("Phase 6 FAILED — could not find (0.75, 0.50)")
    grbl_positions[8] = list(arm.position())
    verified += 1
    mm_per_pct_h = grid_right_dist / 0.25
    log.info(f"  Horizontal: {grid_right_dist:.1f}mm / 0.25 = {mm_per_pct_h:.3f} mm/unit")

    # 4. Find (0.50, 0.60) — probe down, jumping to phase-3 distance
    _go_center(arm)
    log.info(f"  Probing down → (0.50, 0.60), starting at {down_dist:.1f}mm...")
    grid_down_dist = _probe_dot(arm, cam, down_vec, z_tap,
                                start_dist=down_dist, max_dist=down_dist + 10)
    if grid_down_dist is None:
        raise RuntimeError("Phase 6 FAILED — could not find (0.50, 0.60)")
    grbl_positions[10] = list(arm.position())
    verified += 1
    mm_per_pct_v = grid_down_dist / 0.10
    log.info(f"  Vertical: {grid_down_dist:.1f}mm / 0.10 = {mm_per_pct_v:.3f} mm/unit")

    _go_center(arm)

    # 5. Visit remaining 12 dots. After each verified dot, refit the affine
    #    from ALL verified dots so far → better predictions for the next dot.
    verified_indices = [CENTER_INDEX, 8, 10]  # center, right, down

    for i in VISIT_REMAINING:
        x_pct, y_pct = GRID_SCREEN_PCT[i]

        # Predict position using affine from all verified dots so far
        if len(verified_indices) >= 3:
            v_idx = np.array(verified_indices)
            affine, _ = cv2.estimateAffine2D(
                GRID_SCREEN_PCT[v_idx], grbl_positions[v_idx])
            if affine is not None:
                pt = affine @ np.array([x_pct, y_pct, 1.0])
                target_x, target_y = float(pt[0]), float(pt[1])
            else:
                # Fallback to linear scale
                target_x = (x_pct - 0.50) * mm_per_pct_h * rx + (y_pct - 0.50) * mm_per_pct_v * dx
                target_y = (x_pct - 0.50) * mm_per_pct_h * ry + (y_pct - 0.50) * mm_per_pct_v * dy
        else:
            target_x = (x_pct - 0.50) * mm_per_pct_h * rx + (y_pct - 0.50) * mm_per_pct_v * dx
            target_y = (x_pct - 0.50) * mm_per_pct_h * ry + (y_pct - 0.50) * mm_per_pct_v * dy

        if _tap_at(arm, cam, target_x, target_y, z_tap):
            grbl_positions[i] = list(arm.position())
            verified_indices.append(i)
            verified += 1
            log.debug(f"  Dot {i} ({x_pct}, {y_pct}) ✓ "
                      f"GRBL ({grbl_positions[i][0]:.2f}, {grbl_positions[i][1]:.2f})")
        else:
            pos = _explore_dot(arm, cam, target_x, target_y, z_tap,
                               right_vec, down_vec)
            if pos is not None:
                grbl_positions[i] = [pos[0], pos[1]]
                verified_indices.append(i)
                verified += 1
                log.debug(f"  Dot {i} ({x_pct}, {y_pct}) ✓ (explored) "
                          f"GRBL ({pos[0]:.2f}, {pos[1]:.2f})")
            else:
                raise RuntimeError(
                    f"Phase 6 FAILED — dot {i} ({x_pct}, {y_pct}) not found. "
                    f"Tried position ({target_x:.2f}, {target_y:.2f}) and "
                    f"explored ±{EXPLORE_MAX_R:.0f}mm squares. Check phone screen and lighting.")

    log.info(f"  Verified all {verified}/{len(GRID_SCREEN_PCT)} dots")

    _go_center(arm)

    # 6. Compute affine transforms with RANSAC
    pct_to_grbl, pct_to_pixel = compute_affine_transforms(
        GRID_SCREEN_PCT, grbl_positions, grid_px)

    cal = GridCalibration(pct_to_grbl=pct_to_grbl, pct_to_pixel=pct_to_pixel)

    # 7. Save debug image: draw phone screen boundary and grid on camera frame
    debug = frame.copy()
    # Green rectangle: phone screen edges
    corners_pct = [(0, 0), (1, 0), (1, 1), (0, 1)]
    corners_px = [cal.pct_to_cam_pixel(x, y) for x, y in corners_pct]
    for j in range(4):
        cv2.line(debug, corners_px[j], corners_px[(j + 1) % 4], (0, 255, 0), 2)
    # Green edge lines along the clockwise check path
    edge_pcts = [(0.5, 0), (1, 0), (1, 0.5), (1, 1),
                 (0.5, 1), (0, 1), (0, 0.5), (0, 0), (0.5, 0)]
    edge_px = [cal.pct_to_cam_pixel(x, y) for x, y in edge_pcts]
    for j in range(len(edge_px) - 1):
        cv2.line(debug, edge_px[j], edge_px[j + 1], (0, 255, 255), 1)
    # Blue circles: 15 calibrated dot positions
    for i in range(len(GRID_SCREEN_PCT)):
        x_pct, y_pct = GRID_SCREEN_PCT[i]
        px = cal.pct_to_cam_pixel(x_pct, y_pct)
        cv2.circle(debug, px, 5, (255, 0, 0), -1)
    from datetime import datetime
    from physiclaw.camera import SNAPSHOT_DIR
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
    debug_path = SNAPSHOT_DIR / f'{ts}_grid.jpg'
    cv2.imwrite(str(debug_path), debug)
    log.info(f"Phase 6 done — debug image saved to {debug_path}")

    return cal


def trace_screen_edge(arm: StylusArm, cal: GridCalibration):
    """Trace the phone screen border clockwise for visual verification.

    Moves the arm to 8 edge points (top-center → top-right → right-center
    → bottom-right → bottom-center → bottom-left → left-center → top-left
    → back to top-center), pausing 2s at each. Then returns to center.
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
    _go_center(arm)
    log.info("Tracing phone edge clockwise...")
    for x_pct, y_pct, label in check_points:
        gx, gy = cal.pct_to_grbl_mm(x_pct, y_pct)
        log.info(f"  → {label} ({x_pct}, {y_pct}) = GRBL ({gx:.2f}, {gy:.2f})")
        arm._fast_move(gx, gy)
        arm.wait_idle()
        time.sleep(2)

    _go_center(arm)
    log.info("Edge trace done")
