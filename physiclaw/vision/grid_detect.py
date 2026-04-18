"""
Grid + dot detection — pure CV helpers used during calibration.

Detects red calibration dots (3×5 grid) and orange validation dots in
camera frames. Also provides the affine transform helper that maps screen
percentages to GRBL mm and camera pixels.

Pure functions: frame in → results out. No hardware dependency.
"""

import logging

import cv2
import numpy as np

log = logging.getLogger(__name__)


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
        if m["m00"] == 0:
            continue
        cx = m["m10"] / m["m00"]
        cy = m["m01"] / m["m00"]
        dots.append((cx, cy))

    log.debug(f"Detected {len(dots)} red dots")
    return dots


def sort_dots_to_grid(
    dots: list[tuple[float, float]], rows: int, cols: int
) -> np.ndarray:
    """Sort detected dot centroids into row-major grid order.

    Returns shape (rows*cols, 2) array.
    Raises RuntimeError if dot count doesn't match expected grid.
    """
    expected = rows * cols
    if len(dots) != expected:
        raise RuntimeError(
            f"Expected {expected} red dots but detected {len(dots)}. "
            f"Check lighting, camera focus, and that the grid page is displayed."
        )

    # Sort by Y to group into rows
    dots_sorted = sorted(dots, key=lambda d: d[1])

    grid = []
    for r in range(rows):
        row_dots = dots_sorted[r * cols : (r + 1) * cols]
        # Sort by X within each row
        row_dots.sort(key=lambda d: d[0])
        grid.extend(row_dots)

    return np.array(grid, dtype=np.float64)


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
        raise RuntimeError(
            "Failed to compute affine transforms — not enough valid calibration points"
        )

    return pct_to_grbl, pct_to_pixel


# ─── Orange dot detection (for validation) ───────────────────


def detect_orange_dot(frame: np.ndarray) -> tuple[float, float] | None:
    """Detect a single orange dot in a camera frame.

    Returns (cx, cy) in camera pixels, or None if not found.
    Orange #f97316 ≈ HSV H=20°, S=90%, V=97% → OpenCV H=10, S=230, V=247.
    """
    from physiclaw.vision.util import find_largest_hsv_blob

    return find_largest_hsv_blob(
        frame, [5, 100, 100], [25, 255, 255], min_area=50
    )
