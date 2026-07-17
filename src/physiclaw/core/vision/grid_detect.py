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

from physiclaw.core.vision.blobs import (
    contour_centroid,
    find_all_hsv_blobs,
    find_largest_hsv_blob,
    hsv_blob_centroids,
)
from physiclaw.core.vision.colors import CORNER_HSV_RANGES, ORANGE_HSV_RANGE, redness
from physiclaw.core.vision.preprocess import to_hsv

log = logging.getLogger(__name__)

# R - max(G, B) floor for a calibration dot. Dots are small and render on a
# bright white screen, so the camera desaturates them to a dim pink that an
# HSV saturation floor misses entirely — but their red channel still leads.
# Kept generous (catch faint dots even in a dim room); the median-area trim
# in detect_red_dots then drops the extra glare/noise blobs it lets through.
RED_DOT_MIN_REDNESS = 30

# Contour filters for a calibration dot (small, round). The area bounds
# were tuned on 1920×1080 capture; they scale with the frame's pixel
# count relative to that reference (a 4K frame renders the same physical
# dot with 4× the area — DOT_MAX_AREA would otherwise reject real dots).
# Never scaled below the tuned values: sub-1080p frames keep them as-is.
DOT_MIN_AREA = 50
DOT_MAX_AREA = 10000
DOT_MIN_CIRCULARITY = 0.5
_DOT_AREA_REF_PX = 1920 * 1080


def _dot_area_bounds(frame: np.ndarray) -> tuple[float, float]:
    """(min, max) contour area for a dot, scaled to the frame's resolution."""
    scale = max(1.0, (frame.shape[0] * frame.shape[1]) / _DOT_AREA_REF_PX)
    return DOT_MIN_AREA * scale, DOT_MAX_AREA * scale


# ─── Red dot detection ────────────────────────────────────────


def detect_red_dots(
    frame: np.ndarray, expected: int | None = None
) -> list[tuple[float, float]]:
    """Detect red calibration dots; return (cx, cy) pixel centers.

    Dots are isolated by a redness floor (``R - max(G, B)``) rather than HSV
    saturation, because small dots on a bright screen desaturate to dim pink
    that a saturation floor would miss.

    When ``expected`` is given and more blobs pass the filters than expected,
    the ``expected`` whose area is closest to the median are kept. The grid
    dots are uniform in size, so glare / merged / noise blobs (area outliers)
    fall away — far more robust than betting on one exact threshold.
    """
    mask: np.ndarray = (redness(frame) >= RED_DOT_MIN_REDNESS).astype(np.uint8) * 255

    # Clean up noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    min_area, max_area = _dot_area_bounds(frame)
    cands: list[tuple[float, float, float]] = []  # (cx, cy, area)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            continue
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue
        if 4 * np.pi * area / (perimeter * perimeter) < DOT_MIN_CIRCULARITY:
            continue
        c = contour_centroid(cnt)
        if c is None:
            continue
        cands.append((c[0], c[1], area))

    if expected is not None and len(cands) > expected:
        med = float(np.median([a for *_, a in cands]))
        cands = sorted(cands, key=lambda c: abs(c[2] - med))[:expected]

    log.debug(f"Detected {len(cands)} red dots")
    return [(cx, cy) for cx, cy, _ in cands]


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


# ─── Screen region from the corner blocks ─────────────────────

# A corner cluster (bridge.html `corners` phase) is a 2×2 RGBM block ~20% of
# the phone's shorter side; the four sit at the screen corners, which are far
# apart in frame. 25% of the frame's shorter side groups one cluster's blobs
# without ever merging two corners. Shared by both corner detectors below.
_CORNER_CLUSTER_SPAN_FRAC = 0.25


def _is_clockwise_rgbm_cluster(
    cluster: dict[str, tuple[float, float]], max_span: float
) -> bool:
    """Four RGBM centroids must be tightly grouped and traverse R→G→M→B
    clockwise around their centroid (any cyclic rotation, so any of the
    four camera rotations passes)."""
    from math import atan2

    xs = [p[0] for p in cluster.values()]
    ys = [p[1] for p in cluster.values()]
    if max(xs) - min(xs) > max_span or max(ys) - min(ys) > max_span:
        return False
    cx = sum(xs) / 4
    cy = sum(ys) / 4
    ordered = sorted(
        cluster.items(), key=lambda kv: atan2(kv[1][1] - cy, kv[1][0] - cx)
    )
    return "".join(name for name, _ in ordered) in "RGMBRGMB"


def detect_bridge_corners(
    frame: np.ndarray, max_cluster_span: float | None = None
) -> dict | None:
    """Find one intact RGBM cluster rendered by bridge.html's ``corners``
    phase. bridge.html draws the same 2×2 RGBM cluster at all four
    screen corners, so a stylus occluding up to three of them still
    leaves enough to identify the camera.

    Returns the four centroids of the first intact cluster, or ``None``
    if no combination of detected blobs forms a tight-enough group
    whose clockwise order is a cyclic rotation of RGBM.

    ``max_cluster_span`` defaults to ``_CORNER_CLUSTER_SPAN_FRAC`` of the
    frame's shorter side — each on-phone cluster is ~20% of the phone's
    shorter side (`bs` = 10% per quadrant in bridge.html's ``corners``
    case), and the phone fills 60–80% of the frame during setup, so 25%
    leaves comfortable margin without admitting cross-cluster pairings.
    """
    if max_cluster_span is None:
        max_cluster_span = min(frame.shape[:2]) * _CORNER_CLUSTER_SPAN_FRAC

    hsv = to_hsv(frame)
    blobs = {
        name: hsv_blob_centroids(hsv, hsv_ranges)
        for name, hsv_ranges in CORNER_HSV_RANGES.items()
    }

    if not all(blobs[k] for k in "RGBM"):
        return None

    # Brute force — at most 4 clusters × 1 blob per color per cluster =
    # 4⁴ = 256 candidates. The span filter rejects cross-cluster pairings
    # almost immediately, so this stays cheap in practice.
    for r in blobs["R"]:
        for g in blobs["G"]:
            for b in blobs["B"]:
                for m in blobs["M"]:
                    candidate = {"R": r, "G": g, "B": b, "M": m}
                    if _is_clockwise_rgbm_cluster(candidate, max_cluster_span):
                        return candidate
    return None


def detect_screen_corners(
    frame: np.ndarray, max_cluster_span: float | None = None
) -> list[tuple[float, float]]:
    """Locate the screen-corner RGBM blocks from bridge.html's ``corners``
    phase; return one center per detected corner (camera px), unordered.

    Each corner is a tight 2×2 RGBM cluster. Blobs of every corner colour are
    pooled and grouped by proximity; a group spanning ≥2 of the four colours is
    accepted as a corner and reduced to its centroid. Up to four are returned —
    two diagonal corners already bound the screen, four give the full quad.
    Used to fence dot detection to the screen so off-screen reflections are
    excluded (the cause of a validation dot back-projecting off the panel).
    """
    if max_cluster_span is None:
        max_cluster_span = min(frame.shape[:2]) * _CORNER_CLUSTER_SPAN_FRAC

    hsv = to_hsv(frame)
    pts: list[tuple[float, float, str]] = []
    for name, ranges in CORNER_HSV_RANGES.items():
        for cx, cy in hsv_blob_centroids(hsv, ranges):
            pts.append((cx, cy, name))

    used = [False] * len(pts)
    corners: list[tuple[float, float]] = []
    for i, (sx, sy, _) in enumerate(pts):
        if used[i]:
            continue
        group = [i]
        used[i] = True
        for j in range(i + 1, len(pts)):
            if used[j]:
                continue
            if (
                abs(pts[j][0] - sx) <= max_cluster_span
                and abs(pts[j][1] - sy) <= max_cluster_span
            ):
                group.append(j)
                used[j] = True
        if len({pts[k][2] for k in group}) >= 2:
            mx = sum(pts[k][0] for k in group) / len(group)
            my = sum(pts[k][1] for k in group) / len(group)
            corners.append((mx, my))
    log.debug(f"Detected {len(corners)} screen corners")
    return corners


def screen_polygon(
    corners: list[tuple[float, float]], margin: float = 0.0
) -> np.ndarray | None:
    """Build a closed polygon (Nx2 float32) bounding the screen from detected
    corner centers, grown outward by ``margin`` px about the centroid.

    ≥3 corners → their convex hull (the true quad when all four are found).
    2 corners → the axis-aligned box of that diagonal. <2 → ``None`` (can't
    bound; caller should skip the spatial gate and warn). The outward margin
    keeps a real grid dot near the cluster from being clipped while still
    excluding anything beyond the panel edge.
    """
    if len(corners) < 2:
        return None
    poly: np.ndarray
    if len(corners) == 2:
        (x0, y0), (x1, y1) = corners
        lo_x, hi_x = min(x0, x1), max(x0, x1)
        lo_y, hi_y = min(y0, y1), max(y0, y1)
        poly = np.array(
            [[lo_x, lo_y], [hi_x, lo_y], [hi_x, hi_y], [lo_x, hi_y]],
            dtype=np.float32,
        )
    else:
        poly = cv2.convexHull(np.array(corners, dtype=np.float32)).reshape(-1, 2)

    if margin:
        c = poly.mean(axis=0)
        v = poly - c
        norms = np.linalg.norm(v, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        poly = c + v * (1.0 + margin / norms)
    return poly.astype(np.float32)


def point_in_polygon(poly: np.ndarray | None, x: float, y: float) -> bool:
    """True if (x, y) is inside ``poly`` (or if ``poly`` is None — no gate)."""
    if poly is None:
        return True
    return cv2.pointPolygonTest(poly, (float(x), float(y)), False) >= 0


# ─── Orange dot detection (for validation) ───────────────────

_ORANGE_MIN_AREA = 50


def detect_orange_dot(
    frame: np.ndarray,
    near: tuple[float, float] | None = None,
    *,
    max_dist: float | None = None,
) -> tuple[float, float] | None:
    """Detect the orange validation dot; return (cx, cy) in camera px, or None.

    Without ``near``, returns the *largest* orange blob — fine when the dot is
    the only orange thing in view. (Note ``near`` is a pixel position, unlike
    :func:`detect_red_dots`' ``expected``, which is a dot count.)

    With ``near`` (the dot's predicted camera pixel, e.g. from Mapping B),
    returns the orange blob *nearest* that point instead. This is far more
    robust: a stray orange reflection elsewhere in the frame — even a much
    larger one — can't out-vote the small on-screen dot, which is exactly what
    a largest-blob search wrongly picks. With ``max_dist`` (px), a nearest blob
    farther than that is rejected (returns None), so the caller can fall back to
    the known position rather than steer the arm to a bogus, off-panel target.
    """
    if near is None:
        return find_largest_hsv_blob(
            frame, *ORANGE_HSV_RANGE, min_area=_ORANGE_MIN_AREA
        )
    blobs = find_all_hsv_blobs(frame, *ORANGE_HSV_RANGE, min_area=_ORANGE_MIN_AREA)
    if not blobs:
        return None
    ex, ey = near
    cx, cy = min(blobs, key=lambda b: (b[0] - ex) ** 2 + (b[1] - ey) ** 2)
    if max_dist is not None and (cx - ex) ** 2 + (cy - ey) ** 2 > max_dist * max_dist:
        return None
    return (cx, cy)
