"""Camera ↔ screen mapping (Mapping B) — 15 red dots → affine.

The phone shows 15 red dots at the same canonical grid positions the
arm tapped; detecting them in a rotated camera frame and fitting an
affine gives screen 0-1 → camera 0-1. Two robustness layers turn a
silently-wrong fit into a loud retry: the screen is first fenced off
via the RGBM corner blocks (an off-screen reflection can't masquerade
as a dot), and every candidate fit is gated on RANSAC-homography
inliers plus affine reprojection residual — a mis-corresponded grid
lands a full row/column away, so the gates cleanly separate good from
bad while tolerating lens distortion and mild perspective.
"""

import logging
import time

import cv2
import numpy as np

from physiclaw.core.bridge import CalibrationState
from physiclaw.core.calibration._common import (
    grid_positions,
    require_screen_dimension,
    require_viewport_shift,
)
from physiclaw.core.calibration.state import ROTATION_NAMES
from physiclaw.core.hardware.camera import Camera
from physiclaw.core.vision.grid_detect import (
    detect_red_dots,
    detect_screen_corners,
    screen_polygon,
    sort_dots_to_grid,
)

log = logging.getLogger(__name__)

# Mapping B acceptance gates — turn a silently-wrong fit into a loud retry.
# A correct grid reprojects every dot to within a few px; a mis-corresponded
# one lands a full row/column away, so these cleanly separate good from bad
# while tolerating lens distortion and the mild perspective an affine can't
# model.
GRID_FIT_MIN_INLIERS = 14  # of 15; RANSAC homography @ 0.01 screen-0-1
GRID_FIT_MAX_RESIDUAL = 0.04  # max affine reprojection error, camera 0-1
# Outward margin on the corner-bounded screen polygon, as a fraction of the
# shorter frame side — generous enough never to clip a real grid dot.
SCREEN_POLY_MARGIN_FRAC = 0.05


def _detect_screen_region(cam: Camera, rotation: int) -> np.ndarray | None:
    """Grab a frame in the ``corners`` phase and return the screen-bounding
    polygon (camera px, rotated frame) from the RGBM corner blocks — or None if
    fewer than two corners are found (caller then proceeds ungated)."""
    f = cam.raw_frame()
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

    require_viewport_shift(cal)
    require_screen_dimension(cal)

    # Known grid target positions (frame-independent), in the same row-major
    # order sort_dots_to_grid yields, converted to screenshot 0-1 so
    # Mapping B shares Mapping A's coordinate space.
    screen_pcts = np.array(
        [
            list(cal.viewport_pct_to_screenshot_pct(col, row))
            for col, row in grid_positions(cal)
        ],
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
        f = cam.raw_frame()
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
                f"(rotation={ROTATION_NAMES.get(rotation, str(rotation))}); "
                f"mapping screenshot 0-1 ↔ camera 0-1"
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
