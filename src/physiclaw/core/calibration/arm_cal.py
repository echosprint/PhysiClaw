"""Arm calibration — screen↔arm affine (Mapping A) + tilt diagnostic.

Bootstrap first, then refine: a 3-tap probe triangle at known arm
offsets yields a rough screen→arm affine, which is only good enough to
*predict* where the 15 grid positions land in arm mm; the final affine
is refit from all 18 (arm mm, screen 0-1) pairs. Each tap fires the
solenoid — there is no Z depth to find. The fitted linear part also
yields a tilt diagnostic (how far arm travel is rotated from the phone
axes), and the step ends by re-origining the arm at screen center so
every later move is relative to a physically meaningful zero.
"""

import logging
import time

import cv2
import numpy as np

from physiclaw.core.bridge import CalibrationState
from physiclaw.core.calibration._common import (
    _tap_and_read,
    grid_positions,
    require_screen_dimension,
    require_viewport_shift,
)
from physiclaw.core.geometry import apply_affine
from physiclaw.core.hardware.arm import StylusArm

log = logging.getLogger(__name__)

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
    require_viewport_shift(cal)
    require_screen_dimension(cal)
    cal.set_phase("center")
    time.sleep(0.5)

    # Probe triangle — bootstrap the screen→arm mapping.
    log.info(
        f"  Probe triangle: 3 taps at (0,0), (+{PROBE_D:.0f},0), (0,+{PROBE_D:.0f})"
    )
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
    probe_grbl = np.array([[0, 0], [PROBE_D, 0], [0, PROBE_D]], dtype=np.float64)
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
        scr_col, scr_row = cal.viewport_pct_to_screenshot_pct(col, row)
        gx, gy = apply_affine(probe_affine, scr_col, scr_row)
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
        f"  Tilt ratio: {tilt:.4f} (want < {TILT_ALIGNED_THRESHOLD}; aligned={aligned})"
    )
    if not aligned:
        log.warning(
            "  Phone/arm axes are skewed — consider straightening phone "
            "orientation if tilt stays high across reruns."
        )

    # Re-origin at screen center.
    center_gx, center_gy = apply_affine(pct_to_grbl, 0.5, 0.5)
    log.info(
        f"  Re-origin: screen center is at arm "
        f"({center_gx:.2f}, {center_gy:.2f})mm → setting as (0, 0)"
    )
    arm.rapid_to(center_gx, center_gy)
    arm.wait_idle()
    arm.set_origin()
    pct_to_grbl[0, 2] -= center_gx
    pct_to_grbl[1, 2] -= center_gy
    cal.set_phase("center")

    log.info(f"  ✓ Arm calibration done: {len(grbl_pts)} pairs, tilt={tilt:.4f}")
    return pct_to_grbl, tilt, grid_touches
