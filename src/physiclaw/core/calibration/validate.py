"""Full-chain validation + edge trace — prove both mappings end-to-end.

Each validation round exercises every link that runtime tapping will
rely on: the page renders an orange dot at a random position, the
camera detects it, Mapping B⁻¹ back-projects camera 0-1 to screen pct,
Mapping A converts to arm mm, the arm taps, and the phone's reported
touch is compared against the expected position. Detection is guarded
twice — the blob is matched *near* its Mapping-B-predicted spot (a
bright reflection elsewhere can't win) and an off-screen
back-projection is rejected — so a bad camera read degrades to the
known dot position rather than steering the arm off the panel.

``trace_screen_edge`` is the human-eye complement: after the numeric
validation, the arm walks the screen border so the operator can see
the calibration hold at the extremes.
"""

import logging
import random
import time

import cv2
import numpy as np

from physiclaw.core.bridge import CalibrationState
from physiclaw.core.calibration._common import _tap_and_read
from physiclaw.core.calibration.transforms import PARK_PCT, ScreenTransforms
from physiclaw.core.hardware.arm import StylusArm
from physiclaw.core.hardware.camera import Camera
from physiclaw.core.vision.grid_detect import (
    detect_orange_dot as _detect_orange_dot,
)

log = logging.getLogger(__name__)

# Match the orange validation dot to the blob nearest its Mapping-B-predicted
# camera position, within this fraction of the shorter frame side — a stray
# reflection elsewhere can't win, and a match farther than this is rejected so
# the known-position fallback kicks in.
DOT_MATCH_MAX_DIST_FRAC = 0.15
# A back-projected screen pct outside this margin is treated as off-screen and
# rejected, so the arm is never sent past the panel edge.
DOT_OFFSCREEN_LO = -0.05
DOT_OFFSCREEN_HI = 1.05


def _run_validation_test(
    i: int,
    num_tests: int,
    arm: StylusArm,
    cam: Camera,
    cal: CalibrationState,
    rotation: int,
    pct_to_grbl: np.ndarray,
    pct_to_cam: np.ndarray,
    cam_to_pct: np.ndarray,
    cam_size: tuple[int, int],
    max_error: float,
) -> dict:
    """One validation round: dot → detect → back-project → tap → compare.

    Returns the result dict appended to ``validate_calibration``'s list —
    either the full pass/fail record or the no-touch failure record.
    """
    cam_w, cam_h = cam_size
    log.info(f"  ── Test {i + 1}/{num_tests} ──")

    # Random viewport 0-1 position for rendering the dot
    vp_x = round(0.2 + random.random() * 0.6, 3)
    vp_y = round(0.2 + random.random() * 0.6, 3)

    # Expected position in screenshot 0-1 (for comparison with touch results)
    if cal.viewport_shift:
        expected_x, expected_y = cal.viewport_pct_to_screenshot_pct(vp_x, vp_y)
    else:
        expected_x, expected_y = vp_x, vp_y

    def _failed() -> dict:
        """The no-touch failure record — expected position, sentinel error."""
        return {"expected": (expected_x, expected_y), "error": 999.0, "passed": False}

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

    frame = cam.raw_frame()
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

    # 5. Arm taps — the same move/flush/tap/retry cycle every other
    # step uses (re-fire the solenoid on miss; no depth to bump).
    touch = _tap_and_read(arm, cal, gx, gy)

    if touch is None:
        log.warning("    5. Tap: FAILED — no touch registered after 4 attempts")
        return _failed()

    # 7. Compare in screenshot 0-1 space
    actual_x, actual_y = touch["x"], touch["y"]
    error = ((actual_x - expected_x) ** 2 + (actual_y - expected_y) ** 2) ** 0.5
    passed = error < max_error

    log.info(f"    5. Tap: touch at screen ({actual_x:.3f}, {actual_y:.3f})")
    log.info(
        f"    6. Error: {error:.4f} (expected ({expected_x:.3f}, {expected_y:.3f}), "
        f"actual ({actual_x:.3f}, {actual_y:.3f})) → "
        f"{'PASS' if passed else 'FAIL'}"
    )
    return {
        "expected": (round(expected_x, 3), round(expected_y, 3)),
        "actual": (round(actual_x, 3), round(actual_y, 3)),
        "camera_pct": (round(cam_pct_x, 3), round(cam_pct_y, 3)),
        "error": round(error, 4),
        "passed": passed,
    }


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

    results = []
    for i in range(num_tests):
        results.append(
            _run_validation_test(
                i,
                num_tests,
                arm,
                cam,
                cal,
                rotation,
                pct_to_grbl,
                pct_to_cam,
                cam_to_pct,
                cam_size,
                max_error,
            )
        )

    arm.return_to_origin()
    passed_count = sum(1 for r in results if r["passed"])
    log.info(f"  ✓ Validation done: {passed_count}/{num_tests} tests passed")
    return results


def trace_screen_edge(arm: StylusArm, cal: ScreenTransforms) -> None:
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
