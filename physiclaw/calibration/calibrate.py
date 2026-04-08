"""
Plan-based calibration — touch coordinates + camera detection.

Implements the architecture plan's calibration exactly:

  Step 0: Z-depth via touch event detection (stationary descent)
  Step 1: Arm-phone alignment via touch coordinates
  Step 2: Camera physical rotation check (phone shape in frame)
  Step 3: Software rotation via UP/RIGHT markers
  Step 4: GRBL ↔ Screen mapping via touch coordinates (Mapping A), sets screen center as arm origin
  Step 5: Camera ↔ Screen mapping via red dot detection (Mapping B)
  Step 6: Full-chain validation (camera → Mapping B → Mapping A → tap → touch)

No green flash. Touch events for contact detection and coordinate mapping.
Camera for marker/dot visual detection only.
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

PEN_DEPTH_FILE = "data/pen/z-tap"


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

    This must run before Step 0 so that all subsequent touch coordinates
    are correctly converted from viewport space to screenshot 0-1 space.

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
    return transform


# ─── Step 0: Z-axis surface detection ────────────────────────


def find_pen_depth(arm: StylusArm, cal: CalibrationState) -> float:
    """Probe Z depth with tap-and-release at each level. Plan Step 0.

    Full tap cycle at each Z: pen down, dwell 150ms, pen up.
    After each tap, check if the phone reported a touch event.
    Spring-loaded stylus bounces back on pen-up, so no motor hold needed.

    Returns z_tap (contact depth + margin, validated at ±10mm offsets).
    """
    log.info("═══ Step 0: Z-axis surface detection ═══")
    log.info("  Goal: find the Z depth where stylus reliably registers on touchscreen")
    cal.set_phase("center")
    time.sleep(0.5)
    log.info("  Phase: center — orange circle at screen center as touch target")

    # Clear any stale touches
    cal.flush_touches()

    # Phase A: find first contact by descending in 0.3mm steps
    z_contact = None
    z_steps = [round(0.5 + i * 0.3, 2) for i in range(32)]  # 0.5 to 9.8mm
    log.info(
        f"  Phase A: descending from {z_steps[0]}mm to {z_steps[-1]}mm "
        f"in 0.3mm steps to find first contact"
    )

    for z in z_steps:
        _tap_once(arm, z, z_speed=SLOW_Z_SPEED)
        time.sleep(0.3)

        touches = cal.flush_touches()
        if touches:
            z_contact = z
            log.info(f"  Phase A: first contact at Z={z:.2f}mm — stylus touched screen")
            break
        else:
            log.debug(f"  Phase A: Z={z:.2f}mm — no contact")

    if z_contact is None:
        raise RuntimeError(
            "Step 0 FAILED — no touch detected up to 9.8mm. "
            "Check stylus alignment and phone placement."
        )

    # Phase B: find a Z depth that works at center AND ±10mm in each direction.
    # Taps: center, +X, center, -X, center, +Y, center, -Y (8 taps per round).
    # This catches surface unevenness that center-only testing would miss.
    PROBE_OFFSETS = [
        ("center", 0, 0),
        ("+X", 10, 0),
        ("center", 0, 0),
        ("-X", -10, 0),
        ("center", 0, 0),
        ("+Y", 0, 10),
        ("center", 0, 0),
        ("-Y", 0, -10),
    ]
    z_try = round(z_contact + 0.25, 2)
    z_tap = None
    log.info(
        f"  Phase B: reliability test starting at Z={z_try:.2f}mm "
        f"(first contact {z_contact:.2f}mm + 0.25mm margin)"
    )
    log.info(
        f"  Phase B: tapping center and ±10mm in X/Y — "
        f"{len(PROBE_OFFSETS)} taps per round must all register"
    )

    for _ in range(10):  # max 10 rounds of 0.25mm increases
        log.info(f"  Phase B: testing Z={z_try:.2f}mm ...")
        all_ok = True
        for i, (label, ox, oy) in enumerate(PROBE_OFFSETS):
            arm._fast_move(ox, oy)
            arm.wait_idle()
            cal.flush_touches()
            _tap_once(arm, z_try, z_speed=SLOW_Z_SPEED)
            time.sleep(0.3)
            touches = cal.flush_touches()
            if touches:
                log.debug(
                    f"    tap {i + 1}/{len(PROBE_OFFSETS)} {label} "
                    f"at ({ox}, {oy})mm — registered"
                )
            else:
                log.info(
                    f"    tap {i + 1}/{len(PROBE_OFFSETS)} {label} "
                    f"at ({ox}, {oy})mm — missed"
                )
                all_ok = False
                break

        # Return to center after each round
        arm._fast_move(0, 0)
        arm.wait_idle()

        if all_ok:
            z_tap = z_try
            log.info(
                f"  Phase B: {len(PROBE_OFFSETS)}/{len(PROBE_OFFSETS)} taps "
                f"registered at Z={z_tap:.2f}mm"
            )
            break
        else:
            z_try = round(z_try + 0.25, 2)
            log.info(f"  Phase B: increasing to Z={z_try:.2f}mm")

    if z_tap is None:
        raise RuntimeError(
            "Step 0 FAILED — could not find reliable Z depth. "
            "Check stylus and phone placement."
        )

    log.info(f"  ✓ Step 0 done: z_tap={z_tap}mm (reliable contact depth)")
    return z_tap


# ─── Step 1: Arm-phone alignment check ───────────────────────


def check_arm_tilt(
    arm: StylusArm, cal: CalibrationState, z_tap: float, separation_mm: float = 25.0
) -> float:
    """Two taps along arm X-axis, compare touch Y coords. Plan Step 1.

    Returns tilt_ratio. < 0.02 means aligned (< ~1 degree).
    """
    log.info("═══ Step 1: Arm-phone alignment check ═══")
    log.info(
        f"  Goal: verify arm X-axis is parallel to phone axis "
        f"(2 taps {separation_mm:.0f}mm apart)"
    )
    cal.set_phase("center")
    time.sleep(0.3)

    half = separation_mm / 2
    touches = []
    labels = ["A (left)", "B (right)"]
    for idx, x in enumerate([-half, half]):
        log.info(f"  Tap {labels[idx]}: moving arm to X={x:.1f}mm, Y=0mm")
        arm._fast_move(x, 0)
        arm.wait_idle()
        got = None
        z = z_tap
        for _ in range(4):
            cal.flush_touches()
            _tap_once(arm, z)
            time.sleep(0.3)
            got = cal.flush_touches()
            if got:
                if z > z_tap:
                    z_tap = z  # propagate deeper z
                    log.info(f"  Tap {labels[idx]}: hit after z bump to {z:.2f}mm")
                else:
                    log.info(f"  Tap {labels[idx]}: registered at Z={z:.2f}mm")
                break
            z = round(z + 0.25, 2)
            log.warning(f"  Tap {labels[idx]}: missed, increasing Z to {z:.2f}mm")
        if not got:
            raise RuntimeError(f"Step 1 FAILED — no touch at arm X={x:.1f}mm")
        touches.append(got[-1])

    arm._fast_move(0, 0)
    arm.wait_idle()

    sx1, sy1 = touches[0]["x"], touches[0]["y"]
    sx2, sy2 = touches[1]["x"], touches[1]["y"]
    log.info(f"  Tap A: arm X={-half:.1f}mm → screen pos ({sx1:.3f}, {sy1:.3f})")
    log.info(f"  Tap B: arm X={+half:.1f}mm → screen pos ({sx2:.3f}, {sy2:.3f})")
    dx = abs(sx2 - sx1)
    dy = abs(sy2 - sy1)

    # Arm X may map to phone X or phone Y (90° rotation is fine).
    # Check that the movement is straight: minor axis / major axis < 0.02.
    major = max(dx, dy)
    minor = min(dx, dy)
    if major < 0.01:
        raise RuntimeError(
            "Step 1 FAILED — both taps at same position. "
            "Check arm movement and phone placement."
        )

    tilt = minor / major
    axis_name = "Y" if dy > dx else "X"
    log.info(
        f"  Screen displacement: Δx={dx:.3f}, Δy={dy:.3f} → "
        f"arm X-axis maps to phone {axis_name}-axis"
    )
    log.info(f"  Tilt ratio: {tilt:.4f} (cross-axis / main-axis, want < 0.02)")
    if tilt < 0.02:
        log.info("  ✓ Step 1 done: phone is aligned (tilt < 1°)")
    else:
        log.warning(
            f"  ✗ Step 1: phone is tilted — ratio {tilt:.4f} exceeds 0.02, adjust phone"
        )
    return tilt


# ─── Step 2: Camera physical rotation check ──────────────────


def detect_camera_rotation(cam: Camera, screen_dimension: dict | None = None) -> dict:
    """Check camera orientation, tilt, and coverage. Plan Step 2.

    Checks:
    1. Long axes aligned: phone long axis matches image long axis
    2. No tilt: phone aspect ratio in image ≈ actual screen aspect ratio
    3. Coverage: phone area ≥ 70% of image area

    Returns dict with ok, messages, and measurements.
    """
    log.info("═══ Step 2: Camera physical rotation check ═══")
    log.info("  Goal: verify camera is straight, aligned, and close enough to phone")
    frame = cam._fresh_frame()
    if frame is None:
        raise RuntimeError("Step 2 FAILED — camera read failed")

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise RuntimeError("Step 2 FAILED — no bright region in camera frame")

    largest = max(contours, key=cv2.contourArea)
    # Use minAreaRect for orientation and area (handles rotated phones)
    rect = cv2.minAreaRect(largest)
    rect_w, rect_h = rect[1]
    phone_area_px = rect_w * rect_h
    img_h, img_w = frame.shape[:2]
    image_area = img_w * img_h
    issues = []

    # Save annotated image with actual contour + min-area rect
    annotated = frame.copy()
    cv2.drawContours(annotated, [largest], -1, (0, 255, 0), 3)
    box = cv2.boxPoints(rect)
    box = np.int32(box)
    cv2.drawContours(annotated, [box], -1, (0, 200, 255), 2)
    coverage = phone_area_px / image_area
    label = f"area {coverage:.0%}"
    bx, by, _, _ = cv2.boundingRect(largest)
    cv2.putText(
        annotated, label, (bx + 5, by + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
    )
    cv2.imwrite("/tmp/physiclaw_camera_rotation.jpg", annotated)

    # 1. Camera rotation — phone edges should be parallel to image edges.
    # Compute angle from the longest edge of the min-area rect.
    phone_long = max(rect_w, rect_h)
    phone_short = min(rect_w, rect_h)
    pts = cv2.boxPoints(rect)
    # Find the longest edge and compute its angle to horizontal
    edges = [(pts[i], pts[(i + 1) % 4]) for i in range(4)]
    longest_edge = max(edges, key=lambda e: np.linalg.norm(e[1] - e[0]))
    dx = longest_edge[1][0] - longest_edge[0][0]
    dy = longest_edge[1][1] - longest_edge[0][1]
    # Angle of longest edge relative to horizontal (or vertical)
    angle_deg = abs(np.degrees(np.arctan2(dy, dx)))
    # Normalize to deviation from nearest axis (0°, 90°, 180°)
    rotation_dev = min(angle_deg % 90, 90 - angle_deg % 90)
    rotation_ok = rotation_dev < 3.0  # < 3° deviation
    if not rotation_ok:
        issues.append(
            f"Straighten camera — phone edges rotated {rotation_dev:.1f}° from image edges"
        )
    log.info(
        f"  Camera frame: {img_w}×{img_h}px, phone region: {rect_w:.0f}×{rect_h:.0f}px"
    )
    log.info(
        f"  Check 1 — Edge straightness: {rotation_dev:.1f}° deviation from image axis "
        f"(threshold < 3°) → {'OK' if rotation_ok else 'FAIL — straighten camera'}"
    )

    # 2. Long axes aligned
    image_long_horizontal = img_w > img_h
    bx, by, bw, bh = cv2.boundingRect(largest)
    bbox_long_horizontal = bw > bh
    axes_ok = bbox_long_horizontal == image_long_horizontal
    if not axes_ok:
        issues.append("Rotate camera 90° — long axes not aligned")
    log.info(
        f"  Check 2 — Long axis alignment: image long axis is "
        f"{'horizontal' if image_long_horizontal else 'vertical'}, "
        f"phone long axis is {'horizontal' if bbox_long_horizontal else 'vertical'} → "
        f"{'OK' if axes_ok else 'FAIL — rotate camera 90°'}"
    )

    # 3. Aspect ratio check (tilt detection)
    phone_ratio = phone_long / max(phone_short, 1)
    if screen_dimension:
        screen_w = screen_dimension.get("width", 430)
        screen_h = screen_dimension.get("height", 932)
        expected_ratio = max(screen_w, screen_h) / max(min(screen_w, screen_h), 1)
    else:
        expected_ratio = 2.0  # typical phone ~19.5:9 ≈ 2.17
    ratio_diff = abs(phone_ratio - expected_ratio) / expected_ratio
    tilt_ok = ratio_diff < 0.15
    if not tilt_ok:
        issues.append(
            f"Camera may be tilted — aspect ratio {phone_ratio:.2f} "
            f"vs expected {expected_ratio:.2f} (diff {ratio_diff:.0%})"
        )
    log.info(
        f"  Check 3 — Aspect ratio (tilt detection): phone={phone_ratio:.2f}, "
        f"expected={expected_ratio:.2f}, diff={ratio_diff:.0%} "
        f"(threshold < 15%) → {'OK' if tilt_ok else 'FAIL — camera may be tilted'}"
    )

    # 4. Coverage check (min-area rect ≥ 30% of image area)
    coverage_ok = coverage >= 0.30
    if not coverage_ok:
        issues.append(
            f"Move camera closer — phone covers only {coverage:.0%} of image (need ≥30%)"
        )
    log.info(
        f"  Check 4 — Coverage: phone occupies {coverage:.0%} of camera frame "
        f"(threshold ≥ 30%) → {'OK' if coverage_ok else 'FAIL — move camera closer'}"
    )

    ok = axes_ok and tilt_ok and coverage_ok
    if ok:
        log.info("  ✓ Step 2 done: camera position is good")
    else:
        log.warning(f"  ✗ Step 2: {len(issues)} issue(s) — {'; '.join(issues)}")
    return {
        "ok": ok,
        "issues": issues,
        "phone_region": [round(rect_w), round(rect_h)],
        "image_size": [img_w, img_h],
        "aspect_ratio": round(phone_ratio, 2),
        "coverage": round(coverage, 2),
    }


# ─── Step 3: Software rotation via UP/RIGHT markers ──────────


def pick_frame_rotation(cam: Camera, cal: CalibrationState) -> int:
    """Detect blue UP/RIGHT markers, determine software rotation. Plan Step 3.

    Returns cv2 rotation code (cv2.ROTATE_* or -1 for no rotation).
    """
    log.info("═══ Step 3: Software rotation via UP/RIGHT markers ═══")
    log.info("  Goal: detect how camera image is rotated relative to phone orientation")
    cal.set_phase("markers")
    time.sleep(1.0)
    log.info(
        "  Phase: markers — phone shows blue UP label (top) and red RIGHT label (right)"
    )

    frame = cam._fresh_frame()
    if frame is None:
        raise RuntimeError("Step 3 FAILED — camera read failed")

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
            raise RuntimeError(f"Step 3 FAILED — {label} marker not found")
        return best[0], best[1]

    # UP = blue (#2563eb → HSV H≈110), RIGHT = red (#ef4444 → HSV H≈0/180)
    up_x, up_y = _find_blob([100, 80, 80], [130, 255, 255], "UP (blue)")
    right_x, right_y = _find_blob([0, 80, 80], [10, 255, 255], "RIGHT (red)")
    # Also check wrapped red hue (170-180)
    try:
        rx2, ry2 = _find_blob([170, 80, 80], [180, 255, 255], "RIGHT (red high)")
        # Pick whichever red blob is larger / was found
        right_x, right_y = rx2, ry2
    except RuntimeError:
        pass  # first range was enough

    log.info(f"  Blue UP marker: camera pixel ({up_x:.0f}, {up_y:.0f})")
    log.info(f"  Red RIGHT marker: camera pixel ({right_x:.0f}, {right_y:.0f})")

    # Determine rotation by relative position of UP vs RIGHT
    if up_y < right_y and abs(up_x - right_x) < abs(up_y - right_y):
        rotation = -1  # no rotation
        rot_label = "0° — no rotation needed"
    elif up_x < right_x and abs(up_y - right_y) < abs(up_x - right_x):
        rotation = cv2.ROTATE_90_CLOCKWISE
        rot_label = "90° clockwise"
    elif up_y > right_y and abs(up_x - right_x) < abs(up_y - right_y):
        rotation = cv2.ROTATE_180
        rot_label = "180°"
    else:
        rotation = cv2.ROTATE_90_COUNTERCLOCKWISE
        rot_label = "90° counter-clockwise"

    log.info(
        f"  UP is {'above' if up_y < right_y else 'below'} RIGHT, "
        f"{'left of' if up_x < right_x else 'right of'} RIGHT"
    )
    log.info(f"  ✓ Step 3 done: camera needs {rot_label} to match phone orientation")
    return rotation


# ─── Step 4: GRBL ↔ Screen mapping (Mapping A) ──────────────


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


def compute_grbl_mapping(
    arm: StylusArm, cal: CalibrationState, z_tap: float
) -> tuple[np.ndarray, list[dict]]:
    """Probe scale, tap grid points across the screen, compute affine. Plan Step 4.

    Phase 1: Tap center, +10mm X, +10mm Y to discover mm-per-screen-unit scale.
    Phase 2: Use scale + grid_cols/rows to tap 15 points spanning the full screen.
    Phase 3: Verify with 3 additional taps at offset positions.

    Touch x,y are 0-1 screen percentages.
    Returns (pct_to_grbl affine (2,3), list of touch events).
    """
    log.info("═══ Step 4: GRBL ↔ Screen mapping (Mapping A) ═══")
    log.info("  Goal: compute affine transform from screen 0-1 → arm GRBL mm")
    cal.set_phase("center")
    time.sleep(0.3)

    PROBE_D = 10.0  # mm offset for scale probes

    # ── Phase 1: Probe to find scale ──
    log.info("  Phase 1: Probing scale — 3 taps to discover mm-per-screen-unit")
    log.info("    Tap 1/3: arm center (0, 0)mm")
    t_center, z_tap = _tap_and_read(arm, cal, 0, 0, z_tap)
    if not t_center:
        raise RuntimeError("Step 4 FAILED — no touch at center")
    log.info(f"    Tap 1/3: screen pos ({t_center['x']:.3f}, {t_center['y']:.3f})")

    log.info(f"    Tap 2/3: arm +X ({PROBE_D:.0f}, 0)mm")
    t_x, z_tap = _tap_and_read(arm, cal, PROBE_D, 0, z_tap)
    if not t_x:
        raise RuntimeError("Step 4 FAILED — no touch at +X probe")
    log.info(f"    Tap 2/3: screen pos ({t_x['x']:.3f}, {t_x['y']:.3f})")

    log.info(f"    Tap 3/3: arm +Y (0, {PROBE_D:.0f})mm")
    t_y, z_tap = _tap_and_read(arm, cal, 0, PROBE_D, z_tap)
    if not t_y:
        raise RuntimeError("Step 4 FAILED — no touch at +Y probe")
    log.info(f"    Tap 3/3: screen pos ({t_y['x']:.3f}, {t_y['y']:.3f})")

    # Screen displacement per mm in each arm direction
    sx_per_mm_x = (t_x["x"] - t_center["x"]) / PROBE_D
    sy_per_mm_x = (t_x["y"] - t_center["y"]) / PROBE_D
    sx_per_mm_y = (t_y["x"] - t_center["x"]) / PROBE_D
    sy_per_mm_y = (t_y["y"] - t_center["y"]) / PROBE_D
    log.info(
        f"  Phase 1 result: arm +{PROBE_D:.0f}mm X → "
        f"screen Δ({sx_per_mm_x:.4f}, {sy_per_mm_x:.4f}) per mm"
    )
    log.info(
        f"  Phase 1 result: arm +{PROBE_D:.0f}mm Y → "
        f"screen Δ({sx_per_mm_y:.4f}, {sy_per_mm_y:.4f}) per mm"
    )

    # Build screen→grbl affine from the 3 probe points
    probe_screen = np.array(
        [
            [t_center["x"], t_center["y"]],
            [t_x["x"], t_x["y"]],
            [t_y["x"], t_y["y"]],
        ],
        dtype=np.float64,
    )
    probe_grbl = np.array(
        [
            [0, 0],
            [PROBE_D, 0],
            [0, PROBE_D],
        ],
        dtype=np.float64,
    )
    probe_affine, _ = cv2.estimateAffine2D(probe_screen, probe_grbl)

    # ── Phase 2: Tap grid points across full screen ──
    cols = cal.GRID_COLS_PCT  # [0.25, 0.50, 0.75]
    rows = cal.GRID_ROWS_PCT  # [0.20, 0.40, 0.50, 0.60, 0.80]
    log.info(
        f"  Phase 2: Tapping {len(cols)}×{len(rows)}={len(cols) * len(rows)} grid points "
        f"across full screen using probe affine to predict arm positions"
    )

    grbl_pts = []
    screen_pts = []
    all_touches = []

    # Include probe points
    for sp, gp in zip(probe_screen, probe_grbl):
        screen_pts.append(sp.tolist())
        grbl_pts.append(gp.tolist())

    grid_idx = 0
    grid_total = len(cols) * len(rows)
    for row in rows:
        for col in cols:
            grid_idx += 1
            # Convert viewport 0-1 to screenshot 0-1 for probe affine
            # (probe_affine maps screenshot 0-1 → GRBL mm)
            if cal.viewport_shift:
                scr_col, scr_row = cal.viewport_pct_to_screenshot_pct(col, row)
            else:
                scr_col, scr_row = col, row
            predicted = probe_affine @ np.array([scr_col, scr_row, 1.0])
            gx, gy = predicted[0], predicted[1]
            log.info(
                f"    Grid {grid_idx}/{grid_total}: "
                f"viewport ({col:.2f}, {row:.2f}) → "
                f"predicted arm ({gx:.1f}, {gy:.1f})mm"
            )
            touch, z_tap = _tap_and_read(arm, cal, gx, gy, z_tap)
            if not touch:
                log.warning(f"    Grid {grid_idx}/{grid_total}: NO TOUCH — skipped")
                continue
            log.info(
                f"    Grid {grid_idx}/{grid_total}: "
                f"touch registered at screen ({touch['x']:.3f}, {touch['y']:.3f})"
            )
            grbl_pts.append([gx, gy])
            screen_pts.append([touch["x"], touch["y"]])
            all_touches.append(touch)

    arm._fast_move(0, 0)
    arm.wait_idle()

    log.info(
        f"  Phase 2 result: {len(grbl_pts)} point pairs collected "
        f"(3 probes + {len(grbl_pts) - 3} grid hits)"
    )
    if len(grbl_pts) < 6:
        raise RuntimeError(f"Step 4 FAILED — only {len(grbl_pts)} valid taps (need ≥6)")

    # Compute final affine from all points
    screen_to_grbl, _ = cv2.estimateAffine2D(
        np.array(screen_pts, dtype=np.float64), np.array(grbl_pts, dtype=np.float64)
    )
    if screen_to_grbl is None:
        raise RuntimeError("Step 4 FAILED — affine computation failed")
    log.info(f"  Affine computed from {len(grbl_pts)} point pairs")

    # ── Phase 3: Verify with 3 taps at non-grid positions ──
    verify_pcts = [(0.20, 0.35), (0.80, 0.65), (0.50, 0.50)]
    log.info(
        f"  Phase 3: Verifying accuracy with {len(verify_pcts)} taps at non-grid positions"
    )
    max_error = 0
    for vi, (vsx, vsy) in enumerate(verify_pcts, 1):
        predicted = screen_to_grbl @ np.array([vsx, vsy, 1.0])
        vgx, vgy = predicted[0], predicted[1]
        log.info(
            f"    Verify {vi}/{len(verify_pcts)}: "
            f"screen ({vsx}, {vsy}) → predicted arm ({vgx:.1f}, {vgy:.1f})mm"
        )
        touch, z_tap = _tap_and_read(arm, cal, vgx, vgy, z_tap)
        if not touch:
            log.warning(f"    Verify {vi}/{len(verify_pcts)}: NO TOUCH — skipped")
            continue
        actual_grbl = screen_to_grbl @ np.array([touch["x"], touch["y"], 1.0])
        error = ((actual_grbl[0] - vgx) ** 2 + (actual_grbl[1] - vgy) ** 2) ** 0.5
        max_error = max(max_error, error)
        log.info(
            f"    Verify {vi}/{len(verify_pcts)}: "
            f"touch at screen ({touch['x']:.3f}, {touch['y']:.3f}), "
            f"position error={error:.2f}mm"
        )

    # ── Re-origin: move arm to screen center and set as new origin ──
    center_grbl = screen_to_grbl @ np.array([0.5, 0.5, 1.0])
    log.info(
        f"  Re-origin: screen center (0.5, 0.5) is at arm ({center_grbl[0]:.2f}, {center_grbl[1]:.2f})mm"
    )
    arm._fast_move(center_grbl[0], center_grbl[1])
    arm.wait_idle()
    arm.set_origin()
    log.info("  Re-origin: arm position reset to (0, 0) at screen center")

    # Update affine to reflect new origin: subtract center offset from translation
    screen_to_grbl[0, 2] -= center_grbl[0]
    screen_to_grbl[1, 2] -= center_grbl[1]

    log.info(
        f"  ✓ Step 4 done: Mapping A ready, "
        f"verification max error={max_error:.2f}mm, "
        f"origin set at screen center"
    )
    return screen_to_grbl, all_touches


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
            [
                list(cal.viewport_pct_to_screenshot_pct(x, y))
                for y in cal.GRID_ROWS_PCT
                for x in cal.GRID_COLS_PCT
            ],
            dtype=np.float64,
        )
    else:
        screen_pcts = np.array(
            [[x, y] for y in cal.GRID_ROWS_PCT for x in cal.GRID_COLS_PCT],
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
    num_tests: int = 3,
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
        park_gx, park_gy = pct_to_grbl @ np.array([0.5, -0.5, 1.0])
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

    arm._fast_move(0, 0)
    arm.wait_idle()
    passed_count = sum(1 for r in results if r["passed"])
    log.info(f"  ✓ Step 6 done: {passed_count}/{num_tests} tests passed")
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
    arm._fast_move(0, 0)
    arm.wait_idle()
    log.info("Tracing phone edge clockwise...")
    for x_pct, y_pct, label in check_points:
        gx, gy = cal.pct_to_grbl_mm(x_pct, y_pct)
        log.info(f"  → {label} ({x_pct}, {y_pct}) = GRBL ({gx:.2f}, {gy:.2f})")
        arm._fast_move(gx, gy)
        arm.wait_idle()
        time.sleep(2)

    arm._fast_move(0, 0)
    arm.wait_idle()
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

    1. Single-tap → iOS takes a screenshot (Photos).
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
