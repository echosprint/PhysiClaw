"""Pre-cal: screenshot coordinate mapping (viewport CSS → screenshot 0-1).

The phone page draws an orange square at a known viewport CSS position;
the user uploads a phone screenshot (double-tap AssistiveTouch) and the
server locates the square in it. Square size vs known CSS size yields
the device pixel ratio; expected vs detected center yields the status
bar / safe-area offset. Everything downstream (arm grid, camera grid,
validation) converts viewport coordinates through this ViewportShift,
so this step must run before arm calibration.

The measured screenshot is cached on disk at ``VIEWPORT_CACHE_STEM`` so
non-interactive reruns skip the manual upload; interactive setup passes
``fresh=True`` to force a real measurement.
"""

import logging
import time
from pathlib import Path

import cv2
import numpy as np

from physiclaw.common import paths
from physiclaw.core.bridge import BridgeState, CalibrationState
from physiclaw.core.geometry import (
    SQUARE_CSS_SIZE,
    SQUARE_CSS_X,
    SQUARE_CSS_Y,
    ViewportShift,
)
from physiclaw.core.vision.colors import ORANGE_HSV_RANGE, hsv_mask
from physiclaw.core.vision.preprocess import to_hsv

log = logging.getLogger(__name__)

VIEWPORT_CACHE_STEM = paths.calibration_cache_dir() / "viewport"

# The pre-cal square's CSS position is single-sourced in
# core/geometry/screen.py (imported at top): the page draws it exactly
# where this module's screenshot mapping expects to find it.


def _find_viewport_cache() -> Path | None:
    """Return the first existing cached viewport screenshot, or None."""
    for ext in ("png", "jpg"):
        p = VIEWPORT_CACHE_STEM.with_suffix(f".{ext}")
        if p.exists():
            return p
    return None


def _decode_screenshot_square(
    data: bytes,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Decode screenshot bytes and locate the orange pre-cal square.

    Returns (image, bounding rect). Raises RuntimeError when the data
    doesn't decode or shows no square — callers holding a maybe-stale
    upload catch that and fall back to waiting for a fresh one.
    """
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError("Failed to decode screenshot image")

    sh, sw = img.shape[:2]
    log.info(f"  Screenshot decoded: {sw}×{sh}px")

    # Detect orange square — ORANGE_HSV_RANGE is the same table
    # _detect_orange_dot matches against, so the two can't drift.
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
    return img, cv2.boundingRect(largest)


def measure_viewport_shift(
    cal: CalibrationState,
    bridge: BridgeState,
    *,
    fresh: bool = False,
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

    # Show orange square at viewport center. If the phone was already in
    # this phase (the wizard switches as soon as the step opens), the
    # phase-entry clock keeps ticking from that earlier switch.
    cal.set_phase("screenshot_cal")
    time.sleep(0.5)
    log.info(
        "  Phase: screenshot_cal — showing orange square at CSS "
        f"({SQUARE_CSS_X}, {SQUARE_CSS_Y})"
    )

    cached = None if fresh else _find_viewport_cache()
    if cached is not None:
        data = cached.read_bytes()
        log.info(f"  Using cached screenshot: {cached} ({len(data)} bytes)")
        img, rect = _decode_screenshot_square(data)
    else:
        # Best case: the user already double-tapped while the square was on
        # screen (the wizard says to upload before pressing Measure) — use
        # that shot straight away. The phase-entry cutoff excludes uploads
        # that predate the square; a shot that decodes but shows no square
        # (tapped mid-render) falls through to a fresh wait.
        img = rect = None
        data = bridge.take_pending_screenshot(received_after=cal.phase_since)
        if data is not None:
            try:
                img, rect = _decode_screenshot_square(data)
                log.info(f"  Using already-uploaded screenshot: {len(data)} bytes")
            except RuntimeError as e:
                log.info(f"  Pending upload unusable ({e}) — waiting for a fresh one")
        if img is None:
            bridge.clear_screenshot()
            log.info("  Waiting for phone screenshot (double-tap AssistiveTouch)...")
            data = bridge.wait_screenshot(timeout=30.0)
            if data is None:
                raise RuntimeError(
                    "Timeout — no screenshot received. Double-tap AssistiveTouch to upload."
                )
            log.info(f"  Screenshot received: {len(data)} bytes")
            img, rect = _decode_screenshot_square(data)

    sh, sw = img.shape[:2]
    x, y, w, h = rect
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
