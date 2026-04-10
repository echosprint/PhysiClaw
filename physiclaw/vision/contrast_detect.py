"""Local-contrast button detector.

Finds small, high-contrast UI elements (FABs, quantity steppers, add-to-cart
buttons) that general-purpose icon detectors miss. Color-agnostic: works on
filled FABs, outlined buttons, and icon-bearing controls alike.

Core principle
──────────────
Buttons are designed to stand out, which on a pixel level means their local
region has much higher visual variation than the surrounding background. We
quantify this as the local standard deviation of pixel intensities in a small
sliding window. Regions where this stddev is high are candidate buttons,
regardless of their color or exact shape.

Pipeline
────────
    crop to ROI → grayscale → local stddev → threshold → morphology
    → findContours → size/aspect/contrast filters → NMS dedupe
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

import cv2
import numpy as np

log = logging.getLogger(__name__)


# ── Tunable parameters ─────────────────────────

# Region of interest
DEFAULT_X_MIN = 0.65             # scan x >= 65% of width (right side)
DEFAULT_X_MAX_LEFT = 0.15        # scan x <= 15% of width (left side)

# Size filters (as fraction of screen width)
DEFAULT_MIN_SIZE_RATIO = 0.025   # button width/height ≥ 2.5% of screen width
DEFAULT_MAX_SIZE_RATIO = 0.12    # button width/height ≤ 12% of screen width

# Contrast / morphology
DEFAULT_STDDEV_WINDOW = 9        # sliding window for local stddev
DEFAULT_MIN_CONTRAST = 40        # stddev threshold to binarize
DEFAULT_MORPH_KERNEL = 7         # ellipse kernel for CLOSE/OPEN

# Shape filters
DEFAULT_ASPECT_MIN = 0.6         # w/h lower bound (squarish)
DEFAULT_ASPECT_MAX = 1.8         # w/h upper bound

# NMS
DEFAULT_IOU_THRESHOLD = 0.4

# Confidence normalization — stddev of 80 is "very high contrast"
_CONF_NORMALIZER = 80.0


@dataclass
class ContrastDetection:
    """A button candidate found by local-contrast detection."""
    bbox: list[float]    # [x1, y1, x2, y2] normalized to 0-1
    conf: float          # derived from bbox interior contrast strength


# ── Core signal: local standard deviation ──────


def _local_stddev(gray: np.ndarray, window: int) -> np.ndarray:
    """Compute per-pixel standard deviation within a sliding window.

    Uses the shift form of variance:  Var(X) = E[X²] - (E[X])²

    This turns an otherwise O(H·W·window²) computation into two O(H·W) box
    filters, because cv2.blur uses integral images and runs in time
    independent of window size. ~40× faster than naive sliding window on
    typical phone screenshots.

    High output values mark visually "busy" regions — the signature of
    buttons against flat backgrounds, regardless of color.
    """
    f = gray.astype(np.float32)
    mean = cv2.blur(f, (window, window))
    sq_mean = cv2.blur(f * f, (window, window))
    variance = np.maximum(sq_mean - mean * mean, 0)
    return np.sqrt(variance).astype(np.uint8)


# ── Morphological cleanup ──────────────────────


def _morph_cleanup(mask: np.ndarray, kernel_size: int) -> np.ndarray:
    """Close small gaps inside button interiors, then remove speckle noise.

    Order matters:

    1. CLOSE first — fills the hollow interior of outlined buttons and the
       center of filled buttons (where stddev drops back to zero). This makes
       the whole button a single connected blob instead of a ring.
    2. OPEN second — removes isolated bright pixels from text antialiasing,
       JPEG artifacts, and texture noise. Real buttons, now solid blocks after
       CLOSE, survive the erosion step easily.

    Reversing the order would erode the thin edges of hollow buttons before
    they could be filled, destroying the very buttons we want to detect.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    return mask


# ── Shape filters ──────────────────────────────


def _passes_shape_filters(
    w: int,
    h: int,
    min_px: int,
    max_px: int,
    aspect_min: float,
    aspect_max: float,
) -> bool:
    """Accept only button-sized, roughly-square candidates."""
    if w < min_px or h < min_px:
        return False
    if w > max_px or h > max_px:
        return False
    if h == 0:
        return False
    aspect = w / h
    return aspect_min <= aspect <= aspect_max


# ── IoU and NMS ────────────────────────────────


def _iou(a: list[float], b: list[float]) -> float:
    """Intersection over Union for two normalized bboxes."""
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _nms(
    detections: list[ContrastDetection],
    iou_threshold: float,
) -> list[ContrastDetection]:
    """Non-Maximum Suppression: keep the highest-confidence detection in each
    overlapping cluster."""
    ordered = sorted(detections, key=lambda d: d.conf, reverse=True)
    kept: list[ContrastDetection] = []
    for det in ordered:
        if any(_iou(det.bbox, k.bbox) > iou_threshold for k in kept):
            continue
        kept.append(det)
    return kept


# ── Main entry point ───────────────────────────


def detect_contrast_buttons(
    frame: np.ndarray,
    x_min: float = DEFAULT_X_MIN,
    x_max_left: float = DEFAULT_X_MAX_LEFT,
    min_size_ratio: float = DEFAULT_MIN_SIZE_RATIO,
    max_size_ratio: float = DEFAULT_MAX_SIZE_RATIO,
    min_contrast: int = DEFAULT_MIN_CONTRAST,
    stddev_window: int = DEFAULT_STDDEV_WINDOW,
    morph_kernel: int = DEFAULT_MORPH_KERNEL,
    aspect_min: float = DEFAULT_ASPECT_MIN,
    aspect_max: float = DEFAULT_ASPECT_MAX,
    iou_threshold: float = DEFAULT_IOU_THRESHOLD,
    debug_dir: str | None = None,
) -> list[ContrastDetection]:
    """Detect small high-contrast buttons on the left and right edges of a screen.

    Scans two ROIs: x < x_max_left (left nav rail) and x >= x_min (right side).

    Returns normalized bboxes (0-1 coordinates) with confidence scores derived
    from the bbox's interior contrast strength. Designed as a fallback to a
    general-purpose icon detector, targeting the small FABs and quantity
    steppers that trained models frequently miss.

    Parameters
    ──────────
    frame : BGR image (np.ndarray, HxWx3, uint8)
    x_min : left edge of the right scan region, as a fraction of screen width
    x_max_left : right edge of the left scan region, as a fraction of screen width
    min_size_ratio, max_size_ratio : size bounds as fractions of screen width
    min_contrast : minimum local stddev to consider a pixel "busy"
    stddev_window : sliding-window size for local stddev (odd, typically 7-11)
    morph_kernel : ellipse kernel size for CLOSE/OPEN (odd, typically 5-9)
    aspect_min, aspect_max : allowed width/height ratio
    iou_threshold : IoU above which detections are considered duplicates
    debug_dir : if given, write intermediate pipeline images to this directory

    Returns
    ───────
    List of ContrastDetection with normalized bboxes and confidence ∈ [0, 1]
    """
    H, W = frame.shape[:2]
    if H == 0 or W == 0:
        return []

    kwargs = dict(
        min_size_ratio=min_size_ratio, max_size_ratio=max_size_ratio,
        min_contrast=min_contrast, stddev_window=stddev_window,
        morph_kernel=morph_kernel, aspect_min=aspect_min,
        aspect_max=aspect_max,
    )

    # Scan both edges
    right_dets = _scan_roi(frame, int(W * x_min), W, H, W, **kwargs)
    left_dets = _scan_roi(frame, 0, int(W * x_max_left), H, W, **kwargs)
    detections = right_dets + left_dets

    final = _nms(detections, iou_threshold)

    log.debug(
        "contrast detector: %d right + %d left → %d after NMS",
        len(right_dets), len(left_dets), len(final),
    )

    if debug_dir:
        # Debug uses right ROI only for backward compat
        x_start = int(W * x_min)
        roi = frame[:, x_start:]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        stddev_map = _local_stddev(gray, stddev_window)
        _, mask = cv2.threshold(stddev_map, min_contrast, 255, cv2.THRESH_BINARY)
        mask = _morph_cleanup(mask, morph_kernel)
        _dump_debug(debug_dir, roi, stddev_map, mask, final, x_start, W, H)

    return final


def _scan_roi(
    frame: np.ndarray,
    x_start: int,
    x_end: int,
    H: int,
    W: int,
    *,
    min_size_ratio: float,
    max_size_ratio: float,
    min_contrast: int,
    stddev_window: int,
    morph_kernel: int,
    aspect_min: float,
    aspect_max: float,
) -> list[ContrastDetection]:
    """Run the contrast pipeline on a single horizontal ROI strip."""
    roi = frame[:, x_start:x_end]
    if roi.size == 0:
        return []

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    stddev_map = _local_stddev(gray, stddev_window)
    _, mask = cv2.threshold(stddev_map, min_contrast, 255, cv2.THRESH_BINARY)
    mask = _morph_cleanup(mask, morph_kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    min_px = int(W * min_size_ratio)
    max_px = int(W * max_size_ratio)
    detections: list[ContrastDetection] = []

    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if not _passes_shape_filters(w, h, min_px, max_px, aspect_min, aspect_max):
            continue

        # Interior contrast check
        interior = gray[y:y + h, x:x + w]
        if interior.size == 0:
            continue
        global_contrast = float(interior.std())
        if global_contrast < min_contrast:
            continue

        # Surrounding-area calmness check — real buttons sit on flat backgrounds
        pad = 3
        rh, rw = gray.shape[:2]
        sy1, sy2 = max(0, y - pad), min(rh, y + h + pad)
        sx1, sx2 = max(0, x - pad), min(rw, x + w + pad)
        border_pixels = np.concatenate([
            gray[sy1:y, sx1:sx2].ravel(),
            gray[y + h:sy2, sx1:sx2].ravel(),
            gray[y:y + h, sx1:x].ravel(),
            gray[y:y + h, x + w:sx2].ravel(),
        ])
        if border_pixels.size > 0 and float(border_pixels.std()) > 10:
            continue

        # Map ROI-local coords to full-frame normalized coords
        conf = round(min(1.0, global_contrast / _CONF_NORMALIZER), 2)
        detections.append(ContrastDetection(
            bbox=[(x + x_start) / W, y / H, (x + w + x_start) / W, (y + h) / H],
            conf=conf,
        ))

    return detections


# ── Debug visualization ────────────────────────


def _dump_debug(
    out_dir: str,
    roi: np.ndarray,
    stddev_map: np.ndarray,
    mask: np.ndarray,
    detections: list[ContrastDetection],
    x_start: int,
    W: int,
    H: int,
) -> None:
    """Write pipeline intermediates for visual tuning."""
    os.makedirs(out_dir, exist_ok=True)
    cv2.imwrite(f"{out_dir}/01_roi.jpg", roi)
    cv2.imwrite(f"{out_dir}/02_stddev.jpg", stddev_map)
    cv2.imwrite(f"{out_dir}/03_mask.jpg", mask)

    vis = roi.copy()
    for d in detections:
        # Convert full-frame normalized coords back to ROI pixel coords
        x1 = int(d.bbox[0] * W) - x_start
        y1 = int(d.bbox[1] * H)
        x2 = int(d.bbox[2] * W) - x_start
        y2 = int(d.bbox[3] * H)
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 255), 2)
        cv2.putText(
            vis, f"{d.conf:.2f}",
            (x1, max(y1 - 4, 12)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1,
        )
    cv2.imwrite(f"{out_dir}/04_detected.jpg", vis)


# ── CLI ────────────────────────────────────────


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(
        description="Detect high-contrast buttons (FABs, steppers) missed by "
                    "general icon detectors."
    )
    parser.add_argument("image", help="Path to phone screen image")
    parser.add_argument("--x-min", type=float, default=DEFAULT_X_MIN,
                        help="Left edge of scan region (0-1)")
    parser.add_argument("--min-contrast", type=int, default=DEFAULT_MIN_CONTRAST,
                        help="Minimum local stddev threshold")
    parser.add_argument("--debug-dir", default=None,
                        help="Write pipeline intermediates here for tuning")
    parser.add_argument("--json", action="store_true",
                        help="Output detections as JSON")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG, format="%(message)s")

    frame = cv2.imread(args.image)
    if frame is None:
        raise SystemExit(f"cannot read image: {args.image}")

    detections = detect_contrast_buttons(
        frame,
        x_min=args.x_min,
        min_contrast=args.min_contrast,
        debug_dir=args.debug_dir,
    )

    if args.json:
        print(json.dumps(
            [{"bbox": [round(v, 3) for v in d.bbox], "conf": d.conf}
             for d in detections],
            indent=2, ensure_ascii=False,
        ))
    else:
        print(f"found {len(detections)} high-contrast buttons:")
        for i, d in enumerate(detections):
            print(f"  [{i}] bbox={[round(v, 3) for v in d.bbox]}  conf={d.conf}")
