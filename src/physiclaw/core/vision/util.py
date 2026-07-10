"""Image codec, similarity, and shape-analysis utilities."""

import json
import logging
import tempfile
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger(__name__)

_ROTATION_DEBUG_PATH = str(Path(tempfile.gettempdir()) / "physiclaw_camera_rotation.jpg")

FRAME_SIMILARITY_SIZE = (320, 240)


def encode_jpeg(frame: np.ndarray, quality: int = 85) -> bytes:
    """Encode a BGR frame to JPEG bytes."""
    _, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return jpeg.tobytes()


def phone_screen_crop_box(
    frame: np.ndarray, transforms
) -> tuple[int, int, int, int] | None:
    """Camera pixel rectangle (left, top, right, bottom) enclosing the phone screen.

    Returns None if calibration is missing or the rectangle degenerates
    after clamping to frame bounds.
    """
    if transforms is None:
        return None
    (x0, y0), (x1, y1) = transforms.bbox_to_pixel_rect([0.0, 0.0, 1.0, 1.0])
    h, w = frame.shape[:2]
    left, right = max(0, min(x0, x1)), min(w, max(x0, x1))
    top, bottom = max(0, min(y0, y1)), min(h, max(y0, y1))
    if right <= left or bottom <= top:
        return None
    return (left, top, right, bottom)


def crop_to_phone_screen(
    frame: np.ndarray, transforms, max_long_edge: int = 1024
) -> np.ndarray:
    """Crop to the phone-screen region and downscale to cap vision tokens.

    Returns the frame untouched if calibration is missing.
    """
    box = phone_screen_crop_box(frame, transforms)
    if box is None:
        return frame
    left, top, right, bottom = box
    cropped = frame[top:bottom, left:right]
    long_edge = max(cropped.shape[:2])
    if long_edge > max_long_edge:
        scale = max_long_edge / long_edge
        new_size = (int(cropped.shape[1] * scale), int(cropped.shape[0] * scale))
        cropped = cv2.resize(cropped, new_size, interpolation=cv2.INTER_AREA)
    return cropped


# Laplacian variance is not scale-invariant: the same screen captured at
# a higher resolution scores differently, so sharpness scores (and the
# thresholds tuned against them) only transfer between rigs when measured
# at one working width. Frames wider than this are downscaled before
# scoring; narrower ones are scored as-is (upscaling would interpolation-
# smooth genuinely sharp pixels into false blur). 480 ≈ the crop width of
# the corpora the thresholds were calibrated on.
NORMALIZED_WIDTH = 480


def laplacian_variance(frame: np.ndarray) -> float:
    """Variance of Laplacian — a focus/blur estimate. Higher = sharper.

    Accepts a BGR or already-gray frame; frames wider than
    NORMALIZED_WIDTH are downscaled first so scores are comparable
    across cameras and resolutions. Sharp phone screenshots with
    text/icons typically score 300+; severe motion blur or out-of-focus
    drops it under 80. Run on the cropped phone-screen region —
    backgrounds (cutting mat, ruler) contain their own edges that would
    mask real blur on the screen.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    h, w = gray.shape[:2]
    if w > NORMALIZED_WIDTH:
        gray = cv2.resize(
            gray,
            (NORMALIZED_WIDTH, round(h * NORMALIZED_WIDTH / w)),
            interpolation=cv2.INTER_AREA,
        )
    return float(cv2.Laplacian(gray, cv2.CV_16S).var())


def decode_image(data: bytes) -> np.ndarray:
    """Decode image bytes (PNG or JPEG) to a BGR frame. Raises on failure."""
    arr = np.frombuffer(data, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise RuntimeError("Failed to decode image bytes")
    return frame


def hsv_mask(hsv: np.ndarray, ranges) -> np.ndarray:
    """OR of ``cv2.inRange`` over one or more ``(lower, upper)`` HSV ranges.

    Lets one call cover a hue that wraps the 0/180 seam — red is passed both
    ``[0..10]`` and ``[170..180]``. The leaf primitive every colour detector
    builds on (blob centroids here, the dock badge's pixel count in the
    watchdog), so "red needs two ranges" lives in exactly one place.
    """
    mask = None
    for lo, hi in ranges:
        m = cv2.inRange(hsv, np.array(lo), np.array(hi))
        mask = m if mask is None else (mask | m)
    return mask


def redness(frame: np.ndarray) -> np.ndarray:
    """Per-pixel "how red" map: ``R - max(G, B)``, clipped to 0–255 (uint8).

    Robust where an HSV saturation floor isn't: small red marks on a bright
    screen desaturate to pink under a camera (low S, dim V), but their red
    channel still sits clearly above green/blue. Redness isolates them when
    ``red_ranges`` + an S/V threshold would wipe them out. Use this for faint
    targets (calibration dots); use :func:`red_ranges` for bold solid red
    (orientation markers, dock badges, corner squares).
    """
    bgr = frame.astype(np.int16)
    r = bgr[:, :, 2] - np.maximum(bgr[:, :, 0], bgr[:, :, 1])
    return np.clip(r, 0, 255).astype(np.uint8)


def red_ranges(s_min: int = 100, v_min: int = 100):
    """The two HSV ranges covering red across the 0/180 hue seam.

    Callers pick the S/V floor that suits them: the orientation marker and the
    camera-pick corner blocks pass 80 (both wash out under a camera on a bright
    screen), while the dock badge keeps the default 100. Hue bounds are fixed.
    Faint targets like the calibration dots use :func:`redness` instead.
    """
    return [
        ([0, s_min, v_min], [10, 255, 255]),
        ([170, s_min, v_min], [180, 255, 255]),
    ]


def _as_ranges(lower, upper):
    """Normalise a colour spec to a list of ``(lower, upper)`` pairs:
    ``(lower, upper)`` → one range; ``upper=None`` → ``lower`` is already a
    list of ranges (e.g. ``red_ranges()``)."""
    return [(lower, upper)] if upper is not None else list(lower)


def contour_centroid(cnt) -> tuple[float, float] | None:
    """Area-weighted centroid ``(cx, cy)`` of a contour, or ``None`` if it's
    degenerate (zero area)."""
    m = cv2.moments(cnt)
    if m["m00"] == 0:
        return None
    return (m["m10"] / m["m00"], m["m01"] / m["m00"])


def _hsv_blob_centroids(
    hsv: np.ndarray,
    ranges,
    *,
    min_area: int,
    morph_op: int,
    morph_kernel: tuple[int, int],
) -> list[tuple[float, float]]:
    """Core of the HSV-blob pipeline, reusable when the caller already
    has an HSV frame. Returns every centroid with area ≥ ``min_area``.
    """
    mask = hsv_mask(hsv, ranges)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, morph_kernel)
    mask = cv2.morphologyEx(mask, morph_op, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out: list[tuple[float, float]] = []
    for cnt in contours:
        if cv2.contourArea(cnt) < min_area:
            continue
        c = contour_centroid(cnt)
        if c is not None:
            out.append(c)
    return out


def find_all_hsv_blobs(
    frame: np.ndarray,
    lower,
    upper=None,
    *,
    min_area: int = 50,
    morph_op: int = cv2.MORPH_OPEN,
    morph_kernel: tuple[int, int] = (5, 5),
) -> list[tuple[float, float]]:
    """Return centroids of every HSV-matched blob above ``min_area``.

    One range as ``lower``/``upper``, or a list of ranges as ``lower``
    (``upper`` omitted) for a wrapping hue — see :func:`red_ranges`. Same
    pipeline as :func:`find_largest_hsv_blob` but keeps every qualifying
    contour; order is undefined, so callers cluster by position.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    return _hsv_blob_centroids(
        hsv,
        _as_ranges(lower, upper),
        min_area=min_area,
        morph_op=morph_op,
        morph_kernel=morph_kernel,
    )


def find_largest_hsv_blob(
    frame: np.ndarray,
    lower,
    upper=None,
    *,
    min_area: int = 50,
    morph_op: int = cv2.MORPH_OPEN,
    morph_kernel: tuple[int, int] = (5, 5),
) -> tuple[float, float] | None:
    """Centroid (cx, cy) of the largest HSV-matched blob, or None.

    One range as ``lower``/``upper``, or a list of ranges as ``lower``
    (``upper`` omitted) to cover a hue that wraps the 0/180 seam — see
    :func:`red_ranges`. Applies one morphology pass (``open`` kills
    salt-and-pepper, ``close`` seals gaps) and returns the biggest contour's
    centroid, or ``None`` when none reaches ``min_area``.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = hsv_mask(hsv, _as_ranges(lower, upper))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, morph_kernel)
    mask = cv2.morphologyEx(mask, morph_op, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < min_area:
        return None
    return contour_centroid(largest)


# S/V floor for the corner blocks. On a dim rig the captured blocks sit at
# ~S/V 100-120, so the old floor of 100 was right at the edge — a slightly
# dimmer setup would miss them. 80 adds margin. Don't drop below ~60: there
# the search starts matching colourful home-screen app icons and the cluster
# check mis-reads them as a corner cluster.
_CORNER_SV_MIN = 80
# Magenta, not yellow, is the 4th corner colour: a camera renders the screen's
# yellow as a yellow-green (H≈44) that drifts out of the Yellow band and into
# Green, so the yellow blocks vanish and the all-four-colours check fails.
# Magenta (#ff00ff = red+blue, no green) sits alone in the otherwise-empty
# 130–170 hue channel — far from R/G/B and from anything in a typical workshop.
CORNER_HSV_RANGES = {
    "R": red_ranges(_CORNER_SV_MIN, _CORNER_SV_MIN),
    "G": [([40, _CORNER_SV_MIN, _CORNER_SV_MIN], [80, 255, 255])],
    "B": [([100, _CORNER_SV_MIN, _CORNER_SV_MIN], [130, 255, 255])],
    "M": [([134, _CORNER_SV_MIN, _CORNER_SV_MIN], [169, 255, 255])],
}


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

    ``max_cluster_span`` defaults to 25% of the frame's shorter side —
    each on-phone cluster is ~20% of the phone's shorter side (`bs` =
    10% per quadrant in bridge.html's ``corners`` case), and the phone
    fills 60–80% of the frame during setup, so 25% leaves comfortable
    margin without admitting cross-cluster pairings.
    """
    if max_cluster_span is None:
        max_cluster_span = min(frame.shape[:2]) * 0.25

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    blobs: dict[str, list[tuple[float, float]]] = {k: [] for k in "RGBM"}
    for name, hsv_ranges in CORNER_HSV_RANGES.items():
        blobs[name] = _hsv_blob_centroids(
            hsv, hsv_ranges, min_area=50,
            morph_op=cv2.MORPH_OPEN, morph_kernel=(5, 5),
        )

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


def frame_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Normalized cross-correlation of two frames in [-1, 1].

    Downsample to a common grayscale size and let cv2.matchTemplate
    compute Pearson's r. ~1 means same scene, ~0 uncorrelated.
    """
    ga = cv2.resize(cv2.cvtColor(a, cv2.COLOR_BGR2GRAY), FRAME_SIMILARITY_SIZE)
    gb = cv2.resize(cv2.cvtColor(b, cv2.COLOR_BGR2GRAY), FRAME_SIMILARITY_SIZE)
    return float(cv2.matchTemplate(ga, gb, cv2.TM_CCOEFF_NORMED)[0, 0])


def check_phone_in_frame(frame: np.ndarray) -> dict:
    """Shape/coverage/straightness diagnostic from one overhead frame.

    Returns ``{ok, issues, coverage, aspect_ratio, image_size, phone_region}``.
    Saves an annotated frame to ``<tempdir>/physiclaw_camera_rotation.jpg``.
    Raises if no bright region is detected (camera read failed or phone off).
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise RuntimeError("No bright region in camera frame — is the phone on?")

    largest = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(largest)
    rect_w, rect_h = rect[1]
    phone_area_px = rect_w * rect_h
    img_h, img_w = frame.shape[:2]
    coverage = phone_area_px / (img_w * img_h)
    bx, by, bw, bh = cv2.boundingRect(largest)
    issues: list[str] = []

    annotated = frame.copy()
    cv2.drawContours(annotated, [largest], -1, (0, 255, 0), 3)
    cv2.drawContours(annotated, [np.int32(cv2.boxPoints(rect))], -1, (0, 200, 255), 2)
    cv2.putText(
        annotated,
        f"area {coverage:.0%}",
        (bx + 5, by + 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2,
    )
    cv2.imwrite(_ROTATION_DEBUG_PATH, annotated)

    # Phone edges should be parallel to image edges (< 3° deviation).
    pts = cv2.boxPoints(rect)
    edges = [(pts[i], pts[(i + 1) % 4]) for i in range(4)]
    longest_edge = max(edges, key=lambda e: np.linalg.norm(e[1] - e[0]))
    angle_deg = abs(
        np.degrees(
            np.arctan2(
                longest_edge[1][1] - longest_edge[0][1],
                longest_edge[1][0] - longest_edge[0][0],
            )
        )
    )
    rotation_dev = min(angle_deg % 90, 90 - angle_deg % 90)
    if rotation_dev >= 3.0:
        issues.append(
            f"Straighten camera — phone edges rotated {rotation_dev:.1f}° from image"
        )

    # Long axes aligned (phone long axis parallel to image long axis).
    if (bw > bh) != (img_w > img_h):
        issues.append("Rotate camera 90° — long axes not aligned")

    # Aspect ratio sanity check (camera tilt).
    phone_long = max(rect_w, rect_h)
    phone_short = min(rect_w, rect_h)
    phone_ratio = phone_long / max(phone_short, 1)
    ratio_diff = abs(phone_ratio - 2.0) / 2.0
    if ratio_diff >= 0.15:
        issues.append(
            f"Camera may be tilted — phone aspect {phone_ratio:.2f} (diff {ratio_diff:.0%})"
        )

    # Coverage: phone should fill ≥ 25% of frame.
    if coverage < 0.25:
        issues.append(
            f"Move camera closer — phone covers only {coverage:.0%} of image (need ≥25%)"
        )

    log.info(
        f"  Phone in frame: {rect_w:.0f}×{rect_h:.0f}px, "
        f"edge dev {rotation_dev:.1f}°, aspect {phone_ratio:.2f}, coverage {coverage:.0%}"
    )
    if issues:
        log.warning(f"  Camera setup issues: {'; '.join(issues)}")

    return {
        "ok": not issues,
        "issues": issues,
        "phone_region": [round(rect_w), round(rect_h)],
        "image_size": [img_w, img_h],
        "aspect_ratio": round(phone_ratio, 2),
        "coverage": round(coverage, 2),
    }


def validate_bbox(bbox: list[float]) -> list[float]:
    """Raise ValueError if bbox is malformed; return `bbox` unchanged.

    Defense-in-depth runtime check before any GRBL move. IDENTICAL
    LOGIC to the engine validator's bbox check (in
    `src/physiclaw/agent/engine/validator.py`) — same checks, same
    order, same messages — so the agent sees the same diagnostic
    regardless of which layer catches the violation. Keep the two in
    sync.
    """
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        raise ValueError(f"bbox: must be [left, top, right, bottom]; got {bbox!r}")
    if not all(isinstance(v, (int, float)) for v in bbox):
        raise ValueError(f"bbox: each coord must be a number; got {bbox!r}")
    left, top, right, bottom = bbox
    if any(v < 0 or v > 1 for v in bbox):
        raise ValueError(
            f"bbox: each coord must be in [0, 1]; got [{left}, {top}, {right}, {bottom}]"
        )
    if left >= right or top >= bottom:
        raise ValueError(
            f"bbox: left < right, top < bottom; got [{left}, {top}, {right}, {bottom}]"
        )
    return bbox


def bbox_on_screen(bbox: list[float]) -> bool:
    """True if bbox is a valid box fully within the phone screen."""
    try:
        validate_bbox(bbox)
        return True
    except ValueError:
        return False


# iPhone passcode numpad grid (row, col), 0-based
_NUMPAD_GRID = {
    "1": (0, 0),
    "2": (0, 1),
    "3": (0, 2),
    "4": (1, 0),
    "5": (1, 1),
    "6": (1, 2),
    "7": (2, 0),
    "8": (2, 1),
    "9": (2, 2),
    "0": (3, 1),
}


def _infer_numpad(key_a: str, pos_a: tuple, key_b: str, pos_b: tuple) -> dict:
    """Infer full numpad coordinates from two detected keys.

    Requires keys on different rows AND different columns.
    Returns {digit: (cx, cy)} for all 10 digits.
    """
    r_a, c_a = _NUMPAD_GRID[key_a]
    r_b, c_b = _NUMPAD_GRID[key_b]
    col_step = (pos_a[0] - pos_b[0]) / (c_a - c_b)
    row_step = (pos_a[1] - pos_b[1]) / (r_a - r_b)
    x_origin = pos_a[0] - c_a * col_step
    y_origin = pos_a[1] - r_a * row_step
    return {
        key: (x_origin + c * col_step, y_origin + r * row_step)
        for key, (r, c) in _NUMPAD_GRID.items()
    }


def find_numpad_digit(elements: list[dict], digit: str) -> list[float] | None:
    """Find a passcode digit bbox from OCR elements. Falls back to grid inference.

    1. Direct match: look for an element whose label is exactly the digit.
    2. Inference: if not found, use any two detected digits on different
       rows and columns to infer the full numpad layout.

    Returns [left, top, right, bottom] as 0-1 decimals, or None.
    """
    # Collect single-digit elements in the keypad area (y ∈ [0.2, 0.8])
    detected: dict[str, dict] = {}
    for e in elements:
        label = e["label"].strip()
        _, y1, _, y2 = e["bbox"]
        if len(label) == 1 and label.isdigit() and 0.2 <= y1 and y2 <= 0.8:
            detected[label] = e

    # Direct match
    if digit in detected:
        return detected[digit]["bbox"]

    # Infer from any two digits on different rows and columns
    keys = list(detected.keys())
    for i, ka in enumerate(keys):
        ra, ca = _NUMPAD_GRID[ka]
        for kb in keys[i + 1 :]:
            rb, cb = _NUMPAD_GRID[kb]
            if ra == rb or ca == cb:
                continue
            ba, bb = detected[ka]["bbox"], detected[kb]["bbox"]
            cx_a, cy_a = (ba[0] + ba[2]) / 2, (ba[1] + ba[3]) / 2
            cx_b, cy_b = (bb[0] + bb[2]) / 2, (bb[1] + bb[3]) / 2
            cx, cy = _infer_numpad(ka, (cx_a, cy_a), kb, (cx_b, cy_b))[digit]
            hw = (ba[2] - ba[0]) / 2
            hh = (ba[3] - ba[1]) / 2
            return [cx - hw, cy - hh, cx + hw, cy + hh]

    return None


def compact_json(items: list[dict]) -> str:
    """Pretty-print a list of dicts with one item per line (for file output)."""
    lines = [json.dumps(item, ensure_ascii=False) for item in items]
    return "[\n" + ",\n".join(f"  {line}" for line in lines) + "\n]\n"


def format_elements(items: list[dict]) -> str:
    """Human/agent-friendly element list — one line per element, no JSON noise.

    The header line is also documented for the agent in
    ``src/physiclaw/agent/context/PHYSICLAW.md`` — keep the two in sync.
    """
    lines = ['id [kind] "label" [left,top,right,bottom] conf']
    for e in items:
        bbox = ",".join(f"{v:.3f}" for v in e["bbox"])
        label = e.get("label") or ""
        lines.append(f'{e["id"]} [{e["kind"]}] "{label}" [{bbox}] {e["conf"]:.2f}')
    return "\n".join(lines)
