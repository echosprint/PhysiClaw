"""Image codec, similarity, and shape-analysis utilities."""

import json
import logging

import cv2
import numpy as np

log = logging.getLogger(__name__)

FRAME_SIMILARITY_SIZE = (320, 240)


def encode_jpeg(frame: np.ndarray, quality: int = 85) -> bytes:
    """Encode a BGR frame to JPEG bytes."""
    _, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return jpeg.tobytes()


def decode_image(data: bytes) -> np.ndarray:
    """Decode image bytes (PNG or JPEG) to a BGR frame. Raises on failure."""
    arr = np.frombuffer(data, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise RuntimeError("Failed to decode image bytes")
    return frame


def _hsv_blob_centroids(
    hsv: np.ndarray,
    lower,
    upper,
    *,
    min_area: int,
    morph_op: int,
    morph_kernel: tuple[int, int],
) -> list[tuple[float, float]]:
    """Core of the HSV-blob pipeline, reusable when the caller already
    has an HSV frame. Returns every centroid with area ≥ ``min_area``.
    """
    mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, morph_kernel)
    mask = cv2.morphologyEx(mask, morph_op, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out: list[tuple[float, float]] = []
    for cnt in contours:
        if cv2.contourArea(cnt) < min_area:
            continue
        m = cv2.moments(cnt)
        if m["m00"] == 0:
            continue
        out.append((m["m10"] / m["m00"], m["m01"] / m["m00"]))
    return out


def find_all_hsv_blobs(
    frame: np.ndarray,
    lower,
    upper,
    *,
    min_area: int = 50,
    morph_op: int = cv2.MORPH_OPEN,
    morph_kernel: tuple[int, int] = (5, 5),
) -> list[tuple[float, float]]:
    """Return centroids of every HSV-matched blob above ``min_area``.

    Same pipeline as :func:`find_largest_hsv_blob` but keeps every
    qualifying contour. Order is undefined; callers cluster by position.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    return _hsv_blob_centroids(
        hsv, lower, upper,
        min_area=min_area, morph_op=morph_op, morph_kernel=morph_kernel,
    )


def find_largest_hsv_blob(
    frame: np.ndarray,
    lower,
    upper,
    *,
    min_area: int = 50,
    morph_op: int = cv2.MORPH_OPEN,
    morph_kernel: tuple[int, int] = (5, 5),
) -> tuple[float, float] | None:
    """Centroid (cx, cy) of the largest HSV-matched blob, or None.

    Converts to HSV, thresholds by the given range, applies one
    morphology pass (``open`` by default to kill salt-and-pepper, or
    ``close`` to seal gaps), and returns the centroid of the biggest
    contour whose area is at least ``min_area``. Returns ``None`` when
    no blob passes the filter.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, morph_kernel)
    mask = cv2.morphologyEx(mask, morph_op, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < min_area:
        return None
    m = cv2.moments(largest)
    if m["m00"] == 0:
        return None
    return (m["m10"] / m["m00"], m["m01"] / m["m00"])


CORNER_HSV_RANGES = {
    "R": [([0, 100, 100], [10, 255, 255]),
          ([170, 100, 100], [180, 255, 255])],
    "G": [([40, 100, 100], [80, 255, 255])],
    "B": [([100, 100, 100], [130, 255, 255])],
    "Y": [([20, 100, 100], [35, 255, 255])],
}


def _is_clockwise_rgby_cluster(
    cluster: dict[str, tuple[float, float]], max_span: float
) -> bool:
    """Four RGBY centroids must be tightly grouped and traverse R→G→B→Y
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
    return "".join(name for name, _ in ordered) in "RGBYRGBY"


def detect_bridge_corners(
    frame: np.ndarray, max_cluster_span: float | None = None
) -> dict | None:
    """Find one intact RGBY cluster rendered by bridge.html's ``corners``
    phase. bridge.html draws the same 2×2 RGBY cluster at all four
    screen corners, so a stylus occluding up to three of them still
    leaves enough to identify the camera.

    Returns the four centroids of the first intact cluster, or ``None``
    if no combination of detected blobs forms a tight-enough group
    whose clockwise order is a cyclic rotation of RGBY.

    ``max_cluster_span`` defaults to 25% of the frame's shorter side —
    each on-phone cluster is ~12% of the phone's shorter side, and the
    phone always spans a big fraction of the frame during setup, so 25%
    is comfortable margin without admitting cross-cluster pairings.
    """
    if max_cluster_span is None:
        max_cluster_span = min(frame.shape[:2]) * 0.25

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    blobs: dict[str, list[tuple[float, float]]] = {k: [] for k in "RGBY"}
    for name, hsv_ranges in CORNER_HSV_RANGES.items():
        for lo, hi in hsv_ranges:
            blobs[name].extend(_hsv_blob_centroids(
                hsv, lo, hi,
                min_area=50, morph_op=cv2.MORPH_OPEN, morph_kernel=(5, 5),
            ))

    if not all(blobs[k] for k in "RGBY"):
        return None

    # Brute force — at most 4 clusters × 1 blob per color per cluster =
    # 4⁴ = 256 candidates. The span filter rejects cross-cluster pairings
    # almost immediately, so this stays cheap in practice.
    for r in blobs["R"]:
        for g in blobs["G"]:
            for b in blobs["B"]:
                for y in blobs["Y"]:
                    candidate = {"R": r, "G": g, "B": b, "Y": y}
                    if _is_clockwise_rgby_cluster(candidate, max_cluster_span):
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
    Saves an annotated frame to /tmp/physiclaw_camera_rotation.jpg.
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
        annotated, f"area {coverage:.0%}", (bx + 5, by + 30),
        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
    )
    cv2.imwrite("/tmp/physiclaw_camera_rotation.jpg", annotated)

    # Phone edges should be parallel to image edges (< 3° deviation).
    pts = cv2.boxPoints(rect)
    edges = [(pts[i], pts[(i + 1) % 4]) for i in range(4)]
    longest_edge = max(edges, key=lambda e: np.linalg.norm(e[1] - e[0]))
    angle_deg = abs(np.degrees(np.arctan2(
        longest_edge[1][1] - longest_edge[0][1],
        longest_edge[1][0] - longest_edge[0][0],
    )))
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

    # Coverage: phone should fill ≥ 30% of frame.
    if coverage < 0.30:
        issues.append(
            f"Move camera closer — phone covers only {coverage:.0%} of image (need ≥30%)"
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


def validate_bbox(bbox: list[float]):
    """Raise ValueError if bbox is malformed."""
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        raise ValueError(f"bbox must be [left, top, right, bottom], got {bbox!r}")
    left, top, right, bottom = bbox
    if not all(isinstance(v, (int, float)) for v in bbox):
        raise ValueError(f"bbox values must be numbers, got {bbox!r}")
    if not all(0 <= v <= 1 for v in bbox):
        raise ValueError(f"bbox values must be in [0, 1], got {bbox!r}")
    if left >= right or top >= bottom:
        raise ValueError(
            f"bbox must have left < right and top < bottom, got {bbox!r}"
        )


def bbox_on_screen(bbox: list[float]) -> bool:
    """True if bbox is a valid box fully within the phone screen."""
    try:
        validate_bbox(bbox)
        return True
    except ValueError:
        return False


# iPhone passcode numpad grid (row, col), 0-based
_NUMPAD_GRID = {
    "1": (0, 0), "2": (0, 1), "3": (0, 2),
    "4": (1, 0), "5": (1, 1), "6": (1, 2),
    "7": (2, 0), "8": (2, 1), "9": (2, 2),
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
        for kb in keys[i + 1:]:
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
    """Human/agent-friendly element list — one line per element, no JSON noise."""
    lines = ["id [kind] \"label\" [left,top,right,bottom] conf"]
    for e in items:
        bbox = ",".join(f"{v:.3f}" for v in e["bbox"])
        label = e.get("label") or ""
        lines.append(f'{e["id"]} [{e["kind"]}] "{label}" [{bbox}] {e["conf"]:.2f}')
    return "\n".join(lines)
