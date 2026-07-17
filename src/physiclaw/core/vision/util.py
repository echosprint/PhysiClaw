"""Image codec, bbox-validation, and one-off diagnostic utilities.

Preprocessing stages (grayscale/HSV/blur/resize/crop) live in
`physiclaw.core.vision.preprocess`; color and blob primitives in
`colors` / `blobs`; the sharpness metric next to its thresholds in
`quality`.
"""

import json
import logging
import re
import tempfile
from pathlib import Path

import cv2
import numpy as np

from physiclaw.common.bbox import validate_bbox
from physiclaw.common.config import CONFIG
from physiclaw.core.vision.preprocess import grayscale, resize_to_max_edge

# `validate_bbox` comes from `physiclaw.common.bbox` (the shared
# core/agent contract) and is deliberately re-exported here:
# orchestration keeps importing it from this module, and
# `bbox_on_screen` below builds on it.

log = logging.getLogger(__name__)

_ROTATION_DEBUG_PATH = str(
    Path(tempfile.gettempdir()) / "physiclaw_camera_rotation.jpg"
)


def encode_jpeg(frame: np.ndarray, quality: int = 85) -> bytes:
    """Encode a BGR frame to JPEG bytes."""
    _, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return jpeg.tobytes()


def encode_view_jpeg(frame: np.ndarray) -> bytes:
    """Encode an LLM-bound view: cap the long edge at the shared
    ``[compact] max_image_edge_px`` knob, then JPEG-encode.

    Every producer of agent-facing views (peek, screenshot, gesture
    after-views) routes through this, so the size cap holds by
    construction — a new producer can't ship an oversized image just
    because its source frame wasn't pre-cropped.
    """
    return encode_jpeg(resize_to_max_edge(frame, CONFIG.compact.max_image_edge_px))


def decode_image(data: bytes) -> np.ndarray:
    """Decode image bytes (PNG or JPEG) to a BGR frame. Raises on failure."""
    arr = np.frombuffer(data, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise RuntimeError("Failed to decode image bytes")
    return frame


def check_phone_in_frame(frame: np.ndarray) -> dict:
    """Shape/coverage/straightness diagnostic from one overhead frame.

    Returns ``{ok, issues, coverage, aspect_ratio, image_size, phone_region}``.
    Saves an annotated frame to ``<tempdir>/physiclaw_camera_rotation.jpg``.
    Raises if no bright region is detected (camera read failed or phone off).
    """
    gray = grayscale(frame)
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


# Lock-screen text signatures for `looks_locked`: substrings of the
# keypad header ("Swipe up for Face ID or Enter Passcode" / "输入密码") —
# matched case-insensitively, so a partial OCR read still hits.
_LOCKED_PHRASES = ("passcode", "swipe up", "密码")
_TIME_RE = re.compile(r"\d{1,2}:\d{2}")
# The cover wallpaper clock spans ~0.8 of the screen width; home-screen
# clock widgets stay under ~0.45 and the status-bar clock under 0.2.
LOCK_CLOCK_MIN_WIDTH = 0.6
# The passcode prompt / keypad header sits high (measured y ≈ 0.18–0.22),
# above digit row 1 (~0.35). Only match lock phrases in this band: a
# generic "swipe up" or 密码 in an app's body (lower) is not the lock
# screen — and matching it there would forge "still locked" on a phone
# that unlocked into an app, sending the retry's wake tap into that app.
LOCK_PHRASE_MAX_Y = 0.33


def looks_locked(elements: list[dict]) -> bool:
    """Does this OCR listing look like the iOS lock/passcode screen?

    Two signatures, either one:
    - the cover wallpaper clock: a time-shaped label spanning most of
      the screen width, which OCRs reliably even on frames far below the
      blur threshold (measured at sharpness 7 in session logs);
    - a lock-screen phrase (`_LOCKED_PHRASES`) in the top band where the
      passcode prompt lives (`LOCK_PHRASE_MAX_Y`) — the keypad screen,
      which has no wide clock.

    Used by `unlock_phone` to verify the passcode actually took. Both
    signatures are deliberately narrow: a false "locked" costs one
    wasted agent retry whose wake tap can land inside a live app, so it
    must not fire on ordinary app content. Keypad digits are likewise
    NOT a signature — a home-screen badge count would read as a single
    digit in the keypad band and forge "locked".
    """
    for e in elements:
        label = e["label"].strip().lower()
        left, _, right, bottom = e["bbox"]
        if _TIME_RE.fullmatch(label) and right - left >= LOCK_CLOCK_MIN_WIDTH:
            return True
        if bottom <= LOCK_PHRASE_MAX_Y and any(p in label for p in _LOCKED_PHRASES):
            return True
    return False


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
