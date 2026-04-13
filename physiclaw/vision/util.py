"""Image codec and serialization utilities."""

import json

import cv2
import numpy as np


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
    """Pretty-print a list of dicts with one item per line."""
    lines = [json.dumps(item, ensure_ascii=False) for item in items]
    return "[\n" + ",\n".join(f"  {line}" for line in lines) + "\n]\n"
