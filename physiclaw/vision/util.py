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


def compact_json(items: list[dict]) -> str:
    """Pretty-print a list of dicts with one item per line."""
    lines = [json.dumps(item, ensure_ascii=False) for item in items]
    return "[\n" + ",\n".join(f"  {line}" for line in lines) + "\n]\n"
