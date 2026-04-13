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


def compact_json(items: list[dict]) -> str:
    """Pretty-print a list of dicts with one item per line."""
    lines = [json.dumps(item, ensure_ascii=False) for item in items]
    return "[\n" + ",\n".join(f"  {line}" for line in lines) + "\n]\n"
