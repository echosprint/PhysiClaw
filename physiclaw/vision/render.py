"""
Rendering helpers — draw bboxes, grid overlays, watermarks, and JPEG encode.

All image-output operations the project performs live here. Pure
functions: take a frame (and optionally a ScreenTransforms), return an
annotated frame or JPEG bytes. No hardware dependency.
"""

import cv2
import numpy as np

# Drawing constants
BBOX_COLOR = (0, 255, 0)  # green BGR

GRID_COLOR_MAP = {
    "green": (0, 255, 0),
    "red": (0, 0, 255),
    "yellow": (0, 255, 255),
}

def draw_bbox(frame: np.ndarray, bbox: list[float], transforms) -> np.ndarray:
    """Draw a green rectangle for a 0-1 bbox onto a copy of the frame."""
    tl, br = transforms.bbox_to_pixel_rect(bbox)
    out = frame.copy()
    cv2.rectangle(out, tl, br, BBOX_COLOR, 2)
    return out


def draw_grid_overlay(
    frame: np.ndarray, transforms, color: str = "green", rows: int = 9, cols: int = 4
) -> np.ndarray:
    """Draw evenly-spaced reference grid lines on a copy of the frame.

    Args:
        color: line color — "green", "red", or "yellow".
        rows: number of horizontal lines (e.g. 9 → 0.10, 0.20, ..., 0.90).
        cols: number of vertical lines (e.g. 4 → 0.20, 0.40, 0.60, 0.80).
    """
    bgr = GRID_COLOR_MAP.get(color, (0, 255, 0))
    out = frame.copy()
    cal = transforms
    font = cv2.FONT_HERSHEY_SIMPLEX

    pad = 4

    def _draw_label(canvas, label, cx, cy):
        (tw, th), _ = cv2.getTextSize(label, font, 0.8, 2)
        lx, ly = cx - tw // 2, cy + th // 2
        cv2.rectangle(
            canvas, (lx - pad, ly - th - pad), (lx + tw + pad, ly + pad), bgr, -1
        )
        cv2.putText(canvas, label, (lx, ly), font, 0.8, (255, 255, 255), 2)

    # Vertical lines — labels at top and bottom
    for i in range(1, cols + 1):
        x_val = round(i / (cols + 1), 2)
        pt_top = cal.pct_to_cam_pixel(x_val, 0)
        pt_bot = cal.pct_to_cam_pixel(x_val, 1)
        cv2.line(out, pt_top, pt_bot, bgr, 1)
        label = f"{x_val:.2f}"
        _draw_label(out, label, pt_top[0], pt_top[1] - 15)
        _draw_label(out, label, pt_bot[0], pt_bot[1] + 15)

    # Horizontal lines — labels at left and right
    for i in range(1, rows + 1):
        y_val = round(i / (rows + 1), 2)
        pt_left = cal.pct_to_cam_pixel(0, y_val)
        pt_right = cal.pct_to_cam_pixel(1, y_val)
        cv2.line(out, pt_left, pt_right, bgr, 1)
        label = f"{y_val:.2f}"
        (tw, _), _ = cv2.getTextSize(label, font, 0.8, 2)
        _draw_label(out, label, pt_left[0] - tw // 2 - 10, pt_left[1])
        _draw_label(out, label, pt_right[0] + tw // 2 + 10, pt_right[1])

    return out


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


def watermark_index(frame: np.ndarray, index: int) -> np.ndarray:
    """Draw a large semi-transparent index label in the center of the frame.

    Used by /api/camera-preview/{index} so the user can tell which camera
    a preview JPEG belongs to when several previews are open at once.
    Returns a copy of the frame with the watermark applied.
    """
    out = frame.copy()
    h, w = out.shape[:2]
    label = str(index)
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = h / 150
    thickness = max(2, int(scale * 2))
    (tw, th), _ = cv2.getTextSize(label, font, scale, thickness)
    cx, cy = w // 2, h // 2

    # 50% opacity black background plate
    overlay = out.copy()
    pad = int(scale * 20)
    cv2.rectangle(
        overlay,
        (cx - tw // 2 - pad, cy - th // 2 - pad),
        (cx + tw // 2 + pad, cy + th // 2 + pad),
        (0, 0, 0),
        -1,
    )
    cv2.addWeighted(overlay, 0.5, out, 0.5, 0, out)

    # White label on top
    cv2.putText(
        out,
        label,
        (cx - tw // 2, cy + th // 2),
        font,
        scale,
        (255, 255, 255),
        thickness,
    )
    return out
