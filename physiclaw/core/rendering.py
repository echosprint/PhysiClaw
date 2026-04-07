"""
Rendering helpers — draw bboxes, grid overlays, and annotation listings.

Pure functions: take a frame and grid calibration, return an annotated
frame and/or a text listing. No hardware dependency.
"""

import cv2
import numpy as np

from physiclaw.annotation import classify_bbox

# Drawing constants
BBOX_COLOR = (0, 255, 0)  # green BGR

GRID_COLOR_MAP = {
    "green": (0, 255, 0),
    "red": (0, 0, 255),
    "yellow": (0, 255, 255),
}

# Hex color → human-readable name (for annotation listings)
_COLOR_NAMES = {
    '#ff5252': 'red', '#448aff': 'blue',
    '#69f0ae': 'green', '#ffd740': 'yellow',
    '#e040fb': 'purple', '#00e5ff': 'cyan',
    '#e0e0e0': 'white', '#b2ff59': 'lime',
}


def _hex_to_bgr(hex_color: str) -> tuple[int, int, int]:
    h = hex_color.lstrip('#')
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return (b, g, r)


def draw_bbox(frame: np.ndarray, bbox: list[float], grid_cal) -> np.ndarray:
    """Draw a green rectangle for a 0-1 bbox onto a copy of the frame."""
    tl, br = grid_cal.bbox_to_pixel_rect(bbox)
    out = frame.copy()
    cv2.rectangle(out, tl, br, BBOX_COLOR, 2)
    return out


def draw_grid_overlay(frame: np.ndarray, grid_cal,
                      color: str = "green",
                      rows: int = 9, cols: int = 4) -> np.ndarray:
    """Draw evenly-spaced reference grid lines on a copy of the frame.

    Args:
        color: line color — "green", "red", or "yellow".
        rows: number of horizontal lines (e.g. 9 → 0.10, 0.20, ..., 0.90).
        cols: number of vertical lines (e.g. 4 → 0.20, 0.40, 0.60, 0.80).
    """
    bgr = GRID_COLOR_MAP.get(color, (0, 255, 0))
    out = frame.copy()
    cal = grid_cal
    font = cv2.FONT_HERSHEY_SIMPLEX

    pad = 4

    def _draw_label(canvas, label, cx, cy):
        (tw, th), _ = cv2.getTextSize(label, font, 0.8, 2)
        lx, ly = cx - tw // 2, cy + th // 2
        cv2.rectangle(canvas, (lx - pad, ly - th - pad),
                      (lx + tw + pad, ly + pad), bgr, -1)
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


def process_annotations(frame: np.ndarray,
                        annotations: list[dict],
                        grid_cal) -> tuple[str, np.ndarray] | None:
    """Convert pixel-coordinate annotations to 0-1 screen coords.

    Draws colored numbered boxes on a copy of the frame and returns a text
    listing with coordinates as 0-1 decimals [left, top, right, bottom].

    Args:
        frame: BGR numpy array (the frozen snapshot)
        annotations: list of {left, top, right, bottom, color?, label?, source?}
                     in image pixels
        grid_cal: GridCalibration for pixel → 0-1 conversion

    Returns:
        (text_listing, annotated_frame) or None if annotations is empty.
    """
    if not annotations:
        return None

    cal = grid_cal
    out = frame.copy()
    elements = []
    for i, ann in enumerate(annotations):
        l, t = cal.pixel_to_pct(int(ann['left']), int(ann['top']))
        r, b = cal.pixel_to_pct(int(ann['right']), int(ann['bottom']))
        bbox = [
            max(0.0, min(1.0, round(l, 3))),
            max(0.0, min(1.0, round(t, 3))),
            max(0.0, min(1.0, round(r, 3))),
            max(0.0, min(1.0, round(b, 3))),
        ]
        box_type, coords = classify_bbox(bbox)
        color = ann.get('color', '#42a5f5')
        label = ann.get('label', '')
        source = ann.get('source', 'user')
        elements.append({'id': i + 1, 'type': box_type, 'bbox': coords,
                         'color': color, 'label': label, 'source': source})
        bgr = _hex_to_bgr(color)
        cv2.rectangle(out,
                      (int(ann['left']), int(ann['top'])),
                      (int(ann['right']), int(ann['bottom'])),
                      bgr, 2)
        cv2.putText(out, str(i + 1),
                    (int(ann['left']) + 4, int(ann['top']) + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, bgr, 2)

    lines = [f"# Pending Annotations ({len(elements)} items)\n"]
    for e in elements:
        name = _COLOR_NAMES.get(e['color'], e['color'])
        b = e['bbox']
        desc = f" — {e['label']}" if e['label'] else ""
        src = f" [{e['source']}]" if e['source'] != 'user' else ""
        coords = ", ".join(str(v) for v in b)
        type_tag = f" ({e['type']})" if e['type'] != 'box' else ""
        lines.append(f"- {e['id']}{type_tag} ({name}){src}: [{coords}]{desc}")
    return "\n".join(lines), out


def encode_jpeg(frame: np.ndarray, quality: int = 85) -> bytes:
    """Encode a BGR frame to JPEG bytes."""
    _, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return jpeg.tobytes()
