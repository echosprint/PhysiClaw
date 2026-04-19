"""Rendering helpers — draw overlays on camera/screenshot frames."""

import cv2
import numpy as np


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


_GREEN, _RED = (0, 255, 0), (0, 0, 255)


def annotate_elements(
    frame: np.ndarray,
    elements: list[dict],
    w: int,
    h: int,
    *,
    include_text: bool = False,
) -> np.ndarray:
    """Draw numbered bboxes on a frame for detected UI elements.

    Args:
        elements: list of dicts with "id", "kind", "bbox" keys.
            bbox is [left, top, right, bottom] as 0-1 decimals.
        include_text: when False (default), skip `text` elements —
            their label is in the listing already, and the numbered
            tag tends to cover the first few characters of the text
            it annotates, making the image misleading. Icons have no
            label, so their numbered box is always drawn.
    """
    out = frame.copy()
    for e in elements:
        if not include_text and e["kind"] != "icon":
            continue
        x1, y1 = int(e["bbox"][0] * w), int(e["bbox"][1] * h)
        x2, y2 = int(e["bbox"][2] * w), int(e["bbox"][3] * h)
        color = _GREEN if e["kind"] == "icon" else _RED
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        lbl = str(e["id"])
        (tw, th), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        cv2.rectangle(out, (x1, y1 - th - 4), (x1 + tw + 4, y1), color, -1)
        cv2.putText(
            out, lbl, (x1 + 2, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2
        )
    return out
