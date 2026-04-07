"""
Multi-tool element detection — runs the 3 CV detectors on a frame.

Combines color segmentation, icon detection, and OCR into a single
analysis pass. Returns a markdown-formatted text listing plus three
annotated frames (one per detector).

Pure function: frame in → results out. No hardware dependency.
"""

import logging
from datetime import datetime

import cv2
import numpy as np

from physiclaw.vision.color_segment import detect_color_blocks
from physiclaw.vision.color_segment import annotate as color_annotate

log = logging.getLogger(__name__)


def detect_all_elements(
    frame: np.ndarray,
    transforms,
    icon_detector=None,
    ocr_reader=None,
) -> tuple[str, np.ndarray, np.ndarray, np.ndarray]:
    """Run color + icon + OCR detection on a frame.

    Args:
        frame: BGR camera frame.
        transforms: ScreenTransforms for converting pixel → 0-1 screen coords.
        icon_detector: optional cached IconDetector. If None, one is created
                       on demand (slower for repeated calls).
        ocr_reader: optional cached OCRReader. If None, one is created on demand.

    Returns:
        (elements_text, color_frame, icon_frame, ocr_frame)
        - elements_text: markdown listing all detected elements with 0-1 coords
        - color_frame, icon_frame, ocr_frame: annotated copies of the input
    """
    cal = transforms
    h, w = frame.shape[:2]
    element_id = 0

    # ── Tool 1: Color segmentation ─────────────────────────
    color_table_header = (
        "| id | color | type | bbox [left, top, right, bottom] | h_std |\n"
        "|----|-------|------|------|-------|"
    )
    color_rows = []
    color_frame = frame.copy()
    try:
        blobs = detect_color_blocks(frame)
        color_frame = color_annotate(frame, blobs)
        for blob in blobs:
            element_id += 1
            x1, y1, x2, y2 = blob.bbox
            l, t = cal.pixel_to_pct(int(x1), int(y1))
            r, b = cal.pixel_to_pct(int(x2), int(y2))
            kind = "image" if blob.is_image else "solid" if blob.is_solid else "mixed"
            color_rows.append(
                f"| {element_id} | {blob.color_name} | {kind} "
                f"| [{l:.2f}, {t:.2f}, {r:.2f}, {b:.2f}] "
                f"| {blob.h_std:.1f} |"
            )
    except Exception as ex:
        color_rows.append(f"| — | — | error: {ex} | — | — |")

    # ── Tool 2: Icon detection ─────────────────────────────
    icon_table_header = (
        "| id | bbox [left, top, right, bottom] | conf |\n|----|------|------|"
    )
    icon_rows = []
    icon_frame = frame.copy()
    try:
        from physiclaw.vision.icon_detect import IconDetector, annotate as icon_annotate

        if icon_detector is None:
            icon_detector = IconDetector()
        icons = icon_detector.detect(frame, confidence=0.2)
        icon_frame = icon_annotate(frame, icons)
        for e in icons:
            element_id += 1
            x1, y1, x2, y2 = e.bbox
            l, t = cal.pixel_to_pct(x1, y1)
            r, b = cal.pixel_to_pct(x2, y2)
            icon_rows.append(
                f"| {element_id} "
                f"| [{l:.2f}, {t:.2f}, {r:.2f}, {b:.2f}] "
                f"| {e.confidence:.2f} |"
            )
    except (ImportError, FileNotFoundError) as ex:
        icon_rows.append(f"| — | unavailable: {ex} | — |")

    # ── Tool 3: OCR ────────────────────────────────────────
    text_table_header = (
        "| id | label | bbox [left, top, right, bottom] | conf |\n"
        "|----|-------|------|------|"
    )
    ocr_rows = []
    ocr_frame = frame.copy()
    try:
        from physiclaw.vision.ocr import OCRReader, annotate as ocr_annotate

        if ocr_reader is None:
            ocr_reader = OCRReader()
        texts = ocr_reader.read(frame)
        ocr_frame = ocr_annotate(frame, texts)
        for t in texts:
            element_id += 1
            x1, y1, x2, y2 = t.bbox
            l, tp = cal.pixel_to_pct(x1, y1)
            r, b = cal.pixel_to_pct(x2, y2)
            ocr_rows.append(
                f'| {element_id} | "{t.text}" '
                f"| [{l:.2f}, {tp:.2f}, {r:.2f}, {b:.2f}] "
                f"| {t.confidence:.2f} |"
            )
    except ImportError as ex:
        ocr_rows.append(f"| — | unavailable: {ex} | — | — |")

    ts = datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ")
    elements_text = (
        f"# Screen Parse Result\n\n"
        f"- **resolution**: {w}x{h}\n"
        f"- **timestamp**: {ts}\n\n"
        f"## Color Blocks\n\n{color_table_header}\n"
        + "\n".join(color_rows)
        + f"\n\n## Icons\n\n{icon_table_header}\n"
        + "\n".join(icon_rows)
        + f"\n\n## Text\n\n{text_table_header}\n"
        + "\n".join(ocr_rows)
    )

    return elements_text, color_frame, icon_frame, ocr_frame
