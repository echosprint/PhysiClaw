"""Detect UI elements via icon detection + OCR.

Two element kinds — icon (label="") and text (label=content):

    [
      {"id": 0, "kind": "icon", "label": "",      "bbox": [0.02, 0.06, 0.11, 0.10], "conf": 0.64},
      {"id": 1, "kind": "text", "label": "$29.9", "bbox": [0.37, 0.54, 0.51, 0.56], "conf": 0.91}
    ]
"""

import logging
from dataclasses import dataclass

import numpy as np

from physiclaw.vision.render import annotate_elements

log = logging.getLogger(__name__)


@dataclass
class UIElement:
    id: int
    kind: str  # "icon" | "text"
    label: str  # "" for icon, OCR content for text
    bbox: list[float]  # [left, top, right, bottom] 0-1
    conf: float

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "kind": self.kind,
            "label": self.label,
            "bbox": [round(v, 3) for v in self.bbox],
            "conf": round(self.conf, 2),
        }


# ── Detection ─────────────────────────────────


def detect_ui_elements(
    frame: np.ndarray,
    icon_detector=None,
    ocr_reader=None,
    icon_confidence: float = 0.2,
    min_icon_conf: float = 0.3,
    min_text_conf: float = 0.7,
) -> tuple[list[UIElement], np.ndarray]:
    """Run icon detection + OCR → structured elements + annotated image.

    Args:
        frame: BGR numpy array. Use `cv2.imread(path)` for a file.
    """
    h, w = frame.shape[:2]

    icons = _detect_icons(frame, w, h, icon_detector, icon_confidence)
    texts = _detect_texts(frame, w, h, ocr_reader)
    elements = _clean(icons + texts, min_icon_conf, min_text_conf)

    elements.sort(key=lambda e: (e.bbox[1], e.bbox[0]))
    for i, e in enumerate(elements):
        e.id = i

    annotated = annotate_elements(frame, elements_to_json(elements), w, h)
    n_ico = sum(1 for e in elements if e.kind == "icon")
    log.info(
        "detected %d UI elements (%d icons, %d text)",
        len(elements),
        n_ico,
        len(elements) - n_ico,
    )
    return elements, annotated


def _detect_icons(frame, w, h, detector, confidence) -> list[UIElement]:
    try:
        from physiclaw.vision.icon_detect import IconDetector

        det = detector or IconDetector()
        return [
            UIElement(
                0,
                "icon",
                "",
                [e.bbox[0] / w, e.bbox[1] / h, e.bbox[2] / w, e.bbox[3] / h],
                e.confidence,
            )
            for e in det.detect(frame, confidence=confidence)
        ]
    except (ImportError, FileNotFoundError) as ex:
        log.warning("icon detection unavailable: %s", ex)
        return []


def _detect_texts(frame, w, h, reader) -> list[UIElement]:
    try:
        from physiclaw.vision.ocr import OCRReader

        rdr = reader or OCRReader()
        return [
            UIElement(
                0,
                "text",
                t.text,
                [t.bbox[0] / w, t.bbox[1] / h, t.bbox[2] / w, t.bbox[3] / h],
                t.confidence,
            )
            for t in rdr.read(frame)
        ]
    except ImportError as ex:
        log.warning("OCR unavailable: %s", ex)
        return []


# ── Filtering ─────────────────────────────────


def _clean(
    elements: list[UIElement], min_icon_conf: float = 0.3, min_text_conf: float = 0.7
) -> list[UIElement]:
    """Drop noise, then dedupe by IoU."""
    out = []
    for e in elements:
        w = e.bbox[2] - e.bbox[0]
        h = e.bbox[3] - e.bbox[1]
        if w <= 0 or h <= 0 or w * h < 1e-4:
            continue
        if e.kind == "icon" and e.conf < min_icon_conf:
            continue
        if e.kind == "text" and e.conf < min_text_conf:
            continue
        if e.kind == "icon" and w > 0.95 and h > 0.95:
            continue
        out.append(e)
    return _dedupe(out)


def _dedupe(elements: list[UIElement], iou_thresh: float = 0.7) -> list[UIElement]:
    """Remove near-duplicate boxes by IoU, keeping higher confidence."""
    elements = sorted(elements, key=lambda e: -e.conf)
    keep = []
    for e in elements:
        if any(_iou(e.bbox, k.bbox) > iou_thresh for k in keep):
            continue
        keep.append(e)
    return keep


def _iou(a: list[float], b: list[float]) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    if inter == 0:
        return 0.0
    return inter / (
        (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    )


def elements_to_json(elements: list[UIElement]) -> list[dict]:
    """Convert a list of UIElement objects to dicts."""
    return [e.to_dict() for e in elements]


# ── CLI ───────────────────────────────────────

if __name__ == "__main__":
    import argparse
    from pathlib import Path

    import cv2

    from physiclaw.vision.util import compact_json

    p = argparse.ArgumentParser(description="Detect UI elements (icons + OCR)")
    p.add_argument("image")
    p.add_argument("-c", "--confidence", type=float, default=0.2)
    p.add_argument("--min-icon-conf", type=float, default=0.3)
    p.add_argument("--min-text-conf", type=float, default=0.7)
    p.add_argument("-o", "--output", help="Output path for annotated image")
    p.add_argument("--json", action="store_true", help="Output JSON only")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    frame = cv2.imread(args.image)
    if frame is None:
        raise SystemExit(f"cannot read image: {args.image}")

    elements, annotated = detect_ui_elements(
        frame,
        icon_confidence=args.confidence,
        min_icon_conf=args.min_icon_conf,
        min_text_conf=args.min_text_conf,
    )

    if args.json:
        print(compact_json(elements_to_json(elements)))
    else:
        print(f"{len(elements)} elements detected:")
        for e in elements:
            print(
                f"  [{e.id:3d}] {e.label:30s}  bbox={[round(v, 3) for v in e.bbox]}  conf={e.conf:.2f}"
            )

    out_path = (
        Path(args.output)
        if args.output
        else Path("data/snapshot") / (Path(args.image).stem + "_ui_elements.jpg")
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), annotated)

    json_path = out_path.with_suffix(".json")
    json_path.write_text(compact_json(elements_to_json(elements)))

    print(f"image: {out_path}")
    print(f"json:  {json_path}")
