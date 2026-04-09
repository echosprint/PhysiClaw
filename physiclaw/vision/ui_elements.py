"""Detect UI elements via icon detection + OCR.

Runs both detectors, merges overlapping icon+text into labeled buttons,
assigns contiguous IDs, draws bboxes on one annotated image, and returns
structured JSON with 0-1 normalized coordinates.

Three element types in output:
- button (orange): icon with text inside, e.g. "button: Buy Now"
- icon (green): bare icon, no text overlap
- text (red): standalone OCR text

Output format:
    [
      {"id": 0, "label": "button: Buy Now", "bbox": [0.03, 0.90, 0.97, 0.95], "conf": 0.92},
      {"id": 1, "label": "icon", "bbox": [0.02, 0.06, 0.11, 0.10], "conf": 0.64},
      {"id": 2, "label": "text: $29.9", "bbox": [0.37, 0.54, 0.51, 0.56], "conf": 0.91},
    ]
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger(__name__)

_GREEN = (0, 255, 0)       # icon
_RED = (0, 0, 255)         # text
_ORANGE = (255, 165, 0)    # button (merged icon+text)


@dataclass
class UIElement:
    id: int
    label: str          # "icon", "text: <content>", or "button: <content>"
    bbox: list[float]   # [left, top, right, bottom] as 0-1
    conf: float

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "label": self.label,
            "bbox": [round(v, 3) for v in self.bbox],
            "conf": round(self.conf, 2),
        }


# ── Detection ──────────────────────────────────


def detect_ui_elements(
    image_path: str | Path,
    icon_detector=None,
    ocr_reader=None,
    icon_confidence: float = 0.2,
) -> tuple[list[UIElement], np.ndarray]:
    """Run icon detection + OCR → structured elements + annotated image."""
    frame = cv2.imread(str(image_path))
    if frame is None:
        raise FileNotFoundError(f"cannot read image: {image_path}")
    h, w = frame.shape[:2]

    icons = _detect_icons(frame, w, h, icon_detector, icon_confidence)
    texts = _detect_texts(frame, w, h, ocr_reader)
    merged = _merge(icons, texts)

    # Sort top-to-bottom left-to-right, assign final IDs
    merged.sort(key=lambda e: (e.bbox[1], e.bbox[0]))
    for i, e in enumerate(merged):
        e.id = i

    annotated = _annotate(frame, merged, w, h)

    n_btn = sum(1 for e in merged if e.label.startswith("button:"))
    n_ico = sum(1 for e in merged if e.label == "icon")
    n_txt = sum(1 for e in merged if e.label.startswith("text:"))
    log.info("detected %d UI elements (%d buttons, %d icons, %d text)",
             len(merged), n_btn, n_ico, n_txt)

    return merged, annotated


def _detect_icons(frame, w, h, detector, confidence) -> list[UIElement]:
    try:
        from physiclaw.vision.icon_detect import IconDetector
        if detector is None:
            detector = IconDetector()
        return [
            UIElement(0, "icon", [e.bbox[0]/w, e.bbox[1]/h, e.bbox[2]/w, e.bbox[3]/h], e.confidence)
            for e in detector.detect(frame, confidence=confidence)
        ]
    except (ImportError, FileNotFoundError) as ex:
        log.warning("icon detection unavailable: %s", ex)
        return []


def _detect_texts(frame, w, h, reader) -> list[UIElement]:
    try:
        from physiclaw.vision.ocr import OCRReader
        if reader is None:
            reader = OCRReader()
        return [
            UIElement(0, f"text: {t.text}", [t.bbox[0]/w, t.bbox[1]/h, t.bbox[2]/w, t.bbox[3]/h], t.confidence)
            for t in reader.read(frame)
        ]
    except ImportError as ex:
        log.warning("OCR unavailable: %s", ex)
        return []


# ── Merge icon + text → button ─────────────────


def _merge(icons: list[UIElement], texts: list[UIElement]) -> list[UIElement]:
    """If text center falls inside an icon → merge into button."""
    consumed: set[int] = set()
    merged: list[UIElement] = []
    for icon in icons:
        labels = []
        for j, txt in enumerate(texts):
            if j not in consumed and _center_inside(txt.bbox, icon.bbox):
                labels.append(txt.label.removeprefix("text: "))
                consumed.add(j)
        if labels:
            icon.label = "button: " + " ".join(labels)
        merged.append(icon)
    for j, txt in enumerate(texts):
        if j not in consumed:
            merged.append(txt)
    return merged


def _center_inside(inner: list[float], outer: list[float]) -> bool:
    cx = (inner[0] + inner[2]) / 2
    cy = (inner[1] + inner[3]) / 2
    return outer[0] <= cx <= outer[2] and outer[1] <= cy <= outer[3]


# ── Annotation ─────────────────────────────────


def _annotate(frame: np.ndarray, elements: list[UIElement],
              w: int, h: int) -> np.ndarray:
    out = frame.copy()
    for e in elements:
        x1, y1 = int(e.bbox[0] * w), int(e.bbox[1] * h)
        x2, y2 = int(e.bbox[2] * w), int(e.bbox[3] * h)
        if e.label.startswith("button:"):
            color = _ORANGE
        elif e.label == "icon":
            color = _GREEN
        else:
            color = _RED
        _draw_box(out, e.id, (x1, y1, x2, y2), color)
    return out


def _draw_box(frame: np.ndarray, element_id: int,
              bbox_px: tuple[int, int, int, int],
              color: tuple[int, int, int]) -> None:
    x1, y1, x2, y2 = bbox_px
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    label = str(element_id)
    font_scale, thickness = 0.8, 2
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    cv2.rectangle(frame, (x1, y1 - th - 4), (x1 + tw + 4, y1), color, -1)
    cv2.putText(frame, label, (x1 + 2, y1 - 2),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness)


# ── JSON helpers ───────────────────────────────


def elements_to_json(elements: list[UIElement]) -> list[dict]:
    return [e.to_dict() for e in elements]


def _compact_json(items: list[dict]) -> str:
    """One JSON object per line."""
    lines = [json.dumps(item, ensure_ascii=False) for item in items]
    return "[\n" + ",\n".join(f"  {line}" for line in lines) + "\n]\n"


# ── CLI ────────────────────────────────────────


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Detect UI elements (icons + OCR)")
    parser.add_argument("image", help="Path to phone screen image")
    parser.add_argument("-c", "--confidence", type=float, default=0.2,
                        help="Icon confidence threshold (default: 0.2)")
    parser.add_argument("-o", "--output", help="Output path for annotated image")
    parser.add_argument("-t", "--task", default=None, help="Task hint for semantic relevance")
    parser.add_argument("--json", action="store_true", help="Output JSON only")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    elements, annotated = detect_ui_elements(args.image, icon_confidence=args.confidence)

    if args.json:
        print(_compact_json(elements_to_json(elements)))
    else:
        print(f"{len(elements)} elements detected:")
        for e in elements:
            print(f"  [{e.id:3d}] {e.label:30s}  bbox={[round(v,3) for v in e.bbox]}  conf={e.conf:.2f}")

    output_path = Path(args.output) if args.output else \
        Path("data/snapshot") / (Path(args.image).stem + "_ui_elements.jpg")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), annotated)

    elements_json = elements_to_json(elements)
    json_path = output_path.with_suffix(".json")
    json_path.write_text(_compact_json(elements_json))

    from physiclaw.vision.semantic import SemanticDescriber
    result = SemanticDescriber(auto_zone=True, show_conf=True).describe(
        elements_json, task_hint=args.task)
    md_path = output_path.with_suffix(".md")
    md_path.write_text(result["prompt"])

    print(f"image: {output_path}")
    print(f"json:  {json_path}")
    print(f"semantic: {md_path}")
    print(f"  {result['element_count']} elements across zones: {result['zones_used']}")
