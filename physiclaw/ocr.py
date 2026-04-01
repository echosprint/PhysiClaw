"""
OCR module — read text from phone screen regions.

Uses RapidOCR (PaddleOCR models on ONNX Runtime) for lightweight,
torch-free text recognition. Supports Chinese + English out of the box.

Usage:
    from physiclaw.ocr import OCRReader

    reader = OCRReader()
    results = reader.read(screen_image)
    for r in results:
        print(f"{r.text}  conf={r.confidence:.2f}  bbox={r.bbox}")

    # Or read text within a specific region:
    text = reader.read_crop(screen_image, x1, y1, x2, y2)
"""

import dataclasses
import logging
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger(__name__)


@dataclasses.dataclass
class TextResult:
    """A detected text region."""
    text: str
    bbox: tuple[int, int, int, int]  # (x1, y1, x2, y2) in original image pixels
    confidence: float


class OCRReader:
    """Read text from phone screen images using RapidOCR."""

    def __init__(self):
        try:
            from rapidocr import RapidOCR
        except ImportError:
            raise ImportError(
                "rapidocr is required.\n"
                "Run: /setup-vision-models"
            )
        logging.disable(logging.INFO)
        self._ocr = RapidOCR()
        logging.disable(logging.NOTSET)
        logging.getLogger("RapidOCR").setLevel(logging.WARNING)

    def read(self, frame: np.ndarray) -> list[TextResult]:
        """Detect and read all text in an image.

        Args:
            frame: BGR image (numpy array).

        Returns:
            List of TextResult sorted top-to-bottom, left-to-right.
        """
        result = self._ocr(frame)

        if result.boxes is None or result.txts is None:
            return []

        texts = []
        for points, text, score in zip(result.boxes, result.txts, result.scores):
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            x1, y1 = int(min(xs)), int(min(ys))
            x2, y2 = int(max(xs)), int(max(ys))

            texts.append(TextResult(
                text=text,
                bbox=(x1, y1, x2, y2),
                confidence=float(score),
            ))

        texts.sort(key=lambda t: (t.bbox[1], t.bbox[0]))
        log.debug(f"OCR found {len(texts)} text regions")
        return texts

    def read_crop(self, frame: np.ndarray,
                  x1: int, y1: int, x2: int, y2: int) -> str:
        """Read text within a specific bounding box region.

        Args:
            frame: BGR image (numpy array).
            x1, y1, x2, y2: bounding box in pixel coordinates.

        Returns:
            Concatenated text found in the region, or empty string.
        """
        h, w = frame.shape[:2]
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return ""

        results = self.read(crop)
        return " ".join(r.text for r in results)


def annotate(frame: np.ndarray, results: list[TextResult]) -> np.ndarray:
    """Draw text bounding boxes and labels on a frame. Returns a copy."""
    out = frame.copy()
    for r in results:
        x1, y1, x2, y2 = r.bbox
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 0, 255), 2)

        label = f"{r.text} ({r.confidence:.2f})"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.4
        thickness = 1
        (tw, th), _ = cv2.getTextSize(label, font, font_scale, thickness)

        cv2.rectangle(out, (x1, y1 - th - 4), (x1 + tw + 4, y1), (0, 0, 255), -1)
        cv2.putText(out, label, (x1 + 2, y1 - 2), font, font_scale,
                    (255, 255, 255), thickness)

    return out


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="OCR on a phone screenshot")
    parser.add_argument("image", help="Path to image")
    parser.add_argument("-o", "--output", help="Save annotated image to this path")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG, format="%(message)s")

    img = cv2.imread(args.image)
    if img is None:
        raise SystemExit(f"Cannot read image: {args.image}")

    reader = OCRReader()
    results = reader.read(img)

    print(f"Found {len(results)} text regions:")
    for r in results:
        print(f"  {r.text:30s}  conf={r.confidence:.2f}  bbox={r.bbox}")

    output_path = args.output or str(Path(args.image).stem) + "_ocr.jpg"
    annotated = annotate(img, results)
    cv2.imwrite(output_path, annotated)
    print(f"Saved: {output_path}")
