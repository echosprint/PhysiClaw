"""
OmniParser icon detection — find interactable UI elements on a phone screen.

Uses the OmniParser V2 icon detection model (YOLO11m, finetuned by Microsoft)
via OpenCV DNN from an ONNX export. No torch dependency at runtime.

Usage:
    from physiclaw.vision.icon_detect import IconDetector, annotate

    detector = IconDetector()
    elements = detector.detect(screen_image)
    for e in elements:
        print(f"{e.bbox}  conf={e.confidence:.2f}")

Best practices:
    - Crop the phone screen before detection. Raw camera frames contain desk,
      cables, etc. that waste resolution. The model was trained on clean phone
      screenshots. Cropping gives each UI element 2-3x more pixels and
      significantly improves recall. Use GridCalibration transforms to crop.
    - Lower the confidence threshold for camera frames. Dark icons (TikTok,
      Spotify) on dark backgrounds score 0.2-0.3 in camera frames but 0.6+
      in clean screenshots. A threshold of 0.2 recovers these without adding
      much noise.
"""

import dataclasses
import logging
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger(__name__)

MODEL_PATH = Path(__file__).parent.parent.parent / "data" / "model" / "omniparser_icon_detect" / "model.onnx"
INPUT_SIZE = 1280  # V2 was trained at 1280
MIN_CONFIDENCE = 0.3
NMS_THRESHOLD = 0.5


@dataclasses.dataclass
class Element:
    """A detected UI element."""
    bbox: tuple[int, int, int, int]  # (x1, y1, x2, y2) in original image pixels
    confidence: float


class IconDetector:
    """Detect interactable UI elements using OmniParser V2 icon detect."""

    def __init__(self, model_path: Path = MODEL_PATH):
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model not found at {model_path}\n"
                "Run: /setup-vision-models"
            )
        self.net = cv2.dnn.readNetFromONNX(str(model_path))

    def detect(self, frame: np.ndarray,
               confidence: float = MIN_CONFIDENCE) -> list[Element]:
        """Detect UI elements in a phone screen image.

        Args:
            frame: BGR image (numpy array) of the phone screen.
            confidence: minimum confidence threshold.

        Returns:
            List of Element with bbox in original image coordinates.
        """
        h, w = frame.shape[:2]

        # Letterbox: resize longest edge to INPUT_SIZE, pad bottom-right to square
        scale = INPUT_SIZE / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        resized = cv2.resize(frame, (new_w, new_h))
        padded = np.full((INPUT_SIZE, INPUT_SIZE, 3), 114, dtype=np.uint8)
        padded[:new_h, :new_w] = resized

        # BGR → RGB, HWC → CHW, [0-255] → [0-1], add batch dim
        blob = padded[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
        blob = blob[np.newaxis]

        self.net.setInput(blob)
        output = self.net.forward()[0]  # (5, 33600)

        # YOLOv8/11 output: (5, N) → transpose to (N, 5)
        # Columns: cx, cy, w, h, conf (single class)
        preds = output.T  # (33600, 5)

        # Filter by confidence
        scores = preds[:, 4]
        mask = scores >= confidence
        preds = preds[mask]
        scores = scores[mask]

        if len(preds) == 0:
            return []

        # Convert cx, cy, w, h → x1, y1, w, h (for NMS)
        boxes_xywh = preds[:, :4].copy()
        boxes_xywh[:, 0] -= boxes_xywh[:, 2] / 2  # x1
        boxes_xywh[:, 1] -= boxes_xywh[:, 3] / 2  # y1

        # NMS
        indices = cv2.dnn.NMSBoxes(
            boxes_xywh.tolist(), scores.tolist(),
            confidence, NMS_THRESHOLD,
        )

        elements = []
        for i in indices:
            cx, cy, bw, bh = preds[i, :4]
            conf = float(scores[i])

            # Map back to original image coordinates
            x1 = max(0, (cx - bw / 2) / scale)
            y1 = max(0, (cy - bh / 2) / scale)
            x2 = min(w, (cx + bw / 2) / scale)
            y2 = min(h, (cy + bh / 2) / scale)

            elements.append(Element(
                bbox=(int(x1), int(y1), int(x2), int(y2)),
                confidence=conf,
            ))

        # Sort top-to-bottom, then left-to-right
        elements.sort(key=lambda e: (e.bbox[1], e.bbox[0]))

        log.debug(f"Detected {len(elements)} UI elements")
        return elements


def annotate(frame: np.ndarray, elements: list[Element]) -> np.ndarray:
    """Draw numbered bounding boxes on a frame. Returns a copy."""
    out = frame.copy()
    for i, e in enumerate(elements, 1):
        x1, y1, x2, y2 = e.bbox
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)

        label = f"{i}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        (tw, th), _ = cv2.getTextSize(label, font, font_scale, thickness)

        # Label background
        cv2.rectangle(out, (x1, y1 - th - 4), (x1 + tw + 4, y1), (0, 255, 0), -1)
        cv2.putText(out, label, (x1 + 2, y1 - 2), font, font_scale,
                    (0, 0, 0), thickness)

    return out


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Detect UI elements in a phone screenshot")
    parser.add_argument("image", help="Path to phone screen image")
    parser.add_argument("-c", "--confidence", type=float, default=MIN_CONFIDENCE)
    parser.add_argument("-o", "--output", help="Save annotated image to this path")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG, format="%(message)s")

    img = cv2.imread(args.image)
    if img is None:
        raise SystemExit(f"Cannot read image: {args.image}")

    detector = IconDetector()
    elements = detector.detect(img, confidence=args.confidence)

    print(f"Detected {len(elements)} elements:")
    for i, e in enumerate(elements, 1):
        x1, y1, x2, y2 = e.bbox
        print(f"  {i:3d}: ({x1}, {y1}, {x2}, {y2})  conf={e.confidence:.2f}")

    output_path = args.output or str(Path(args.image).stem) + "_detected.jpg"
    annotated = annotate(img, elements)
    cv2.imwrite(output_path, annotated)
    print(f"Saved: {output_path}")
