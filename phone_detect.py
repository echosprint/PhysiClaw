"""
Detect if a phone is visible in the camera frame using YOLOX Nano.

Uses yolox_nano.onnx (COCO object detector) via cv2.dnn.
COCO class 67 = cell phone.

Usage:
    uv run python phone_detect.py [--camera INDEX]
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

MODEL_PATH = Path(__file__).parent / 'data' / 'model' / 'yolox_nano' / 'yolox_nano.onnx'
MODEL_URL = 'https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_nano.onnx'

COCO_PHONE_CLASS = 67  # cell phone
INPUT_SIZE = 416
MIN_CONFIDENCE = 0.3


def _download_model(path: Path):
    """Download YOLOX Nano model automatically."""
    import urllib.request
    print(f"Downloading YOLOX Nano to {path} ...")
    path.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(MODEL_URL, path)
    print("Download complete")


class PhoneDetector:
    """Detect phone presence using YOLOX Nano (cv2.dnn)."""

    def __init__(self, model_path: Path = MODEL_PATH):
        if not model_path.exists():
            _download_model(model_path)
        self.net = cv2.dnn.readNetFromONNX(str(model_path))

    def detect(self, frame: np.ndarray) -> tuple[bool, float, list | None]:
        """Check if a phone is visible in the frame.
        Returns (detected, confidence, bbox [x1,y1,x2,y2] in original image coords or None).
        """
        # Letterbox resize — YOLOX requires aspect-preserving + gray padding
        h, w = frame.shape[:2]
        scale = min(INPUT_SIZE / h, INPUT_SIZE / w)
        new_h, new_w = int(h * scale), int(w * scale)
        resized = cv2.resize(frame, (new_w, new_h))
        padded = np.full((INPUT_SIZE, INPUT_SIZE, 3), 114, dtype=np.uint8)
        padded[:new_h, :new_w] = resized

        blob = cv2.dnn.blobFromImage(
            padded, 1.0, (INPUT_SIZE, INPUT_SIZE), swapRB=True, crop=False,
        )
        self.net.setInput(blob)
        out = self.net.forward()[0]  # (3549, 85)

        # YOLOX outputs raw logits — apply sigmoid
        obj_conf = 1.0 / (1.0 + np.exp(-out[:, 4]))
        cls_score = 1.0 / (1.0 + np.exp(-out[:, 5 + COCO_PHONE_CLASS]))
        phone_scores = obj_conf * cls_score

        idx = phone_scores.argmax()
        best = float(phone_scores[idx])

        if best <= MIN_CONFIDENCE:
            return False, best, None

        # Map bbox back to original image coordinates
        cx, cy, bw, bh = out[idx, :4]
        x1 = (cx - bw / 2) / scale
        y1 = (cy - bh / 2) / scale
        x2 = (cx + bw / 2) / scale
        y2 = (cy + bh / 2) / scale

        return True, best, [x1, y1, x2, y2]

    def check_camera(self, camera_index: int = 0) -> bool:
        """Grab a frame from the camera, save snapshot, and check for phone."""
        from datetime import datetime
        from camera import Camera

        cam = Camera(camera_index)
        frame = cam.snapshot()
        cam.close()

        if frame is None:
            print("Failed to capture frame")
            return False

        # Save snapshot for debugging
        snapshot_dir = Path(__file__).parent / 'data' / 'snapshot'
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
        snapshot_path = snapshot_dir / f'cam{camera_index}_{timestamp}.jpg'
        cv2.imwrite(str(snapshot_path), frame)
        print(f"Snapshot saved: {snapshot_path}")

        detected, conf, bbox = self.detect(frame)
        if detected:
            print(f"Phone detected ({conf:.0%})  bbox: {[round(v) for v in bbox]}")
        else:
            print(f"No phone detected (best: {conf:.0%})")
        return detected


def main():
    parser = argparse.ArgumentParser(description="Detect phone in camera view")
    parser.add_argument("--camera", type=int, default=0)
    args = parser.parse_args()

    detector = PhoneDetector()
    found = detector.check_camera(args.camera)
    sys.exit(0 if found else 1)


if __name__ == "__main__":
    main()
