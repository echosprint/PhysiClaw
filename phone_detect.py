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

    @staticmethod
    def _generate_grids():
        """Generate YOLOX grid offsets and strides for 416x416 input."""
        grids = []
        strides = []
        for stride in [8, 16, 32]:
            grid_size = INPUT_SIZE // stride
            yv, xv = np.meshgrid(np.arange(grid_size), np.arange(grid_size), indexing='ij')
            grid = np.stack([xv, yv], axis=-1).reshape(-1, 2)
            grids.append(grid)
            strides.append(np.full((grid_size * grid_size, 1), stride))
        return np.concatenate(grids, axis=0), np.concatenate(strides, axis=0)

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

        # Decode YOLOX grid-based output to absolute coords
        grids, strides = self._generate_grids()
        out[:, :2] = (out[:, :2] + grids) * strides       # cx, cy
        out[:, 2:4] = np.exp(out[:, 2:4]) * strides       # w, h

        # YOLOX outputs raw logits — apply sigmoid for confidence
        obj_conf = 1.0 / (1.0 + np.exp(-out[:, 4]))
        cls_score = 1.0 / (1.0 + np.exp(-out[:, 5 + COCO_PHONE_CLASS]))
        phone_scores = obj_conf * cls_score

        idx = phone_scores.argmax()
        best = float(phone_scores[idx])

        if best <= MIN_CONFIDENCE:
            return False, best, None

        # Map bbox back to original image coordinates
        cx, cy, bw, bh = out[idx, :4]
        x1 = max(0, (cx - bw / 2) / scale)
        y1 = max(0, (cy - bh / 2) / scale)
        x2 = min(w, (cx + bw / 2) / scale)
        y2 = min(h, (cy + bh / 2) / scale)

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
            ratio = self.bbox_aspect_ratio(bbox)
            view = self.classify_view(ratio)
            print(f"Phone detected ({conf:.0%})  bbox: {[round(v) for v in bbox]}  "
                  f"ratio: {ratio:.2f}  view: {view}")
            # Save cropped phone region
            h, w = frame.shape[:2]
            x1 = max(0, int(bbox[0]))
            y1 = max(0, int(bbox[1]))
            x2 = min(w, int(bbox[2]))
            y2 = min(h, int(bbox[3]))
            crop = frame[y1:y2, x1:x2]
            crop_path = snapshot_dir / f'cam{camera_index}_{timestamp}_crop.jpg'
            cv2.imwrite(str(crop_path), crop)
            print(f"Cropped saved: {crop_path}")
        else:
            print(f"No phone detected (best: {conf:.0%})")
        return detected

    @staticmethod
    def bbox_aspect_ratio(bbox: list) -> float:
        """max(w, h) / min(w, h) — always >= 1, orientation-independent."""
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        if w <= 0 or h <= 0:
            return 0
        return max(w, h) / min(w, h)

    @staticmethod
    def classify_view(ratio: float) -> str:
        """Classify camera view based on phone bbox aspect ratio.
        Top camera:  sees phone face-on → ratio ~1.5-2.5 (natural phone shape)
        Side camera: sees phone at 45° → perspective distortion → ratio > 2.5 or < 1.5
        """
        if 1.5 <= ratio <= 2.5:
            return 'top'
        return 'side'

    def identify_cameras(self, max_index: int = 5) -> dict[str, int]:
        """Scan camera indices and identify top vs side camera.
        Returns {'top': index, 'side': index} or partial dict.
        """
        from camera import Camera
        result = {}

        for i in range(max_index):
            try:
                cam = Camera(i)
            except RuntimeError:
                continue

            frame = cam.snapshot()
            cam.close()
            if frame is None:
                continue

            detected, conf, bbox = self.detect(frame)
            if not detected:
                print(f"  Camera {i}: no phone detected")
                continue

            ratio = self.bbox_aspect_ratio(bbox)
            view = self.classify_view(ratio)
            print(f"  Camera {i}: phone detected ({conf:.0%})  ratio: {ratio:.2f} → {view}")

            if view not in result:
                result[view] = i

        return result


def main():
    parser = argparse.ArgumentParser(description="Detect phone in camera view")
    parser.add_argument("--camera", type=int, default=None,
                        help="Check a specific camera index")
    parser.add_argument("--identify", action="store_true",
                        help="Scan all cameras and identify top vs side")
    args = parser.parse_args()

    detector = PhoneDetector()

    if args.identify:
        print("Scanning cameras...")
        cameras = detector.identify_cameras()
        print(f"\nResult: {cameras}")
        if 'top' in cameras:
            print(f"  Top camera:  index {cameras['top']}")
        if 'side' in cameras:
            print(f"  Side camera: index {cameras['side']}")
    else:
        found = detector.check_camera(args.camera or 0)
        sys.exit(0 if found else 1)


if __name__ == "__main__":
    main()
