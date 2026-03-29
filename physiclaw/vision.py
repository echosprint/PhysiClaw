"""
Vision module — phone detection and camera discovery.

Uses YOLOX Nano (COCO class 67 = cell phone) via cv2.dnn to:
  1. Check if a phone is placed on the platform
  2. Find the camera that can see the phone
"""

import logging
from pathlib import Path

import cv2
import numpy as np

from physiclaw.camera import Camera

log = logging.getLogger(__name__)

MODEL_PATH = Path(__file__).parent.parent / 'data' / 'model' / 'yolox_tiny' / 'yolox_tiny.onnx'
MODEL_URL = 'https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_tiny.onnx'

COCO_PHONE_CLASS = 67  # cell phone
INPUT_SIZE = 416
MIN_CONFIDENCE = 0.3

def _download_model(path: Path):
    """Download YOLOX Nano model automatically."""
    import urllib.request
    log.info(f"Downloading YOLOX Nano to {path} ...")
    path.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(MODEL_URL, path)
    log.info("Download complete")


class PhoneDetector:
    """Detect phone presence and find camera using YOLOX Nano."""

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

    def is_phone_placed(self, camera_index: int = 0) -> bool:
        """Quick check: is a phone visible from this camera?
        Use to verify the phone is on the platform before starting.
        """
        try:
            cam = Camera(camera_index)
        except RuntimeError:
            return False
        frame = cam.snapshot()
        cam.close()
        if frame is None:
            return False
        detected, _, _ = self.detect(frame)
        return detected

    def detect_from_camera(self, camera_index: int = 0, save_crop: bool = False) -> bool:
        """Grab a frame from the camera, save snapshot, and check for phone."""
        from datetime import datetime

        cam = Camera(camera_index)
        frame = cam.snapshot()
        cam.close()

        if frame is None:
            log.warning("Failed to capture frame")
            return False

        # Save snapshot for debugging
        snapshot_dir = Path(__file__).parent.parent / 'data' / 'snapshot'
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
        snapshot_path = snapshot_dir / f'cam{camera_index}_{timestamp}.jpg'
        cv2.imwrite(str(snapshot_path), frame)
        log.debug(f"Snapshot saved: {snapshot_path}")

        detected, conf, bbox = self.detect(frame)
        if detected:
            log.debug(f"Phone detected ({conf:.0%})  bbox: {[round(v) for v in bbox]}")
            if save_crop:
                h, w = frame.shape[:2]
                x1 = max(0, int(bbox[0]))
                y1 = max(0, int(bbox[1]))
                x2 = min(w, int(bbox[2]))
                y2 = min(h, int(bbox[3]))
                crop = frame[y1:y2, x1:x2]
                crop_path = snapshot_dir / f'cam{camera_index}_{timestamp}_crop.jpg'
                cv2.imwrite(str(crop_path), crop)
                log.debug(f"Cropped saved: {crop_path}")
        else:
            log.debug(f"No phone detected (best: {conf:.0%})")
        return detected

    def find_camera(self, max_index: int = 8) -> Camera | None:
        """Scan cameras and return the one with highest phone detection confidence.

        The top-down USB camera sees the phone large in frame (high confidence).
        A laptop webcam might barely see it from a distance (low confidence).
        Picking highest confidence selects the right camera regardless of index order.

        The returned Camera is kept open and ready for use.
        Returns None if no camera sees a phone.
        """
        log.debug("Scanning for camera that sees the phone...")

        # Detect how many cameras are available
        import subprocess
        try:
            result = subprocess.run(
                ['system_profiler', 'SPCameraDataType'],
                capture_output=True, text=True, timeout=5,
            )
            cam_count = result.stdout.count('Model ID:')
            max_index = min(max_index, max(cam_count, 1))
            log.info(f"System reports {cam_count} cameras, scanning indices 0-{max_index - 1}")
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass  # not macOS or timeout — fall back to max_index

        best_cam = None
        best_conf = 0.0

        for i in range(max_index):
            try:
                cam = Camera(i)
            except RuntimeError:
                continue

            frame = cam.snapshot()
            if frame is None:
                cam.close()
                continue

            detected, conf, _ = self.detect(frame)
            if detected:
                log.info(f"Camera {i}: phone detected ({conf:.0%})")
                if conf > best_conf:
                    if best_cam is not None:
                        best_cam.close()
                    best_cam = cam
                    best_conf = conf
                else:
                    cam.close()
            else:
                log.debug(f"Camera {i}: no phone detected ({conf:.0%})")
                cam.close()

        if best_cam is not None:
            log.info(f"Selected camera {best_cam.index} ({best_conf:.0%} confidence)")
        else:
            log.warning("No phone detected on any camera — is the phone on the platform?")
        return best_cam
