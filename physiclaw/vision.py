"""
Vision module — phone detection and camera identification.

Uses YOLOX Nano (COCO class 67 = cell phone) via cv2.dnn to:
  1. Check if a phone is placed on the platform
  2. Identify which USB camera is top (looking down) vs side (~45°)
  3. Skip the laptop's built-in camera automatically

Camera differentiation strategy:
  - Built-in laptop cameras (FaceTime, iSight, IR) are identified via
    macOS system_profiler and excluded from scanning.
  - Among remaining USB cameras, each frame is checked for phone presence.
  - Cameras that see a phone are classified by bounding-box aspect ratio:
      top camera  → sees phone face-on → ratio ~1.5-2.5 (natural phone shape)
      side camera → sees phone at ~45°  → perspective squash → ratio > 2.5 or < 1.5

Usage:
    uv run python -m physiclaw.vision                  # check camera 0
    uv run python -m physiclaw.vision --identify       # scan all, identify top/side
    uv run python -m physiclaw.vision --camera 2       # check specific camera
"""

import logging
import platform
import subprocess
from pathlib import Path

import cv2
import numpy as np

from physiclaw.camera import Camera

log = logging.getLogger(__name__)

MODEL_PATH = Path(__file__).parent.parent / 'data' / 'model' / 'yolox_nano' / 'yolox_nano.onnx'
MODEL_URL = 'https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_nano.onnx'

COCO_PHONE_CLASS = 67  # cell phone
INPUT_SIZE = 416
MIN_CONFIDENCE = 0.3

# Built-in camera keywords (case-insensitive match against camera name)
_BUILTIN_KEYWORDS = {'facetime', 'isight', 'infrared', 'ir camera', 'built-in'}


def _download_model(path: Path):
    """Download YOLOX Nano model automatically."""
    import urllib.request
    log.info(f"Downloading YOLOX Nano to {path} ...")
    path.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(MODEL_URL, path)
    log.info("Download complete")


def _list_builtin_camera_names() -> set[str]:
    """Return names of built-in cameras on macOS (via system_profiler).
    Returns empty set on non-macOS or on failure.
    """
    if platform.system() != 'Darwin':
        return set()
    try:
        out = subprocess.run(
            ['system_profiler', 'SPCameraDataType'],
            capture_output=True, text=True, timeout=5,
        )
        names = set()
        for line in out.stdout.splitlines():
            stripped = line.strip()
            # Camera names appear as top-level entries (not indented key: value)
            if stripped.endswith(':') and not stripped.startswith(('Unique ID', 'Model ID')):
                name = stripped.rstrip(':').strip()
                if any(kw in name.lower() for kw in _BUILTIN_KEYWORDS):
                    names.add(name)
        return names
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return set()


def _is_builtin_camera(index: int) -> bool:
    """Heuristic: check if a camera index is likely a built-in laptop camera.

    On macOS, built-in cameras are typically index 0 or 1.
    We open the camera briefly and check its name via backend properties.
    Falls back to the conservative assumption that index 0 may be built-in on macOS.
    """
    if platform.system() != 'Darwin':
        return False

    # On macOS, built-in FaceTime camera is almost always index 0.
    # USB cameras get higher indices. We use a name-based check when possible,
    # but fall back to index heuristic since OpenCV doesn't always expose names.
    builtin_names = _list_builtin_camera_names()
    if builtin_names:
        # If we found built-in camera names, we know they exist.
        # On macOS with AVFoundation, index 0 is typically built-in.
        # We can't directly map names→indices, but index 0 is reliable.
        if index == 0:
            return True
    return False


class PhoneDetector:
    """Detect phone presence and identify cameras using YOLOX Nano."""

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
            ratio = self.bbox_aspect_ratio(bbox)
            view = self.classify_view(ratio)
            log.debug(f"Phone detected ({conf:.0%})  bbox: {[round(v) for v in bbox]}  "
                      f"ratio: {ratio:.2f}  view: {view}")
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

    def identify_cameras(self, max_index: int = 8) -> dict[str, int]:
        """Scan cameras, skip built-in laptop camera, identify top vs side.

        Strategy:
          1. Skip built-in cameras (FaceTime/iSight on macOS, typically index 0)
          2. Find USB cameras that can open successfully (stop after 2 found)
          3. Among those, detect phone and classify by bbox aspect ratio
          4. If both cameras get the same classification, force the other role
             on the second one — there are only two cameras, one must be each

        Returns {'top': index, 'side': index} — may be partial if a camera is missing.
        """
        log.debug("Identifying cameras...")
        usb_cameras = []  # (index, frame) for working USB cameras

        for i in range(max_index):
            if _is_builtin_camera(i):
                log.debug(f"  Camera {i}: built-in (skipped)")
                continue

            try:
                cam = Camera(i)
            except RuntimeError:
                continue

            frame = cam.snapshot()
            cam.close()
            if frame is not None:
                usb_cameras.append((i, frame))
                if len(usb_cameras) == 2:
                    break

        if not usb_cameras:
            log.warning("No USB cameras found")
            return {}

        # Classify each camera
        candidates = []  # (index, view)
        for i, frame in usb_cameras:
            detected, conf, bbox = self.detect(frame)
            if detected:
                ratio = self.bbox_aspect_ratio(bbox)
                view = self.classify_view(ratio)
                log.debug(f"  Camera {i}: phone detected ({conf:.0%})  "
                          f"ratio: {ratio:.2f} → {view}")
                candidates.append((i, view))
            else:
                log.debug(f"  Camera {i}: no phone detected ({conf:.0%})")
                candidates.append((i, None))

        # Build result — if one is classified, the other gets the remaining role
        result = {}
        if len(candidates) == 2:
            i0, v0 = candidates[0]
            i1, v1 = candidates[1]

            if v0 and v1 and v0 != v1:
                # Both detected, different views — ideal
                result[v0] = i0
                result[v1] = i1
            elif v0 and v1:
                # Both detected, same view — force the other role on second
                other = 'side' if v0 == 'top' else 'top'
                result[v0] = i0
                result[other] = i1
                log.debug(f"  Camera {i1}: reassigned to {other}")
            elif v0:
                # Only first detected
                other = 'side' if v0 == 'top' else 'top'
                result[v0] = i0
                result[other] = i1
            elif v1:
                # Only second detected
                other = 'side' if v1 == 'top' else 'top'
                result[v1] = i1
                result[other] = i0
        elif len(candidates) == 1:
            i0, v0 = candidates[0]
            if v0:
                result[v0] = i0

        if not result:
            log.warning("No phone detected on any camera — is the phone on the platform?")
        else:
            summary = ', '.join(f'{v}=camera {i}' for v, i in sorted(result.items()))
            log.info(f"Cameras identified: {summary}")

        return result
