"""
Camera module — reusable Camera class and CLI test utilities.

Usage as library:
    from physiclaw.hardware.camera import Camera
    cam = Camera(index=0)
    frame = cam.snapshot()
    green = cam.is_green()
    cam.close()

Usage as CLI:
    uv run python -m physiclaw.camera              # scan all cameras
    uv run python -m physiclaw.camera --index 0    # live preview (q=quit, s=save)
    uv run python -m physiclaw.camera --snap 0     # save one frame

Note: On macOS, OpenCV won't trigger the camera permission dialog.
If the camera returns blank frames, run `imagesnap` once first to
grant camera access to your terminal app, then retry.
"""

import logging
import subprocess
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

SNAPSHOT_DIR = Path(__file__).parent.parent.parent / 'data' / 'snapshot'

log = logging.getLogger(__name__)


def _ensure_camera_permission():
    """On macOS, OpenCV won't trigger the camera permission dialog.
    Run imagesnap once to force the OS prompt, then discard the result."""
    try:
        subprocess.run(
            ["imagesnap", "-w", "0", "/dev/null"],
            capture_output=True, timeout=5,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass  # imagesnap not installed or hung — skip


# ─── Reusable Camera class ──────────────────────────────────────

class Camera:
    """Persistent camera handle for fast repeated frame grabs."""

    def __init__(self, index=0):
        self.index = index
        self.cap = cv2.VideoCapture(index)

        # If cv2 fails, try triggering macOS permission via imagesnap
        if not self.cap.isOpened():
            _ensure_camera_permission()
            self.cap = cv2.VideoCapture(index)

        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera index {index}")

        # Warmup: discard initial auto-exposure frames
        for _ in range(15):
            ret, _ = self.cap.read()

        # Verify we can actually read frames (permission may be denied silently)
        ret, frame = self.cap.read()
        if not ret or frame is None:
            self.cap.release()
            _ensure_camera_permission()
            self.cap = cv2.VideoCapture(index)
            for _ in range(15):
                self.cap.read()
            frame = self._read()

        h, w = frame.shape[:2]
        log.info(f"Camera {index} ready  ({w}x{h})")

    def _read(self):
        """Read a single frame, raise on failure."""
        ret, frame = self.cap.read()
        if not ret or frame is None:
            raise RuntimeError(f"Camera {self.index}: read failed")
        return frame

    def _fresh_frame(self):
        """Flush buffered frames and return the latest one."""
        for _ in range(4):
            self.cap.grab()
        ret, frame = self.cap.read()
        if not ret or frame is None:
            return None
        return frame

    def snapshot(self, bbox=None):
        """Return a fresh BGR frame, rotated to portrait orientation.

        Auto-saves to data/snapshot/ with timestamp.
        If bbox is provided as ((x1,y1), (x2,y2)), draws a green rectangle
        on the frame before saving.
        """
        frame = self._fresh_frame()
        if frame is None:
            return None
        frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        if bbox is not None:
            cv2.rectangle(frame, bbox[0], bbox[1], (0, 255, 0), 2)
        SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
        cv2.imwrite(str(SNAPSHOT_DIR / f'{ts}.jpg'), frame)
        return frame

    def is_green(self):
        """Check if the phone screen is showing a green flash (#22c55e)."""
        frame = self._fresh_frame()
        if frame is None:
            return False
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # #22c55e ≈ HSV(145°, 83%, 77%) → OpenCV scale H=72, S=212, V=196
        lower = np.array([35, 50, 50])
        upper = np.array([90, 255, 255])
        mask = cv2.inRange(hsv, lower, upper)
        ratio = np.count_nonzero(mask) / mask.size
        return ratio > 0.05

    def wait_for_green(self, timeout=1.5):
        """Poll for green screen within timeout."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.is_green():
                return True
            time.sleep(0.05)
        return False

    def wait_for_white(self, timeout=3.0):
        """Wait until the green flash clears."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            if not self.is_green():
                return True
            time.sleep(0.1)
        return False

    def close(self):
        self.cap.release()
