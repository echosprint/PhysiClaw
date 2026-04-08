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
from datetime import datetime
from pathlib import Path

import cv2

SNAPSHOT_DIR = Path(__file__).parent.parent.parent / "data" / "snapshot"

log = logging.getLogger(__name__)


def _ensure_camera_permission():
    """On macOS, OpenCV won't trigger the camera permission dialog.
    Run imagesnap once to force the OS prompt, then discard the result."""
    try:
        subprocess.run(
            ["imagesnap", "-w", "0", "/dev/null"],
            capture_output=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass  # imagesnap not installed or hung — skip


# ─── Reusable Camera class ──────────────────────────────────────


class Camera:
    """Persistent camera handle for fast repeated frame grabs.

    Holds the software rotation code to apply to raw frames. Default is
    ``-1`` (no rotation) — calibration step 3 (`pick_frame_rotation`)
    writes the detected ``cv2.ROTATE_*`` code via ``cam.rotation = code``
    once the phone's orientation is known. Callers that need a rotated
    frame should always use ``peek()``/``snapshot()`` rather than calling
    ``cv2.rotate`` themselves.
    """

    def __init__(self, index=0):
        self.index = index
        self.rotation: int = -1  # no rotation until calibration step 3 sets it
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
        """Flush buffered frames and return the latest one (raw, unrotated)."""
        for _ in range(4):
            self.cap.grab()
        ret, frame = self.cap.read()
        if not ret or frame is None:
            return None
        return frame

    def _rotate(self, frame):
        """Apply ``self.rotation`` to a raw frame. No-op when rotation is -1."""
        if self.rotation == -1:
            return frame
        return cv2.rotate(frame, self.rotation)

    def peek(self):
        """Return a fresh BGR frame with the calibrated rotation applied.

        Used for high-frequency polling (e.g. the phone-watch runtime) where
        writing a JPEG to disk every tick would be wasteful.
        """
        frame = self._fresh_frame()
        if frame is None:
            return None
        return self._rotate(frame)

    def snapshot(self, bbox=None):
        """Return a fresh BGR frame with the calibrated rotation applied,
        and save it to ``data/snapshot/`` with a timestamp.

        If ``bbox`` is provided as ``((x1,y1), (x2,y2))``, a green rectangle
        is drawn on the saved frame.
        """
        frame = self.peek()
        if frame is None:
            return None
        if bbox is not None:
            cv2.rectangle(frame, bbox[0], bbox[1], (0, 255, 0), 2)
        SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        cv2.imwrite(str(SNAPSHOT_DIR / f"{ts}.jpg"), frame)
        return frame

    def close(self):
        self.cap.release()
