"""
Camera module — reusable Camera class and CLI test utilities.

Usage as library:
    from physiclaw.hardware.camera import Camera
    cam = Camera(index=0)
    frame = cam.snapshot()
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
import threading
import time
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

    A background daemon thread continuously calls ``cap.read()`` so the
    macOS AVFoundation pipeline never goes idle (cv2 stalls indefinitely
    on the next read after tens of seconds of inactivity — see opencv
    issue #24393). Callers get the latest frame via ``peek()`` /
    ``snapshot()`` / ``_fresh_frame()`` without blocking on cv2.

    Holds the software rotation code applied to raw frames. Default is
    ``-1`` (no rotation) — calibration step 3 (`pick_frame_rotation`)
    writes the detected ``cv2.ROTATE_*`` code via ``cam.rotation = code``.
    Callers that need a rotated frame should always use
    ``peek()``/``snapshot()`` rather than calling ``cv2.rotate`` themselves.
    """

    # If the reader gets no frame for this long, force close+reopen of
    # cv2.VideoCapture. Recovers from real disconnects.
    STALE_RECONNECT_SECONDS = 5.0

    # Max time _fresh_frame() waits for the reader to produce a frame
    # before returning whatever it last had (or None).
    FRAME_WAIT_SECONDS = 2.0

    # A frame older than this is treated as "not yet fresh" by
    # _fresh_frame() — it'll wait for the reader to publish a newer one.
    FRESH_MAX_AGE_SECONDS = 1.0

    # Backoff after a failed read or reconnect, so a permanently broken
    # camera doesn't spin-loop the reader thread.
    READER_BACKOFF_SECONDS = 0.5

    def __init__(self, index=0):
        self.index = index
        self.rotation: int = -1  # no rotation until calibration step 3 sets it
        self._frame = None
        self._frame_time = 0.0
        self._cond = threading.Condition()
        self._stopped = threading.Event()

        self._open()
        self._warmup()

        self._thread = threading.Thread(
            target=self._reader_loop,
            name=f"Camera-{index}-reader",
            daemon=True,
        )
        self._thread.start()

    # ─── cv2 lifecycle ──────────────────────────────────────────

    def _open(self):
        """Open the underlying ``cv2.VideoCapture``. Retries with macOS perm prompt."""
        self.cap = cv2.VideoCapture(self.index)
        if not self.cap.isOpened():
            _ensure_camera_permission()
            self.cap = cv2.VideoCapture(self.index)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera index {self.index}")
        # AVFoundation (macOS) ignores this; V4L (Linux) honors it.
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    def _warmup(self):
        """Discard initial auto-exposure frames and verify reads work."""
        for _ in range(2):
            for _ in range(15):
                self.cap.read()
            ret, frame = self.cap.read()
            if ret and frame is not None:
                h, w = frame.shape[:2]
                log.info(f"Camera {self.index} ready  ({w}x{h})")
                with self._cond:
                    self._frame = frame
                    self._frame_time = time.monotonic()
                return
            # Read returned no frame — likely macOS perm denied silently.
            self.cap.release()
            _ensure_camera_permission()
            self._open()
        raise RuntimeError(f"Camera {self.index}: read failed")

    def _reopen(self):
        """Close and reopen the cap. Called by the reader on stale frames."""
        log.warning(f"Camera {self.index}: reconnecting cv2.VideoCapture")
        try:
            self.cap.release()
        except Exception:
            pass
        try:
            self._open()
            with self._cond:
                self._frame_time = time.monotonic()  # reset stale clock
        except Exception as e:
            log.error(f"Camera {self.index}: reopen failed: {e!r}")

    # ─── Background reader ──────────────────────────────────────

    def _reader_loop(self):
        """Pull frames continuously so AVFoundation never goes idle.

        ``cap.read()`` blocks for the next native-FPS frame, so the loop
        self-paces — no explicit sleep needed in the steady state.
        """
        while not self._stopped.is_set():
            try:
                ok, frame = self.cap.read()
            except Exception as e:
                log.warning(f"Camera {self.index}: cap.read() raised {e!r}")
                self._reopen()
                self._stopped.wait(self.READER_BACKOFF_SECONDS)
                continue
            now = time.monotonic()
            if ok and frame is not None:
                with self._cond:
                    self._frame = frame
                    self._frame_time = now
                    self._cond.notify_all()
            else:
                with self._cond:
                    stale = now - self._frame_time
                if stale > self.STALE_RECONNECT_SECONDS:
                    self._reopen()
                    self._stopped.wait(self.READER_BACKOFF_SECONDS)

    # ─── Frame accessors ────────────────────────────────────────

    def _fresh_frame(self):
        """Return the latest raw (unrotated) BGR frame, or ``None``.

        Waits up to ``FRAME_WAIT_SECONDS`` for the reader to publish a
        frame fresher than ``FRESH_MAX_AGE_SECONDS``; otherwise returns
        whatever the reader last had.
        """
        with self._cond:
            self._cond.wait_for(
                lambda: (
                    self._frame is not None
                    and time.monotonic() - self._frame_time
                    < self.FRESH_MAX_AGE_SECONDS
                ),
                timeout=self.FRAME_WAIT_SECONDS,
            )
            frame = self._frame
        # Copy outside the lock — a 1080p numpy copy is ~1 ms and would
        # otherwise stall the reader's next publish for that long.
        return frame.copy() if frame is not None else None

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
        self._stopped.set()
        if self._thread.is_alive():
            self._thread.join(timeout=3.0)
        self.cap.release()
