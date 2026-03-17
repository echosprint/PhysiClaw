"""
Camera module — reusable Camera class and CLI test utilities.

Usage as library:
    from camera import Camera
    cam = Camera(index=0)
    frame = cam.snapshot()
    green = cam.is_green()
    cam.close()

Usage as CLI:
    uv run python camera.py              # scan all cameras
    uv run python camera.py --index 0    # live preview (q=quit, s=save)
    uv run python camera.py --snap 0     # save one frame

Note: On macOS, OpenCV won't trigger the camera permission dialog.
If the camera returns blank frames, run `imagesnap` once first to
grant camera access to your terminal app, then retry.
"""

import argparse
import subprocess
import sys
import time

import cv2
import numpy as np


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
        print(f"Camera {index} ready  ({w}x{h})")

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
        return frame if ret else None

    def snapshot(self, path=None):
        """Return a fresh BGR frame. Optionally save to path."""
        frame = self._fresh_frame()
        if frame is not None and path:
            cv2.imwrite(path, frame)
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
        return ratio > 0.15

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


# ─── CLI utilities ───────────────────────────────────────────────

def scan_cameras(max_index=10):
    """Try opening camera indices 0..max_index, report what works."""
    print(f"Scanning camera indices 0-{max_index - 1} ...\n")
    found = []
    for i in range(max_index):
        cap = cv2.VideoCapture(i)
        if not cap.isOpened():
            continue
        ret, frame = cap.read()
        if ret and frame is not None:
            h, w = frame.shape[:2]
            backend = cap.getBackendName()
            print(f"  [{i}]  {w}x{h}  backend={backend}")
            found.append(i)
        else:
            print(f"  [{i}]  opened but read() failed (permission?)")
        cap.release()

    if not found:
        print("\nNo working cameras found.")
    else:
        print(f"\nWorking indices: {found}")
    return found


def live_preview(index):
    """Open a live preview window. Press 'q' to quit, 's' to save a frame."""
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        print(f"Cannot open camera {index}")
        sys.exit(1)

    ret, frame = cap.read()
    if not ret:
        print(f"Camera {index} opened but cannot read frames.")
        cap.release()
        sys.exit(1)

    h, w = frame.shape[:2]
    print(f"Camera {index}: {w}x{h}  backend={cap.getBackendName()}")
    print("Live preview — press 'q' to quit, 's' to save a snapshot\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow(f"Camera {index}", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("s"):
            path = f"camera_{index}_snap.jpg"
            cv2.imwrite(path, frame)
            print(f"  Saved {path}")

    cap.release()
    cv2.destroyAllWindows()


def save_snapshot(index):
    """Grab a single frame and save it."""
    cam = Camera(index)
    path = f"camera_{index}_snap.jpg"
    cam.snapshot(path)
    cam.close()
    print(f"Saved {path}")


def main():
    parser = argparse.ArgumentParser(description="Camera test utility")
    parser.add_argument("--index", type=int, default=None,
                        help="Open live preview for this camera index")
    parser.add_argument("--snap", type=int, default=None,
                        help="Save one frame from this camera index")
    args = parser.parse_args()

    print(f"OpenCV {cv2.__version__}\n")

    if args.snap is not None:
        save_snapshot(args.snap)
    elif args.index is not None:
        live_preview(args.index)
    else:
        scan_cameras()


if __name__ == "__main__":
    main()
