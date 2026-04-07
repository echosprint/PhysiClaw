"""
Grid calibration — `GridCalibration` dataclass and edge-trace verification.

`GridCalibration` stores the affine transforms produced by the calibration
plan in `plan_calibrate.py`:
  - pct_to_grbl: screen 0-1 → GRBL mm
  - pct_to_cam:  screen 0-1 → camera 0-1

The transforms enable coordinate-based tapping: the agent specifies a target
as a 0-1 bounding box and the arm moves directly to its center.
"""

import dataclasses
import logging
import time

import numpy as np

from physiclaw.hardware.stylus_arm import StylusArm

log = logging.getLogger(__name__)


@dataclasses.dataclass
class GridCalibration:
    """Stores and applies grid calibration affine transforms.

    All coordinates use 0-1 decimals (0=left/top, 1=right/bottom).
    Both mappings work in normalized space:
      - pct_to_grbl:  screen 0-1 → GRBL mm
      - pct_to_cam:   screen 0-1 → camera 0-1
    Camera pixel conversion happens at the boundary via cam_size.
    """

    pct_to_grbl: np.ndarray    # (2, 3) screen 0-1 → GRBL mm
    pct_to_cam: np.ndarray     # (2, 3) screen 0-1 → camera 0-1
    cam_size: tuple[int, int]  # (width, height) of camera frame in pixels

    def __init__(self, pct_to_grbl: np.ndarray,
                 pct_to_cam: np.ndarray,
                 cam_size: tuple[int, int] = (1920, 1080)):
        self.pct_to_grbl = pct_to_grbl
        self.pct_to_cam = pct_to_cam
        self.cam_size = cam_size

    def bbox_center_pct(self, bbox: list[float]) -> tuple[float, float]:
        """Compute center of a bounding box in screen coordinates (0-1)."""
        return ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)

    def pct_to_grbl_mm(self, x: float, y: float) -> tuple[float, float]:
        """Convert screen coordinate (0-1) to GRBL mm."""
        pt = np.array([x, y, 1.0])
        result = self.pct_to_grbl @ pt
        return (float(result[0]), float(result[1]))

    def pct_to_cam_pixel(self, x: float, y: float) -> tuple[int, int]:
        """Convert screen coordinate (0-1) to camera pixel."""
        pt = np.array([x, y, 1.0])
        cam_01 = self.pct_to_cam @ pt
        w, h = self.cam_size
        return (int(cam_01[0] * w), int(cam_01[1] * h))

    def pixel_to_pct(self, px_x: int, px_y: int) -> tuple[float, float]:
        """Convert camera pixel to screen coordinate (0-1)."""
        w, h = self.cam_size
        cam_01 = np.array([px_x / w, px_y / h])
        A = self.pct_to_cam[:, :2]  # 2x2
        b = self.pct_to_cam[:, 2]   # translation
        pct = np.linalg.solve(A, cam_01 - b)
        return (float(pct[0]), float(pct[1]))

    def bbox_to_pixel_rect(self, bbox: list[float]) -> tuple[tuple[int, int], tuple[int, int]]:
        """Convert bbox [left, top, right, bottom] (0-1) to camera pixel rectangle."""
        tl = self.pct_to_cam_pixel(bbox[0], bbox[1])
        br = self.pct_to_cam_pixel(bbox[2], bbox[3])
        return (tl, br)


# ─── Edge-trace verification ──────────────────────────────────


def trace_screen_edge(arm: StylusArm, cal: GridCalibration):
    """Trace the phone screen border clockwise for visual verification.

    Moves the arm to 8 edge points (top-center → top-right → right-center
    → bottom-right → bottom-center → bottom-left → left-center → top-left
    → back to top-center), pausing 2s at each. Then returns to center.
    """
    check_points = [
        (0.50, 0, "top center"),
        (1, 0, "top right"),
        (1, 0.50, "right center"),
        (1, 1, "bottom right"),
        (0.50, 1, "bottom center"),
        (0, 1, "bottom left"),
        (0, 0.50, "left center"),
        (0, 0, "top left"),
        (0.50, 0, "top center"),  # close the loop
    ]
    arm._fast_move(0, 0)
    arm.wait_idle()
    log.info("Tracing phone edge clockwise...")
    for x_pct, y_pct, label in check_points:
        gx, gy = cal.pct_to_grbl_mm(x_pct, y_pct)
        log.info(f"  → {label} ({x_pct}, {y_pct}) = GRBL ({gx:.2f}, {gy:.2f})")
        arm._fast_move(gx, gy)
        arm.wait_idle()
        time.sleep(2)

    arm._fast_move(0, 0)
    arm.wait_idle()
    log.info("Edge trace done")
