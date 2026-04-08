"""
Calibration results — `ScreenTransforms` and `ViewportShift`.

Two dataclasses live here, both pure data + coordinate math:

- `ViewportShift` is produced by the pre-calibration step
  (`measure_viewport_shift()` in `calibrate.py`). It maps viewport CSS
  pixels to screenshot 0-1 coordinates, accounting for device pixel
  ratio and the iOS status-bar / safe-area offset.

- `ScreenTransforms` holds the final affine matrices produced by the
  7-step calibration plan: screen 0-1 → GRBL mm and screen 0-1 →
  camera 0-1. These enable coordinate-based tapping.

Hardware-orchestration helpers that *use* these transforms (like
edge-trace verification) live in `calibrate.py`.
"""

import dataclasses

import numpy as np


@dataclasses.dataclass(frozen=True)
class ViewportShift:
    """Viewport CSS pixels → screenshot 0-1 coordinates.

    Produced by the pre-calibration step: server shows an orange square at
    a known CSS position, user uploads a screenshot, server detects the
    square and derives (dpr, offset_x, offset_y) from the mismatch.

    Fields:
        offset_x, offset_y: screenshot-pixel offset caused by iOS status
            bar / safe area (viewport origin is not at screenshot origin).
        dpr: device pixel ratio — CSS px → screenshot px scale factor.
        screenshot_width, screenshot_height: screenshot image size in px.
    """

    offset_x: float
    offset_y: float
    dpr: float
    screenshot_width: int
    screenshot_height: int

    def css_to_pct(self, css_x: float, css_y: float) -> tuple[float, float]:
        """Convert viewport CSS pixel coords to screenshot 0-1 coords."""
        sx = (css_x * self.dpr + self.offset_x) / self.screenshot_width
        sy = (css_y * self.dpr + self.offset_y) / self.screenshot_height
        return (sx, sy)


@dataclasses.dataclass
class ScreenTransforms:
    """Stores and applies grid calibration affine transforms.

    All coordinates use 0-1 decimals (0=left/top, 1=right/bottom).
    Both mappings work in normalized space:
      - pct_to_grbl:  screen 0-1 → GRBL mm
      - pct_to_cam:   screen 0-1 → camera 0-1
    Camera pixel conversion happens at the boundary via cam_size.
    """

    pct_to_grbl: np.ndarray  # (2, 3) screen 0-1 → GRBL mm
    pct_to_cam: np.ndarray  # (2, 3) screen 0-1 → camera 0-1
    cam_size: tuple[int, int]  # (width, height) of camera frame in pixels

    def __init__(
        self,
        pct_to_grbl: np.ndarray,
        pct_to_cam: np.ndarray,
        cam_size: tuple[int, int] = (1920, 1080),
    ):
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
        b = self.pct_to_cam[:, 2]  # translation
        pct = np.linalg.solve(A, cam_01 - b)
        return (float(pct[0]), float(pct[1]))

    def bbox_to_pixel_rect(
        self, bbox: list[float]
    ) -> tuple[tuple[int, int], tuple[int, int]]:
        """Convert bbox [left, top, right, bottom] (0-1) to camera pixel rectangle."""
        tl = self.pct_to_cam_pixel(bbox[0], bbox[1])
        br = self.pct_to_cam_pixel(bbox[2], bbox[3])
        return (tl, br)
