"""
Calibration results — `ScreenTransforms` (and the `ViewportShift` re-export).

- `ScreenTransforms` holds the final affine matrices produced by the
  calibration plan: screen 0-1 → GRBL mm and screen 0-1 →
  camera 0-1. These enable coordinate-based tapping.

- `ViewportShift` (produced by `measure_viewport_shift()` in
  `viewport.py`) now lives in `physiclaw.core.geometry` — the
  shared render↔measure contract both bridge and calibration import —
  and is re-exported here so existing importers keep working.

Hardware-orchestration helpers that *use* these transforms (like
edge-trace verification) live in the calibration step modules —
see the `calibrate.py` facade for the map.
"""

import dataclasses

import numpy as np

# Re-exported from core.geometry (its true home) so existing
# importers — `from ...transforms import ViewportShift` — keep working.
# The redundant alias marks it an intentional re-export.
from physiclaw.core.geometry import ViewportShift as ViewportShift
from physiclaw.core.geometry import apply_affine, invert_affine

# The canonical off-phone resting spot, in screen-pct (0-1; off-phone values
# allowed): left of the screen, slightly above the top edge. The arm parks
# here between every operation, on clean shutdown, and warm-start re-pins the
# origin from it on reconnect. Single source of truth for every off-phone park.
PARK_PCT = (-0.1, -0.05)


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
    ) -> None:
        self.pct_to_grbl = pct_to_grbl
        self.pct_to_cam = pct_to_cam
        self.cam_size = cam_size
        self._cam_to_pct: np.ndarray | None = None  # lazy inverse cache

    def bbox_center_pct(self, bbox: list[float]) -> tuple[float, float]:
        """Compute center of a bounding box in screen coordinates (0-1)."""
        return ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)

    def swipe_end_pct(
        self, bbox: list[float], direction: str, dist: float
    ) -> tuple[float, float]:
        """Compute swipe end point from bbox center + direction + distance.

        Returns end point in screen 0-1, clamped to stay on screen.
        direction: 'up' | 'down' | 'left' | 'right' — stylus motion.
        dist: screen-fraction distance (e.g. 0.1, 0.3, 0.5).
        """
        cx, cy = self.bbox_center_pct(bbox)
        if direction == "up":
            ex, ey = cx, cy - dist
        elif direction == "down":
            ex, ey = cx, cy + dist
        elif direction == "left":
            ex, ey = cx - dist, cy
        elif direction == "right":
            ex, ey = cx + dist, cy
        else:
            raise ValueError(f"direction must be up/down/left/right, got {direction!r}")
        return (max(0.0, min(1.0, ex)), max(0.0, min(1.0, ey)))

    def pct_to_grbl_mm(self, x: float, y: float) -> tuple[float, float]:
        """Convert screen coordinate (0-1) to GRBL mm."""
        return apply_affine(self.pct_to_grbl, x, y)

    def pct_to_cam_pixel(self, x: float, y: float) -> tuple[int, int]:
        """Convert screen coordinate (0-1) to camera pixel."""
        cx, cy = apply_affine(self.pct_to_cam, x, y)
        w, h = self.cam_size
        return (int(cx * w), int(cy * h))

    def cam_to_pct(self) -> np.ndarray:
        """The (2, 3) inverse of Mapping B: camera 0-1 → screen 0-1.
        Cached — the mappings are fixed at construction, and OCR calls
        `pixel_to_pct` twice per detected text element."""
        if self._cam_to_pct is None:
            self._cam_to_pct = invert_affine(self.pct_to_cam)
        return self._cam_to_pct

    def pixel_to_pct(self, px_x: int, px_y: int) -> tuple[float, float]:
        """Convert camera pixel to screen coordinate (0-1)."""
        w, h = self.cam_size
        return apply_affine(self.cam_to_pct(), px_x / w, px_y / h)

    def bbox_to_pixel_rect(
        self, bbox: list[float]
    ) -> tuple[tuple[int, int], tuple[int, int]]:
        """Convert bbox [left, top, right, bottom] (0-1) to camera pixel rectangle."""
        tl = self.pct_to_cam_pixel(bbox[0], bbox[1])
        br = self.pct_to_cam_pixel(bbox[2], bbox[3])
        return (tl, br)
