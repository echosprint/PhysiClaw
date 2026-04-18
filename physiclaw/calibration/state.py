"""Calibration — the typed container for all values learned during setup.

One object, one source of truth. Each field is ``None`` until its step runs.
``complete`` flips True when every required field is filled, and
``transforms()`` returns the :class:`ScreenTransforms` used by MCP tools.

Hardware-side mutable state (``arm.Z_DOWN``, ``arm.MOVE_DIRECTIONS``,
``cam.rotation``) is still set imperatively by each step, but its source
of truth lives here.
"""

from __future__ import annotations

import dataclasses

import cv2
import numpy as np

from physiclaw.calibration.transforms import ScreenTransforms, ViewportShift


ROTATION_NAMES: dict[int, str] = {
    -1: "none",
    cv2.ROTATE_90_CLOCKWISE: "90° CW",
    cv2.ROTATE_180: "180°",
    cv2.ROTATE_90_COUNTERCLOCKWISE: "90° CCW",
}

DEFAULT_ROTATION: int = cv2.ROTATE_90_COUNTERCLOCKWISE


@dataclasses.dataclass
class Calibration:
    viewport_shift: ViewportShift | None = None
    z_tap: float | None = None
    cam_rotation: int | None = None               # cv2 rotation code: -1, 0, 1, 2
    pct_to_grbl: np.ndarray | None = None         # 2×3 affine: screen 0-1 → arm mm
    pct_to_cam: np.ndarray | None = None          # 2×3 affine: screen 0-1 → camera 0-1
    cam_size: tuple[int, int] | None = None       # (width, height) in camera pixels

    @property
    def transforms_ready(self) -> bool:
        """Cheap existence check — true iff transforms() would return non-None."""
        return (
            self.pct_to_grbl is not None
            and self.pct_to_cam is not None
            and self.cam_size is not None
        )

    @property
    def complete(self) -> bool:
        """True when every field (including pre-req ones) is set."""
        return (
            self.viewport_shift is not None
            and self.z_tap is not None
            and self.cam_rotation is not None
            and self.transforms_ready
        )

    def transforms(self) -> ScreenTransforms | None:
        """Build a ScreenTransforms if arm + camera mappings are both set."""
        if not self.transforms_ready:
            return None
        return ScreenTransforms(
            pct_to_grbl=self.pct_to_grbl,
            pct_to_cam=self.pct_to_cam,
            cam_size=self.cam_size,
        )

    def summary(self) -> dict:
        """Per-step status for /api/status — one line per filled field."""
        out: dict = {}
        if self.z_tap is not None:
            out["z_tap"] = f"{self.z_tap}mm"
        if self.viewport_shift is not None:
            t = self.viewport_shift
            out["viewport_shift"] = f"dpr={t.dpr}, offset=({t.offset_x}, {t.offset_y})"
        if self.cam_rotation is not None:
            out["rotation"] = ROTATION_NAMES.get(
                self.cam_rotation, str(self.cam_rotation)
            )
        if self.pct_to_grbl is not None:
            out["mapping_a"] = "OK"
        if self.pct_to_cam is not None:
            out["mapping_b"] = "OK"
        if self.transforms_ready:
            out["validated"] = True
        return out

    def effective_rotation(self) -> int:
        """Rotation code for camera frame processing; falls back to DEFAULT_ROTATION."""
        return self.cam_rotation if self.cam_rotation is not None else DEFAULT_ROTATION
