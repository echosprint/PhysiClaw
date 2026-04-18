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
import json
import logging
from pathlib import Path

import cv2
import numpy as np

from physiclaw.calibration.transforms import ScreenTransforms, ViewportShift

log = logging.getLogger(__name__)


ROTATION_NAMES: dict[int, str] = {
    -1: "none",
    cv2.ROTATE_90_CLOCKWISE: "90° CW",
    cv2.ROTATE_180: "180°",
    cv2.ROTATE_90_COUNTERCLOCKWISE: "90° CCW",
}

DEFAULT_ROTATION: int = cv2.ROTATE_90_COUNTERCLOCKWISE

BUNDLE_PATH = Path("data/calibration/bundle.json")


@dataclasses.dataclass
class Calibration:
    viewport_shift: ViewportShift | None = None
    z_tap: float | None = None
    cam_rotation: int | None = None               # cv2 rotation code: -1, 0, 1, 2
    pct_to_grbl: np.ndarray | None = None         # 2×3 affine: screen 0-1 → arm mm
    pct_to_cam: np.ndarray | None = None          # 2×3 affine: screen 0-1 → camera 0-1
    cam_size: tuple[int, int] | None = None       # (width, height) in camera pixels
    cam_index: int | None = None                  # USB camera index (for warm-restart)

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

    # ─── Persistence ────────────────────────────────────────

    def to_dict(self) -> dict:
        """JSON-safe snapshot of this bundle. numpy arrays become nested lists."""
        return {
            "viewport_shift": (
                dataclasses.asdict(self.viewport_shift)
                if self.viewport_shift is not None else None
            ),
            "z_tap": self.z_tap,
            "cam_rotation": self.cam_rotation,
            "pct_to_grbl": (
                self.pct_to_grbl.tolist() if self.pct_to_grbl is not None else None
            ),
            "pct_to_cam": (
                self.pct_to_cam.tolist() if self.pct_to_cam is not None else None
            ),
            "cam_size": list(self.cam_size) if self.cam_size is not None else None,
            "cam_index": self.cam_index,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "Calibration":
        """Reconstruct from the output of to_dict()."""
        vs = payload.get("viewport_shift")
        pg = payload.get("pct_to_grbl")
        pc = payload.get("pct_to_cam")
        cs = payload.get("cam_size")
        return cls(
            viewport_shift=ViewportShift(**vs) if vs is not None else None,
            z_tap=payload.get("z_tap"),
            cam_rotation=payload.get("cam_rotation"),
            pct_to_grbl=np.array(pg, dtype=np.float64) if pg is not None else None,
            pct_to_cam=np.array(pc, dtype=np.float64) if pc is not None else None,
            cam_size=tuple(cs) if cs is not None else None,
            cam_index=payload.get("cam_index"),
        )

    def save(self, path: Path = BUNDLE_PATH) -> None:
        """Write this bundle to disk as JSON."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))
        log.info(f"Saved calibration bundle → {path}")

    @classmethod
    def load(cls, path: Path = BUNDLE_PATH) -> "Calibration | None":
        """Return the bundle at ``path``, or None if missing/unreadable."""
        if not path.exists():
            return None
        try:
            return cls.from_dict(json.loads(path.read_text()))
        except (json.JSONDecodeError, TypeError, ValueError, KeyError) as e:
            log.warning(f"Failed to load calibration bundle from {path}: {e}")
            return None
