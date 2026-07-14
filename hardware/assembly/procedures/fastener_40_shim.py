"""Shim — fastener reference shape.

Renders a single M5 Ø8×0.5 shim washer on its own as a hardware
reference icon for the manual's HARDWARE — REFERENCE page (page 4).

An exhibit, not an install step — no install motion, so the exploded
and assembled variants are identical; ``views`` keeps only the
assembled drawing (cf. belt_11_clamp_show).

Run from the repo root:

    uv run --group cad python -m hardware step fastener_40_shim
"""

from build123d import Compound

from hardware.assembly.base import ASSEMBLED_CAM1, BaseAssembly
from hardware.assembly.projection import ISO, Camera
from hardware.parts.standard.ring import Ring


class FA40Shim(BaseAssembly):
    camera = [ISO, Camera(96.55, 39.21, -120.96)]
    views = [ASSEMBLED_CAM1]

    def _build(self) -> Compound:
        shim = Ring("M5x8x0.5").build()
        return Compound(label="fastener_40_shim", children=[shim])
