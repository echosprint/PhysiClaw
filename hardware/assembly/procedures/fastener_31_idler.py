"""Idler — fastener reference shape.

Renders a single GT2 20-tooth idler wheel on its own as a hardware
reference icon for the manual's HARDWARE — REFERENCE page (page 4).

An exhibit, not an install step — no install motion, so the exploded
and assembled variants are identical; ``views`` keeps only the
assembled drawing (cf. belt_11_clamp_show).

Run from the repo root:

    uv run --group cad python -m hardware step fastener_31_idler
"""

from build123d import Compound

from hardware.assembly.base import ASSEMBLED_CAM1, BaseAssembly
from hardware.assembly.projection import ISO, Camera
from hardware.parts.standard.pulley import Pulley2GT20T


class FA31Idler(BaseAssembly):
    camera = [ISO, Camera(96.55, 39.21, -120.96)]
    views = [ASSEMBLED_CAM1]

    def _build(self) -> Compound:
        idler = Pulley2GT20T(kind="idler", toothed=True).build()
        return Compound(label="fastener_31_idler", children=[idler])
