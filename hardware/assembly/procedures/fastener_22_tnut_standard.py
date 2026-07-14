"""Standard T-nut — fastener reference shape.

Renders a single 2020 standard (slide-in) T-slot nut on its own as a
hardware reference icon for the manual's HARDWARE — REFERENCE page
(page 4).

An exhibit, not an install step — no install motion, so the exploded
and assembled variants are identical; ``views`` keeps only the
assembled drawing (cf. belt_11_clamp_show).

Run from the repo root:

    uv run --group cad python -m hardware step fastener_22_tnut_standard
"""

from build123d import Compound

from hardware.assembly.base import ASSEMBLED_CAM1, BaseAssembly
from hardware.assembly.projection import ISO, Camera
from hardware.parts.standard.t_nut import TNut


class FA22TNutStandard(BaseAssembly):
    camera = [ISO, Camera(-146.86, -20.29, 53.79)]
    views = [ASSEMBLED_CAM1]

    def _build(self) -> Compound:
        tnut = TNut("standard", "M5").build()
        return Compound(label="fastener_22_tnut_standard", children=[tnut])
