"""Hex driver — fastener reference shape.

Renders a single 2 mm L-shape hex driver (Allen key) on its own as a
hardware reference icon for the manual's HARDWARE — REFERENCE page
(page 4b). The 2 mm driver fits the M3 button- and flat-head sockets
and so sees the most use in the build.

An exhibit, not an install step — no install motion, so the exploded
and assembled variants are identical; ``views`` keeps only the
assembled drawing (cf. belt_11_clamp_show).

Run from the repo root:

    uv run --group cad python -m hardware step fastener_50_hex_driver
"""

from build123d import Compound

from hardware.assembly.base import ASSEMBLED_CAM1, BaseAssembly
from hardware.assembly.projection import ISO, Camera
from hardware.parts.standard.driver import HexDriver


class FA50HexDriver(BaseAssembly):
    camera = [ISO, Camera(127.11, -31.05, 57.85)]
    views = [ASSEMBLED_CAM1]

    def _build(self) -> Compound:
        driver = HexDriver(size="2mm").build()
        return Compound(label="fastener_50_hex_driver", children=[driver])
