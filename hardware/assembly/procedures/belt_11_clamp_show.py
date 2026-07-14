"""Belt clamp — standalone part show.

Renders the bare BeltClamp custom part on its own (no slider, no
belt, no screws), so the manual can show the clamp in isolation
before belt_20_clamp mounts it to the X-rail slider.

Unlike a mounting step there is no install motion, so the exploded
and assembled variants are identical — ``views`` keeps only the
assembled drawings.

See belt_20_clamp for the part's placement on the slider.

Run from the repo root:

    uv run --group cad python -m hardware step belt_11_clamp_show
"""

from build123d import Compound

from hardware.assembly.base import ASSEMBLED_CAM0, ASSEMBLED_CAM1, BaseAssembly
from hardware.assembly.projection import Camera
from hardware.parts.custom.belt_clamp import BeltClamp


class BE11ClampShow(BaseAssembly):
    camera = [Camera(30.41, -62.04, -162.59), Camera(-16.99, 73.64, 11.14)]
    views = [ASSEMBLED_CAM0, ASSEMBLED_CAM1]

    def _build(self) -> Compound:
        clamp = BeltClamp().build()
        return Compound(label="belt_11_clamp_show", children=[clamp])
