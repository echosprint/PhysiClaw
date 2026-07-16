"""Solenoid + tip — stylus tip pressed onto the solenoid's bottom rod.

Two variants:

  * exploded — two-layer drawing. Solid: solenoid + tip in its prep
               position, dropped along -Z below the rod (slides UP to
               install). Ghost (dashed): the tip at its seated
               destination on the rod tip.
  * assembled — single-layer drawing of the finished state: solenoid
                + tip mated at the rod.

The rod's lower 4 mm is grooved (thread-like) and grips the tip's M3
hole bore until the rod tip bottoms in the hole.

Run from the repo root:

    uv run --group cad python -m hardware step tapz_10_solenoid_tip
"""

from build123d import Compound, Location

from hardware.assembly.base import (
    ASSEMBLED_CAM1,
    EXPLODED_CAM1,
    GHOST_LABEL,
    SOLID_LABEL,
    BaseAssembly,
)
from hardware.assembly.projection import FRONT_LEFT_LOW_R70, Camera
from hardware.parts.standard.solenoid import Solenoid
from hardware.parts.standard.tip import Tip

PREP_OFFSET_Z = -25  # mm — prep tip dropped below the seated position


class TZ10SolenoidTip(BaseAssembly):
    camera = [FRONT_LEFT_LOW_R70, Camera(-36.53, 5.28, 3.80)]
    views = [EXPLODED_CAM1, ASSEMBLED_CAM1]

    def _build(self) -> Compound:
        solenoid = Solenoid().build()
        seated = Tip().build()
        solenoid.joints["tip_mount"].connect_to(seated.joints["solenoid_mount"])

        if not self.exploded:
            return Compound(label="tapz_10_solenoid_tip", children=[solenoid, seated])

        # Exploded: a second tip in prep position becomes the solid layer;
        # the seated tip becomes the ghost destination marker.
        prep = Tip().build()
        solenoid.joints["tip_mount"].connect_to(prep.joints["solenoid_mount"])
        prep.move(Location((0, 0, PREP_OFFSET_Z)))
        return Compound(
            label="tapz_10_solenoid_tip",
            children=[
                Compound(label=SOLID_LABEL, children=[solenoid, prep]),
                Compound(label=GHOST_LABEL, children=[seated]),
            ],
        )
