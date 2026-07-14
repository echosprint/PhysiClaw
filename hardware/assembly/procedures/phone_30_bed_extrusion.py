"""Phone bed on its cross-beams — the phone_20 bed sub-assembly set onto two
phone_10 cross-beams so the bed's four hammer T-nuts drop into the beams' top
slots.

World frame (Z up): the bed lies flat in XY with its top (+Z) up — the
PhoneBed native frame, placed at the origin (bottom face at z = 0). Each
cross-beam is laid horizontal with its length along X and its slotted top face
(+native Y) pointing up; the two beams sit at Y = ±ear_y_center so each catches
the bed's two ear nuts at that Y. Beam tops are at z = 0, so the bed rests on
them and its nuts (mouth at z = 0) reach down into the slots.

  beam orientation: native (x, y, z) → world (z, x, y) via rotate X+90 then
  Z+90 — native Z (length) → world X, native Y (height/top) → world Z (up),
  native X (width) → world Y. Then centered in X and dropped so the top is
  at z = 0.

Sub-assemblies are embedded in their assembled form. Two variants:
  * exploded — the whole bed assembly lifted in +Z off the beams.
  * assembled — bed seated on the beams.

Run from the repo root:

    uv run --group cad python -m hardware step phone_30_bed_extrusion
"""

from build123d import Axis, Compound, Location

from hardware.assembly.base import ASSEMBLED_CAM0, BaseAssembly
from hardware.assembly.procedures.phone_10_extrusion import (
    BEAM_LENGTH,
    BEAM_TOP_Y,
    PH10Extrusion,
)
from hardware.assembly.procedures.phone_20_bed_tnut import PH20BedTnut
from hardware.assembly.projection import FRONT_LEFT_HIGH, Camera
from hardware.parts.custom.phone_bed import ear_y_center

BED_EXPLODE = 40  # mm — exploded: bed assembly lifted off the beams in +Z


class PH30BedExtrusion(BaseAssembly):
    camera = [FRONT_LEFT_HIGH, Camera(-5.56, 49.21, 4.80)]
    views = [ASSEMBLED_CAM0]

    def _build(self) -> Compound:
        # Two cross-beams: length along X (centered on X = 0), top face at
        # z = 0, one under each ear row at Y = ±ear_y_center.
        beams = []
        for sy in (-1, 1):
            beam = PH10Extrusion(exploded=False).build()
            beam = beam.rotate(Axis.X, 90).rotate(Axis.Z, 90)
            beam.move(Location((-BEAM_LENGTH / 2, sy * ear_y_center, -BEAM_TOP_Y)))
            beams.append(beam)

        # Bed assembly seated on the beam tops (bottom face at z = 0); its
        # hammer nuts reach down into the slots.
        bed = PH20BedTnut(exploded=False).build()
        if self.exploded:
            bed.move(Location((0, 0, BED_EXPLODE)))

        return Compound(label="phone_30_bed_extrusion", children=[*beams, bed])
