"""Phone-bed mounting hardware — sub-assembly of the PhoneBed plate with the
FHCS + hammer T-nut at each of its four ears.

Each ear has an M3 countersunk through-hole, recessed on the TOP (phone-resting)
face. At each ear:

  * an FHCS M3 × 8 seats flush in the top recess and points DOWN — its conical
    head sits in the countersink, the shank passes down through the ear and
    protrudes below the plate.
  * a hammer M3 T-nut sits below, MOUTH UP (boss toward the plate) to receive
    the screw shank. In the next step this nut slides into the upward-facing
    top slot of a cross-beam and the screw clamps the bed down onto it.

Geometry in the PhoneBed native frame (plate in XY, z = 0 (bottom) → thickness
(top, phone-resting face); ears at (±ear_x_center, ±ear_y_center)).

Two variants:
  * exploded — screws lifted above the bed and nuts dropped below it, along the
               shared vertical (Z) install axis.
  * assembled — screws seated flush in the recesses, nuts under the ears on the
                protruding shanks.

Run from the repo root:

    uv run --group cad python -m hardware step phone_20_bed_tnut
"""

from build123d import Compound, Location

from hardware.assembly.base import ASSEMBLED_CAM0, EXPLODED_CAM1, BaseAssembly
from hardware.assembly.projection import FRONT_LEFT_HIGH, Camera
from hardware.parts.custom.phone_bed import (
    PhoneBed,
    ear_x_center,
    ear_y_center,
    thickness,
)
from hardware.parts.standard.screw import FHCS_DIMS, Screw, head_skirt
from hardware.parts.standard.t_nut import HAMMER_TOTAL_HEIGHT, LENGTHS, TNut

FHCS_LENGTH = 8  # mm — FHCS M3 overall length
SCREW_EXPLODE = 25  # mm — exploded: screw lifts above the bed
NUT_EXPLODE = 30  # mm — exploded: nut drops below the bed

# The four ear hole centers (CounterSinkHole positions in PhoneBed).
EAR_POSITIONS = [
    (sx * ear_x_center, sy * ear_y_center) for sx in (-1, 1) for sy in (-1, 1)
]

# Z of the FHCS local origin (cone base) so the head's flat top lands flush
# with the bed top face. FHCS head spans local z = 0 → k + head_skirt.
SCREW_SEAT_Z = thickness - (FHCS_DIMS["M3"]["k"] + head_skirt)


class PH20BedTnut(BaseAssembly):
    camera = [FRONT_LEFT_HIGH, Camera(-5.56, 49.21, 4.80)]
    views = [EXPLODED_CAM1, ASSEMBLED_CAM0]

    def _build(self) -> Compound:
        bed = PhoneBed().build()

        hammer_len = LENGTHS["hammer"]
        screws, nuts = [], []
        for x, y in EAR_POSITIONS:
            # FHCS in its native orientation (head +Z up, shank -Z down): the
            # conical head seats in the top recess, shank drops through the ear.
            screw = Screw("FHCS", "M3", FHCS_LENGTH).build()
            sz = SCREW_SEAT_Z + (SCREW_EXPLODE if self.exploded else 0)
            screw.move(Location((x, y, sz)))
            screws.append(screw)

            # Hammer T-nut below, mouth up: rotation (90, 0, 0) lays the nut's
            # native +Y bore along world +Z (boss toward the plate). Translate
            # so the mouth sits at the plate underside (z = 0) and the bore
            # lands on the screw axis (native bore at z = length / 2).
            nut = TNut("hammer", "M3").build()
            nz = -HAMMER_TOTAL_HEIGHT - (NUT_EXPLODE if self.exploded else 0)
            nut.move(Location((x, y + hammer_len / 2, nz), (90, 0, 0)))
            nuts.append(nut)

        return Compound(label="phone_20_bed_tnut", children=[bed, *screws, *nuts])
