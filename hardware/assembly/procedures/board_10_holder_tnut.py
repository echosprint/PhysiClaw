"""PCB-holder mounting hardware — sub-assembly of the PcbHolder with the FHCS +
hammer T-nut at each of its two rib mount holes.

The holder's underside rib carries two M5 countersink holes (20 mm apart),
recessed on the top (standoff) face. At each hole:

  * an FHCS M5 × 12 seats flush in the top recess and points DOWN — the conical
    head sits in the countersink, the shank passes down through the plate + rib
    and protrudes below the rib.
  * a hammer M5 T-nut sits below the rib, MOUTH UP (boss toward the rib) to
    receive the screw shank. In the next step this nut slides into the top slot
    of a 2040 extrusion and the screw clamps the holder down onto it.

Geometry in the PcbHolder native frame (plate in XY, z = 0 (bottom) → thickness
(top); rib protrudes -Z to z = -tab_thick; rib holes at (rib_cx ± pitch/2, rib_cy)).

Two variants:
  * exploded — screws lifted above the holder and nuts dropped below the rib,
               along the shared vertical (Z) install axis.
  * assembled — screws seated flush, nuts under the rib on the protruding shanks.

Run from the repo root:

    uv run --group cad python -m hardware.assembly.procedures.board_10_holder_tnut
"""

from build123d import Compound, Location

from hardware.assembly.base import BaseAssembly
from hardware.assembly.projection import Camera, FRONT_LEFT_HIGH
from hardware.parts.custom.pcb_holder import (
    PcbHolder,
    mount_hole_pitch,
    rib_cx,
    rib_cy,
    tab_thick,
    thickness,
)
from hardware.parts.standard.screw import FHCS_DIMS, Screw, head_skirt
from hardware.parts.standard.t_nut import HAMMER_TOTAL_HEIGHT, LENGTHS, TNut

FHCS_LENGTH = 12  # mm — FHCS M5 overall length
SCREW_EXPLODE = 25  # mm — exploded: screw lifts above the holder
NUT_EXPLODE = 30  # mm — exploded: nut drops below the rib

# Z of the FHCS local origin (cone base) so the head's flat top lands flush with
# the holder top face. FHCS head spans local z = 0 → k + head_skirt.
SCREW_SEAT_Z = thickness - (FHCS_DIMS["M5"]["k"] + head_skirt)

# The two rib mount-hole centers.
HOLE_X = (rib_cx - mount_hole_pitch / 2, rib_cx + mount_hole_pitch / 2)


class BO10HolderTnut(BaseAssembly):
    camera = [FRONT_LEFT_HIGH, Camera(-174.71, 33.92, -91.31)]

    def _build(self) -> Compound:
        holder = PcbHolder().build()

        hammer_len = LENGTHS["hammer"]
        screws, nuts = [], []
        for x in HOLE_X:
            # FHCS native orientation (head +Z up, shank -Z down): the conical
            # head seats in the top recess, shank drops through plate + rib.
            screw = Screw("FHCS", "M5", FHCS_LENGTH).build()
            sz = SCREW_SEAT_Z + (SCREW_EXPLODE if self.exploded else 0)
            screw.move(Location((x, rib_cy, sz)))
            screws.append(screw)

            # Hammer T-nut below the rib, mouth up: rotation (90, 0, 0) lays the
            # native +Y bore along world +Z (boss toward the rib). Translate so
            # the mouth sits at the rib underside (z = -tab_thick) and the bore
            # lands on the screw axis (native bore at z = length / 2).
            nut = TNut("hammer", "M5").build()
            nz = (
                -tab_thick - HAMMER_TOTAL_HEIGHT - (NUT_EXPLODE if self.exploded else 0)
            )
            nut.move(Location((x, rib_cy + hammer_len / 2, nz), (90, 0, 0)))
            nuts.append(nut)

        return Compound(label="board_10_holder_tnut", children=[holder, *screws, *nuts])


if __name__ == "__main__":
    for exploded in (True, False):
        asm = BO10HolderTnut(exploded=exploded)
        asm.export()
        asm.render()
