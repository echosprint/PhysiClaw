"""Tube-holder mounting hardware — sub-assembly of the TubeHolder with the FHCS +
hammer T-nut at each of its two rib mount holes. The board_10 step for the PCB
holder, repeated for the tube holder.

The holder's underside rib carries two M5 countersink holes (``mount_hole_pitch``
apart, along Y), recessed on the top face. At each hole:

  * an FHCS M5 × 12 seats flush in the top recess and points DOWN — the conical
    head sits in the countersink, the shank passes down through the plate + rib
    and protrudes below the rib.
  * a hammer M5 T-nut sits below the rib, MOUTH UP (boss toward the rib) to
    receive the screw shank. In the next step (board_32) this nut drops into the
    top slot of the frame's 2040 and the screw clamps the holder down onto it.

Geometry in the TubeHolder native frame (plate in XY, z = 0 (bottom) → thickness
(top); rib protrudes -Z to z = -tab_thick; rib holes at (rib_cx, rib_cy ± pitch/2)).
The rib runs along Y — so the hammer nuts (length native +Z) are rotated to lie
along X (the slot run once board_32 seats the rib on the frame). board_32 carries
this whole sub-assembly to the frame by ``TUBE_HOLDER_PLACEMENT``.

Two variants:
  * exploded — screws lifted above the holder and nuts dropped below the rib,
               along the shared vertical (Z) install axis.
  * assembled — screws seated flush, nuts hanging loosely at the end of the
                protruding shanks (snugged to the rib only once bolted to the slot).

Run from the repo root:

    uv run --group cad python -m hardware step board_31_tube_tnut
"""

from build123d import Compound, Location

from hardware.assembly.base import ASSEMBLED_CAM0, EXPLODED_CAM1, BaseAssembly
from hardware.assembly.projection import Camera
from hardware.parts.custom import tube_holder as TH
from hardware.parts.custom.tube_holder import TubeHolder
from hardware.parts.standard.screw import FHCS_DIMS, Screw, head_skirt
from hardware.parts.standard.t_nut import HAMMER_TOTAL_HEIGHT, LENGTHS, TNut

FHCS_LENGTH = 12  # mm — FHCS M5 overall length (plate + rib into the nut)
SCREW_EXPLODE = 25  # mm — exploded: screw lifts above the holder
NUT_EXPLODE = 30  # mm — exploded: nut drops below the rib
NUT_LOOSE = 1.5  # mm — assembled: nut hangs at the shank end (loosely captured,
#      not yet snugged to the rib). The M5×12 shank protrudes
#      6 mm past the rib and the nut is 4.5 mm tall, so 1.5 mm
#      drops it flush with the shank tip — still fully threaded.

# Z of the FHCS local origin (cone base) so the head's flat top lands flush with
# the holder top face. FHCS head spans local z = 0 → k + head_skirt.
SCREW_SEAT_Z = TH.thickness - (FHCS_DIMS["M5"]["k"] + head_skirt)

# The two rib mount-hole centers, along Y (one nut per slot).
HOLE_Y = (TH.rib_cy - TH.mount_hole_pitch / 2, TH.rib_cy + TH.mount_hole_pitch / 2)

# Hammer-nut orientation: bore (native +Y) → world +Z (up toward the rib) and the
# nut length (native +Z) → world +X, so each nut lies along its slot (the rib
# runs along Y here). The cyclic map is Rz(90)·Rx(90).
NUT_ROT = Location((0, 0, 0), (0, 0, 90)) * Location((0, 0, 0), (90, 0, 0))


class BO31TubeTnut(BaseAssembly):
    camera = [Camera(103.68, 34.46, -94.66), Camera(115.65, 54.89, -96.42)]
    views = [EXPLODED_CAM1, ASSEMBLED_CAM0]

    def _build(self) -> Compound:
        holder = TubeHolder().build()

        hammer_len = LENGTHS["hammer"]
        screws, nuts = [], []
        for y in HOLE_Y:
            # FHCS native orientation (head +Z up, shank -Z down): the conical
            # head seats in the top recess, shank drops through plate + rib.
            screw = Screw("FHCS", "M5", FHCS_LENGTH).build()
            sz = SCREW_SEAT_Z + (SCREW_EXPLODE if self.exploded else 0)
            screw.move(Location((TH.rib_cx, y, sz)))
            screws.append(screw)

            # Hammer T-nut below the rib, mouth up (NUT_ROT lays the native +Y
            # bore along +Z and the length along +X). Translate so the length is
            # centered on the screw axis (bore at the nut-length center) and the
            # mouth sits below the rib underside — dropped NUT_LOOSE so it hangs
            # loosely on the shank end (NUT_EXPLODE when exploded).
            nut = TNut("hammer", "M5").build()
            nz = (
                -TH.tab_thick
                - HAMMER_TOTAL_HEIGHT
                - (NUT_EXPLODE if self.exploded else NUT_LOOSE)
            )
            nut.move(Location((TH.rib_cx - hammer_len / 2, y, nz)) * NUT_ROT)
            nuts.append(nut)

        return Compound(label="board_31_tube_tnut", children=[holder, *screws, *nuts])
