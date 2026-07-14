"""Phone bed mounted to the frame — sets the phone_30 bed-on-beams sub-assembly
under the tapz_20_solenoid_mount frame so the 1020 cross-beams' TOP faces meet
the long extrusions' BOTTOM (+Y) face.

In tapz_20 world the two long Extrusion2040 members run along Z (0 → 335) at
X = ±95; the gantry mechanism (solenoid mount / tip) is on the +Y side, so the
frame underside is the +Y face at Y = +EXT_THICKNESS (= +20), with a slot
centered at X = ±95. phone_30's 210 mm beams carry their frame-nuts at X = ±95,
so they line up with those slots.

phone_30 is rotated +90° about X: its beam-top side (phone_30 +Z) swings to -Y
so the beam tops seat up against the long's +Y bottom face (the beams sit just
below it at +Y); the beams stay along X (spanning both longs) and the bed hangs
on the +Y side with the phone facing back (-Y) toward the solenoid mechanism.
Then translated so the beam tops sit at the long bottom face (Y = +20) and the
pair of beams sits near the longs' mid-length (Z = 147.5, 20 mm toward Z=0).

Two variants:
  * exploded — the bed sub-assembly dropped along +Y off the long bottom face.
  * assembled — beam tops seated on the long bottom face.

Run from the repo root:

    uv run --group cad python -m hardware step phone_40_bed_frame
"""

from build123d import Axis, Compound, Location

from hardware.assembly.base import ASSEMBLED_CAM0, BaseAssembly
from hardware.assembly.procedures.frame_10_extrusion_tnut import (
    EXT_THICKNESS,
    LONG_LENGTH,
)
from hardware.assembly.procedures.phone_30_bed_extrusion import PH30BedExtrusion
from hardware.assembly.procedures.tapz_20_solenoid_mount import TZ20SolenoidMount
from hardware.assembly.projection import MAIN_FRAME_VIEW

# The long 2040's 40 mm dimension runs along Y (Y ∈ [-20, +20]). The mechanism
# (solenoid mount, tip) lives on the +Y side, so the frame's underside / bottom
# face is the +Y face at Y = +EXT_THICKNESS (= +20). The beam tops seat there
# and the bed hangs on the +Y side, phone facing back toward the mechanism.
LONG_BOTTOM_Y = +EXT_THICKNESS  # mm — long 2040 bottom (+Y) face, = +20
BED_Z = LONG_LENGTH / 2 - 20  # mm — 20 mm toward the Z=0 end of mid-length (= 147.5)
EXPLODE = 60  # mm — exploded: bed dropped along +Y off the face


class PH40BedFrame(BaseAssembly):
    camera = MAIN_FRAME_VIEW
    views = [ASSEMBLED_CAM0]

    def _build(self) -> Compound:
        base = TZ20SolenoidMount(exploded=False).build()

        # Rotate phone_30 so its beam-top side (phone_30 +Z) points +Y; beams
        # stay along X and the bed faces +Y into the frame. Then seat the beam
        # tops on the long bottom face and center on the longs' length.
        bed = PH30BedExtrusion(exploded=False).build()
        bed = bed.rotate(Axis.X, 90)
        bed.move(Location((0, LONG_BOTTOM_Y, BED_Z)))
        if self.exploded:
            bed.move(Location((0, EXPLODE, 0)))

        return Compound(label="phone_40_bed_frame", children=[base, bed])
